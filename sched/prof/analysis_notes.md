# foreground-store 간섭 원인 분석 — 중간 기록

## 1단계 기준 (계측 없음, n=3)

| 비교군 | p95 | CPU/run | 판정 |
|---|---|---|---|
| GDS-동시저장 | 6.48/5.51/6.04 | ~1,074s | SLOW 3/3 |
| POSIX-동시저장 | 5.44/5.05/5.22 | ~1,270s | SLOW (cuFile 고유 아님) |
| GDS-지연저장 | 1.196/1.196/1.195 | ~90s | FAST 3/3, CV≈0 |
| 기존-tiering | 4.95/5.17/5.19 | ~146s | tail 높음·CPU 낮음 |

## 계측 런 (PROF_INSTRUMENT=1, 패턴 유지 확인)

Dsync p95 5.77 / CPU 1118s, Ddef p95 1.20 / CPU 115s. 단계별 대조(총합은 스레드-시간):

| 구간 | Dsync | Ddef | 비 |
|---|---|---|---|
| store.CUDA_EVENT_WAIT 평균 | 358ms | 0.7ms | 500× |
| store.CUDA_EVENT_WAIT 총합 | 980s | 2s | ≈ CPU 초과분(1003s) |
| cufile.CUFILE_WRITE 평균 | 15.5ms | 15.4ms | 동일 — native write는 안 느려짐 |
| cufile.CUFILE_READ 평균 | 8.8ms | 3.5ms | 2.6× — r/w 경합 |
| load.QUEUE_WAIT 평균 | 1.40s | 0.10s | 14× — TTFT tail 직행 경로 |
| 엔진 step 수 | 124,523 | 7,018 | 17.7× — 빈 step 공회전 |

## nsys (python-gil trace, 패턴 유지: Dsync p95 10.7 / Ddef 1.19)

- 엔진 스레드 GIL 보유 68.6s(480만 회) vs 27.2s — step 공회전 반영.
- store 스레드 GIL 대기는 각 ~6.7s로 크지 않음 — GIL 기근이 store를 굶기는 그림은 약함.
- store 스레드-시간은 native CUFILE_WRITE로 포화(4,641s ≈ 12스레드 상시 가동).
- CUPTI 동기화 레코드: 건수 동일(254,200) / 총 시간 Dsync 1,021s vs Ddef 96s (10.6×)
  — 같은 sync 호출이 foreground와 겹치면 10배 길어짐. CPU 초과분과 일치.

## 현재 우선 가설(격리 실험으로 판정 예정)

1. store 스레드의 CUDA event 동기화가 foreground compute 동안 장기화되고
   그 대기가 CPU를 태우는 방식(스핀)일 때 CPU 12× 폭증을 설명.
2. 요청 종료가 store 완료에 게이트(delay_free)되어 엔진이 빈 step을 고속 공회전
   (830 step/req) — GIL 보유 2.5×, tail 기여.
3. NVMe read/write 경합 — load 2.6× 감속, tail 직행 경로.
cuFile 고유 문제 아님(POSIX-동시저장도 SLOW, CPU 더 높음).

## 3단계 py-spy (25Hz, 패턴 유지)

- GIL 캡처: 엔진 스레드가 GIL 보유 총량의 절반 이상(50s; 최다 함수 vLLM copy_to_uva).
  IO 스레드 16개는 각 2~3s의 짧은 고빈도 보유(560만 회와 정합).
- CPU-활성 캡처(idle 제외): 활성 샘플 1,783 스레드-초 중 87.9%가 ev.synchronize 스택
  — event 대기가 sleep이 아니라 스핀임을 직접 확인. gdslib.write 5.8%, 엔진 4.8%.

## 4단계 격리 (1차, 각 1회 — 경계 반복 진행 중)

| 변형 | 남긴 동작 | p95 | CPU | 판정 |
|---|---|---|---|---|
| 제어부만 | queue·rename·완료 | 1.168 | 19.5s | FAST — Python 제어부·GIL 무죄 |
| CUDA만 | event sync만, IO 제거 | 1.165 | 1,506.8s | CPU 폭증 단독 재현, tail 정상 |
| 네이티브대기 | GIL 놓는 1.28s sleep | 4.354 | 31.9s | tail 단독 재현, CPU 정상 |
| POSIX만 | pwrite만 | 4.337 | 57.1s | tail 재현 |
| cuFile만 | cuFile write만, ev 제거 | 4.312 | 109.7s | tail 재현 (+cuFile CPU 소폭) |

이중 해리: 두 증상의 원인이 분리된다.
- CPU 폭증 = store 스레드의 CUDA event 스핀 대기 단독 (foreground compute가 event
  완료를 늦추는 동안 8~16 스레드가 busy-wait).
- TTFT tail = store 작업의 시간 점유 자체 (transport·CPU·GIL 무관 — 순수 sleep으로
  완전 재현). 기전: store 점유가 요청 경계를 침범 → 블록 delay-free 장기화 →
  다음 요청 admission 지연·빈 step 공회전. 지연 store가 tail을 고치는 이유는
  gap에서 store를 완결시켜 경계 침범을 없애기 때문.

## 4단계 경계 반복 (3/3, 편차 극소)

| 변형 | p95 (r1/r2/r3) | CPU (r1/r2/r3) |
|---|---|---|
| 제어부만 | 1.168/1.163/1.167 | 19.5/19.2/18.8 |
| 네이티브대기 | 4.354/4.334/4.356 | 31.9/31.5/32.4 |
| CUDA만 | 1.165/1.164/1.165 | 1,506.8/1,504.0/1,503.9 |

## 5단계 제거 검증 (FIX=blocking_event, 전체 GDS-동시저장 경로, 3/3)

torch.cuda.Event(blocking=True) 한 줄 차이. 결과: CPU 1,074s → 118.5/122.3/122.8s
(9배 정상화, 3/3) / tail은 잔존 p95 5.149/5.476/5.271 (예측대로 — tail은 event
스핀과 무관). 이중 해리 예측이 제거 실험에서 그대로 실현.

## 원인 확정 (판정 3조건 충족)

1. CPU 12배 폭증 = expfs 구현의 CUDA event 스핀 대기.
   torch.cuda.Event() 기본 플래그의 cudaEventSynchronize busy-wait를 store 스레드
   8~16개가 수행하고, foreground compute가 event 완료를 늦추는 동안 스핀이 길어진다
   (동일 건수 sync가 10.6배 장기화). 증거 4중: 계측 산술 일치·nsys·py-spy 활성
   87.9%·격리 단독 재현. 제거(blocking flag)로 3/3 소멸 → Python expfs 구현 문제,
   GDS 자체의 약점 아님.
2. TTFT tail = store 작업의 시간 점유가 요청 경계를 침범하는 것 자체.
   transport·CPU·GIL 무관(순수 sleep으로 3/3 재현). 블록 delay-free 장기화 →
   다음 요청 admission 지연·빈 step 공회전. 제거 = store를 요청 밖으로 옮기는 것
   (지연 store가 1단계에서 3/3 FAST, CV≈0) → store scheduling 구조 문제.
3. 기각: cuFile·nvidia-fs 고유(1단계 POSIX도 SLOW + cuda_only tail 정상 + write
   속도 불변), CUDA driver/context 경합(cuda_only tail 정상), Python 제어부·GIL
   단독(ctrl FAST).

## 과거 서사 정정

이전에 "GIL 콘보이"로 부른 병명은 함수 수준 분석 결과 두 요인으로 분해·정정된다.
과거 간접 증거(nsys osrt 4.5배·switchinterval 2.3배 가속)는 load-side 단독 측정
레짐의 것으로, store 동시실행 레짐의 병인과 별개다. 엔진의 GIL 보유 2.5배·빈 step
공회전은 실재하는 관찰이지만 격리에서 tail을 단독으로 만들지 못했다(ctrl FAST).

## 인과 폐쇄 축소 재검증 (LEval 64문서 decode 포함, 4 조건 × closed c4 + Poisson 0.3 × 3회)

| 조건 | c4 p95 중앙 | c4 CPU 중앙 | p0.3 p95 (3런) | p0.3 CPU 중앙 | fence(c4) |
|---|---|---|---|---|---|
| 기존 tiering | 11.551 | 117s | 5.0/5.1/9.8 | 184s | 없음(개념 무) |
| GDS 동시저장 | 1.715 | 355s | 3.9/21.1/3.9 | 705s | 34회/81s |
| 동시저장+blocking event | 1.590 | 155s | 7.3/20.1/7.8 | 196s | 34회/75s |
| 지연저장+blocking event | 1.580 | 156s | 2.7/4.2/20.3 | 199s | 34회/78s |

판정.
1. 이중 해리의 end-to-end 확정: blocking event 한 줄로 CPU가 c4에서 2.3배(355→155),
   Poisson에서 3.6배(705→196) 감소(각 3/3)하고 tail은 변하지 않는다(c4 1.72→1.59).
2. 블록 반환 대기 실측: D 계열 c4에서 fence 34회, 총 75~81초/run(회당 평균 약 2.2초)
   — 원인 규명의 "store 점유가 요청 경계 침범" 기전이 직접 수치로 확인됨.
3. 기존 tiering의 c4 tail(11.6s)은 빈 엔진 step 89%와 동행 — tiering도 동시성에서
   자원 대기 공회전을 겪으며, 이 워크로드의 R2 재사용 라운드에서는 D 계열이 tail 우위.
4. λ=0.3에서 서비스율은 전 arm 동일 0.348 req/s(도착 제한) — 이 부하에서 서비스율
   병목은 관찰되지 않음. open-loop tail은 전 arm에 간헐 ~20초 스파이크가 있어
   3회로는 arm 간 우열 판정 불가.
5. 관찰(페어드 재확인 필요): 구 W3의 지연 store Poisson 붕괴(p95 중앙 26.8, 3/3
   악화) 대비 지연저장+blocking event는 2.7/4.2/20.3으로 크게 개선 — blocking
   event가 gap drain의 스핀을 제거한 효과일 가능성. 동일 하네스에서 지연저장
   단독(bev 없음) 재측정으로 닫을 것.

## GPU staging ring (v3 → v3.1) 검증

설계 응답: tail=블록 점유 규명에 대해 GPU staging ring으로 수명 분리를 재현.
KV 블록 → (D2D, side stream) → ring slot → cuFile 대형 write. store job 완료 =
복사 완료(블록 반환), SSD write는 슬롯에서 비동기.

기능 게이트 통과: opt-125m 왕복 토큰 일치, 파일이 cufile 산출과 바이트 동일(9/9),
TRACE에서 ring 등록 direct(bounce 0), posix_staged 대조군도 동일 통과.

v3 결함과 v3.1 수정: v3는 IO 스레드가 compute event를 동기 대기하며 스레드를
점유해 store queue 대기(1.13초)가 블록 점유에 가산됐다. v3.1은 엔진 스레드가
복사를 side stream에 비동기 인큐만 하고, 완료를 get_finished의 event.query()
비차단 폴링으로 보고(추가 폴링 루프 없음 — 엔진이 매 step 호출).

Bailian 150 (opt-2.7b, V2, b64) 결과: CPU는 완전 해결(동시저장 1,074초 → staged
97~104초, 지연저장 90초와 동급). store 바이트·토큰 동일, write 오류 0.

그러나 tail 미해결(p95 5.1~6.3, 지연저장 1.20에 못 미침). 계측이 기전을 규명:
- 수명 분리는 실증됨: completed_before_file = 2737/2737(완료 보고가 파일 출현보다
  선행), COMMIT_GAP 평균 502ms(store committed와 SSD write 완료 분리).
- 그런데 JOB_HOLD(제출→완료 보고 = GPU 블록 점유) 평균 4.8초로 tail과 일치.
- 원인 = ring slot 고갈. 동시 실행 중 cuFile write가 chunk당 177ms(마이크로벤치
  6ms의 30배 — write가 foreground와 경합)이고 슬롯 6개라, 한 job의 ~22 chunk가
  슬롯을 여러 바퀴 돌며 각 바퀴가 이전 SSD write 완료를 기다린다. 블록이 "복사
  완료"가 아니라 "슬롯 빌 때까지=이전 write까지" 잡혀 수명 분리가 무력화됨.

즉 staging ring은 복사만 놓고 보면 수명을 분리하지만, ring 깊이 × write 처리량이
버스트를 못 흡수하면 블록 점유가 다시 write 지연에 묶인다. 병목이 "블록↔복사"에서
"블록↔슬롯↔write"로 한 칸 이동했을 뿐. 근본은 동시 실행 시 write 처리량 저하.
slot-depth 테스트(12/24 슬롯)로 slot-bound 확인 중.

## 비차단 staging 저장 정책 (Bailian 150, V2 b64, 3회)

포화 시 원본 블록을 붙잡지 않는 정책. block(슬롯 대기)/skip(저장 생략)/cpu_fallback(D2H 후 CPU write).

| 정책 | p95 중앙 | CPU | JOB_HOLD | matched(hit) | 실제 write | drop/fb |
|---|---|---|---|---|---|---|
| 기존 tiering | 5.20 | 139s | - | 20,960 | 43GB | - |
| staging block | 6.63 | 118s | 5,312ms | 19,936 | 53GB | 0/0 |
| staging skip | 1.26 | 22s | 556ms | 8,416 | 7GB | 2682/0 |
| staging cpu_fallback | 1.34 | 48s | 753ms | 12,688 | 16GB | 2078/446 |

판정(3/3): 최우선 두 조건 모두 충족.
1. 블록 점유가 복사 시간으로 제한: JOB_HOLD가 block 5,312ms → skip 556ms/cpufb 753ms
   (7~10배). 원본 블록이 SSD write가 아니라 D2D/D2H 복사 완료로 해제됨을 실증.
2. foreground가 슬롯 대기 안 함: slot_full 재폴링이 block 843만 회 → skip/cpufb 2.5천 회.
   포화 시 대기 대신 즉시 drop/fallback 처리.
결과: tail이 block/tiering 5~6.6초에서 1.26~1.34초로 붕괴, CPU 최저(22~48s).

트레이드오프(공짜 아님): 저장 처리량 벽을 우회한 대가로 storage hit 손실.
skip은 matched 20,960→8,416(hit 60% 손실, 실제 write 7GB), cpu_fallback은 12,688
(hit 34% 손실, write 16GB)로 hit를 더 보존하되 CPU를 더 씀. cpu_fallback이 균형점
(tail은 skip과 동급, hit 손실 절반). 저장 throughput이 진짜 병목이고, 비차단 정책은
"무엇을 저장할지"를 선택해 tail을 산다.

한계: 우선순위 4(긴 prefix·재사용 이력 우선 저장)는 worker가 블록만 보고 prefix
재사용 이력을 못 봐 미구현 — 어느 store를 drop할지가 무작위(도착 순). 재사용 가치
기반 admission은 스케줄러 계층 정보 필요.
