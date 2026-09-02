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

## 진행 중

경계 반복(ctrl/native_wait/cuda_only ×3)과 제거 검증(FIX=blocking_event ×3:
예상 = CPU만 정상화·tail 유지 → 분리 확증).
