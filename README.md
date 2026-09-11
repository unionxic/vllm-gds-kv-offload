### vLLM + GDS KV 오프로드 실험

한 줄 목적: 재사용 prefix가 GPU·CPU 메모리를 넘는 워크로드에서 SSD KV 오프로드의 유효 구간과, vLLM 기존 경로(SSD→CPU→GPU) 대비 GPUDirect Storage 직행(SSD→GPU)의 손익을 단일 노드에서 실측.

## 핵심 결론

- SSD KV 오프로드는 유효하다. opt-2.7b에서 prefix 1,024토큰 이상부터 재계산보다 2배 이상 빨랐다.
- GDS 직행 transport 자체는 결정 요인이 아니다. 같은 control plane에서 cuFile과 POSIX bounce를 교체한 대조에서 두 경로의 foreground 성능은 동일했다.
- tail latency의 원인은 전송 방식이 아니라 store 실행 구조다. store 스레드의 CUDA event spin wait가 CPU를 태웠고(blocking event로 9배 감소), store 작업이 GPU 원본 블록을 SSD 쓰기 완료까지 붙잡아 요청 경계를 침범한 것이 tail을 만들었다(순수 sleep으로 재현). CPU staging의 수명 분리를 GPU staging ring으로 재현하고 포화 시 원본을 붙잡지 않는 비차단 admission을 붙이자 tail이 5초에서 1.3초로, CPU가 1/5로 떨어졌다.
- 무엇을 저장할지(admission)의 최적해는 workload의 재사용 구조에 의존한다. 반복형 재사용에서는 빈도 필터(seen-twice)가 random 대비 같은 tail에서 write를 절반으로 줄이며 이겼으나, 먼 거리 1회 재사용(cold tail)에서는 seen-twice가 구조적으로 실패했고(2번째에 저장하므로 2회 재사용을 못 잡음) 1번째에 저장하는 정책만 cold tail을 잡았다. 단일 최적 정책은 없다.

- 가중치가 GPU와 RAM을 넘어 SSD에서 스트리밍되는 조건(OPT-66B, 16 GB GPU)에서는 KV 오프로드의 손익이 대역폭이 아니라 배치 형성으로 정해진다. forward 하나가 가중치 전송 13초로 고정되므로, KV가 먼저 도착한 요청 하나가 혼자 forward를 돌면 함께 올라간 배치가 쪼개져 적중이 재계산보다 34% 느려졌다. 같은 배치의 로드가 끝날 때까지 계산을 미루는 게이트를 스케줄러에 넣자 forward 수가 기준선으로 돌아오고 적중이 재계산보다 2.5% 빨라졌다. 저장 정책과 int8 양자화는 이 손해를 고치지 못했고, 가중치 경로 자체는 cuFile이 POSIX보다 2.4배 빠르고, forward 고정비는 host 티어/12.3 GB/s + SSD 티어/3.44 GB/s로 실측과 5% 안에서 맞아 host 0.85에서는 PCIe가 7할, SSD가 3할이다.

즉 vLLM에서 GDS의 실용성은 전송 API가 아니라 KV 블록 수명 분리, 저장 admission, 캐시 적중률을 함께 설계하느냐로 결정되고, 가중치까지 스트리밍하는 환경에서는 스케줄러가 배치를 지키느냐가 그 위에 더해진다.

#### 대표 측정표

production trace(Bailian coder, 600요청) end-to-end. 각 구현별 최선 구성 비교이며 동일 파라미터 통제 비교가 아님. 러너(V1/V2) 간 수치 비교는 하지 않음.

| 구성 | TTFT p50 / p95 / p99 (s) | 처리량 (tok/s) | CPU 시간 (s/요청) | Host DRAM 왕복 |
| --- | ---: | ---: | ---: | ---: |
| V2 기존 tiering (block 16) | 1.160 / 3.107 / 4.043 | 1,392 | 0.67 | 498 GiB |
| V2 GDS expfs (block 64) | 1.142 / 2.090 / 2.205 | 1,596 | 8.23 | 0 |
| V1 기존 tiering (block 16) | 1.165 / 3.641 / 5.080 | 1,140 | 0.94 | 496 GiB |
| V1 GDS expfs (block 256) | 1.880 / 4.133 / 5.828 | 830 | 4.44 | 0 |

#### 측정 2: LEval 실제 텍스트 (W2a, 64문서, 반복 측정)

load 중심 재사용(라운드 2)의 비교. cuFile 지연 store 대 POSIX 지연 store가 transport 효과, cuFile 지연 store 대 기존 tiering이 시스템 전체 비교.

| 구성 | n | 재사용 p50 (s) | 재사용 p95 (s, CV) | cold p95 (s) | CPU (s/런) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 재계산 (오프로드 없음) | 3 | 1.090 | 1.161 (0.001) | 1.141 | 5 |
| 기존 tiering (block 16) | 3 | 0.977 | 4.908 (0.023) | 4.474 | 144 |
| cuFile 동시 store | 3 | 0.661 | 0.850 (0.044) | 4.380 | 1,180 |
| POSIX 동시 store | 3 | 0.884 | 1.246 (0.048) | 4.254 | 1,183 |
| cuFile 지연 store | 5 | 0.637 | 0.786 (0.046) | 1.169 | 114 |
| POSIX 지연 store | 5 | 0.920 | 1.206 (0.056) | 1.168 | 123 |

#### 측정 3: GPU staging ring과 비차단 admission (Bailian 150, V2 b64, 3회)

원본 GPU 블록 수명을 SSD 쓰기에서 분리하고, ring 포화 시 원본을 붙잡지 않는 정책. JOB_HOLD는 store 제출부터 완료 보고까지의 GPU 블록 점유 시간.

| 구성 | p95 (s) | CPU (s) | JOB_HOLD (ms) | useful hit | write (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 tiering | 5.28 | 138 | — | 20,960 | 43.5 |
| staging block (슬롯 대기) | 6.63 | 118 | 5,312 | 19,936 | 53 |
| staging skip (비차단) | 1.17 | 22 | 556 | 8,480 | 7.3 |
| staging cpu_fallback | 1.34 | 48 | 753 | 12,688 | 16 |

#### 측정 4: Prefix Value Admission (real-text 혼합 workload, 카테고리별 useful hit)

one_shot·near_reuse·far_reuse·repeated를 섞어 admission 변별을 시험. far_reuse가 SSD의 표적인 cold tail(먼 거리 1회 재사용).

| 구성 | p95 (s) | far_hit | near_hit | rep_hit | 총 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 tiering | 5.17 | 28,576 | 26,656 | 55,072 | 110,304 |
| random skip (1번째 저장) | 1.70 | 2,880 | 1,840 | 7,040 | 11,760 |
| seen-twice (2번째 저장) | 1.51 | 0 | 0 | 10,544 | 10,544 |

반복형 Bailian에서는 seen-twice가 random을 이기지만(hit +29%, write 절반), cold-tail 혼합에서는 seen-twice가 far_reuse를 구조적으로 못 잡아(2회 재사용을 2번째에 저장) random에 진다. 양 러너 동일.

#### 측정 5: 가중치 스트리밍과 KV 오프로드의 결합 (OPT-66B, 16 GB GPU + RAM 125 GiB + NVMe)

가중치 132 GB를 GPU 상주 층, pinned host, SSD로 나눠 매 forward 스트리밍하는 조건. 가중치 경로는 host 0.3에서 SSD 층 42개를 forward마다 읽는 구성, KV 결합은 host 0.85에 GPU KV 5.5 GiB로 요청 5개가 한 배치인 구성.

| 구성 | 측정 | 결과 |
| --- | --- | --- |
| forward 고정비 (host 0.85, 상주 2층) | 모형 / 실측 | PCIe 9.3 s + SSD 3.6 s = 12.8 s / 13.2~13.3 s. host 0.3~0.85에서 5% 안 |
| 가중치 경로, decode step | cuFile / POSIX | 28.5 s / 68.0 s. 최선 정책(2-layer prefetch, 스레드 8, ring 8 MiB)에서 26.7 s |
| cuFile 조각 크기 4~8 MiB | 벽시계 / CPU | 벽시계 동일, CPU 28% 감소. ring 없이 27.0 s로 ring 대체 |
| KV 적중 로드, 배치 5, 게이트 없음 | 2라운드 wall (재계산 439.5 s) | 503.7 s, forward 32에서 38로 |
| 같은 조건, 스케줄러 게이트 | | 428.4 s, forward 32, 마지막 요청 첫 토큰 425 s에서 349 s로 |
| 재사용 4 + 1회성 12, 3라운드 합계 | 재계산 / 게이트 없음 / 게이트 2 | 1,321.9 s / 1,395.4 s / 1,317.7 s |
| seen_twice admission (게이트 없음) | 3라운드 합계 손해 | 64.8 s에서 24.9 s. 읽기를 피한 몫이지 원인 교정 아님 |
| KV int8 저장 (게이트 2) | 디스크 / wall | 절반 / 라운드당 1.4 s 느려짐 |

KV 읽기가 forward와 겹쳐도 forward 길이는 기준선과 같고(게이트를 켜면 겹침 자체가 0) 쓰기 전용 라운드는 기준선과 같아, 손해는 전부 forward 개수에서 온다. 상세와 배제한 가설, 문헌 대조는 docs/detailed-log.md의 결합 절.

#### 측정 6: 모델 크기와 host 비율 기준표 (실제 문서 8개, 프리픽스 1,920, 배치 2, KV는 SSD)

host 비율은 오프로드 가중치 대비. forward는 decode step 중앙값, 모형은 CPU 티어/12.3 GB/s + SSD 티어/3.44 GB/s. 합계는 저장 라운드 + 적중 라운드를 재계산 두 라운드와 비교한 순이익.

| 모델, host | forward 실측 / 모형 | prefill 재계산 / 적중 | 적중 라운드 | 저장 라운드 추가 | 두 라운드 합계 |
| --- | --- | --- | --- | --- | --- |
| OPT-66B 0.7 | 19.37 / 19.14 s | 35.0 / 19.7 s | 683 → 633 s (-7.3%) | +6 s | 1365 → 1323 s (-3.1%) |
| OPT-66B 0.5 | 24.70 / 24.70 s | 40.3 / 25.1 s | 854 → 807 s (-5.4%) | +25 s | 1707 → 1686 s (-1.2%) |
| OPT-66B 0.3 | 29.76 / 29.82 s | 45.3 / 30.3 s | 1015 → 971 s (-4.4%) | +45 s | 2030 → 2032 s (+0.1%) |
| OPT-66B 0.1 | 35.68 / 35.37 s | 51.2 / 36.0 s | 1205 → 1153 s (-4.3%) | +66 s | 2409 → 2424 s (+0.6%) |
| OPT-30b 1.0 | 5.08 / 4.81 s | 13.1 / 5.2 s | 195 → 175 s (-10.4%) | +5 s | 395 → 374 s (-5.2%) |
| OPT-30b 0.7 | 8.92 / 8.69 s | 16.7 / 9.0 s | 317 → 296 s (-6.9%) | +3 s | 634 → 616 s (-2.9%) |
| OPT-30b 0.1 | 16.65 / 16.18 s | 24.4 / 16.8 s | 565 → 543 s (-3.8%) | +52 s | 1154 → 1160 s (+0.5%) |
| OPT-13b 1.0 | 2.18 / 2.05 s | 6.4 / 2.2 s | 87 → 88 s (+1.7%) | +2 s | 175 → 176 s (+0.5%) |
| OPT-13b 0.7 | 3.73 / 3.63 s | 7.8 / 3.8 s | 136 → 125 s (-7.8%) | +1 s | 272 → 263 s (-3.4%) |
| OPT-13b 0.1 | 6.87 / 6.79 s | 10.9 / 7.0 s | 236 → 230 s (-2.8%) | +72 s | 473 → 538 s (+13.8%) |
| OPT-6.7b 1.0 | 1.12 / 1.05 s | 3.5 / 1.1 s | 46 → 39 s (-14.0%) | +1 s | 91 → 86 s (-6.5%) |
| OPT-6.7b 0.7 | 2.23 / 1.89 s | 4.5 / 2.2 s | 81 → 93 s (+14.6%) | +0 s | 161 → 174 s (+8.0%) |
| OPT-6.7b 0.1 | 4.21 / 3.49 s | 6.4 / 4.2 s | 143 → 143 s (-0.3%) | +44 s | 287 → 330 s (+15.1%) |

적중이 아끼는 것은 prefill의 토큰 계산 몫이고 모델 크기에 비례(6.7b 2초, 13b 4초, 30b 8초, 66B 15초)하며 host 비율과 무관하다. 저장 라운드의 추가 시간은 prefill 직후 KV 쓰기와 가중치 SSD 읽기가 겹치는 decode step 하나에서 나며 SSD 가중치 몫에 비례하므로, 저장까지 넣은 순이익은 host 0.5 이상에서만 남고 0.3 이하는 재사용이 두 번 이상이어야 한다. 6.7b와 13b에서 마지막 배치의 KV 로드만 5~21초 느려지는 것은 엔진 폴링 루프의 GIL 독점이고, 출력 없는 step 뒤 1 ms 양보로 6.7b host 0.7의 적중 라운드가 +14.6%에서 −10.4%로 뒤집혔다. 13b는 정적 버퍼 등록이 BAR1 창을 채워 cuFile 읽기가 실패해 등록 총량 상한을 포크에 추가했다. host memory 비율(RAM 대비)로 다시 본 표와 전체 표는 docs/detailed-log.md의 기준표 절. 66B의 RAM 0.5~0.8은 측정 중.

#### 해석과 한계

- 첫 표의 V2 GDS 수치는 최초 측정값. 재현 3회에서 미유지. 원인은 함수 수준까지 규명: CPU 폭증은 store 스레드의 CUDA event 스핀 대기(blocking 플래그로 9배 제거, 3/3), tail은 store 점유의 요청 경계 침범(순수 sleep으로 재현, 지연 store가 제거책).
- CPU 시간 = 전 스레드 누적 CPU time ÷ 요청 수. DRAM 왕복 = 2×(read+store bytes) 유도치. 반복은 조건당 n=3~5, 단일 빠른 실행 불인정.
- W1(store 지속 유입)과 W2(load 중심 재사용)의 우열 차이는 워크로드 경계이지 모순 아님.
- W3(open-loop): closed 동시성은 cuFile 지연 store 우위, Poisson 지속 부하는 처리량 열위가 큐 대기를 증폭해 3~9배 역전 — gap 전용 배출은 단일 스트림 전용.
- 기존 tiering 수치는 종료 race를 가드로 우회한 측정. race는 최신 main #49671로 해결 확인. /dev/shm 누출은 #52596 이후에도 Tiering 경로에서 재현(후속 보고 대상), 로컬 `offload-shm-leak-fix`는 이 경우까지 처리.
- 미해결: cuFile Batch API 엔진 통합. cuFile 1 MiB 조각 경로가 간헐적으로 3배 느려지는 모드의 원인. 스케줄러 게이트의 이득이 2.5%로 작고 GPU 한 장이라 일반성 미확립. 06의 host 0.1 편차 68%는 4 MiB 조각 재측정에서 재현되지 않아 1 MiB 느린 모드로 봄.

#### Directory

네 개 대분류로 정리했다. lib는 재사용 구현, harness는 여러 실험이 공유하는 실행기, experiments는 주제별 실험, results는 원자료.

| 경로 | 내용 |
| --- | --- |
| `env.sh` | 공통 실행 환경. PYTHONPATH에 lib·harness 등록 |
| `lib/` | 재사용 구현: expfs.py(GDS filesystem 스펙), gdslib.py(cuFile ctypes), scheduler.py·policies.py·prof_*.py·value_admission.py·snapshot.py·cufile_batch.py |
| `harness/` | 여러 실험이 공유하는 실행 하네스 run_bench.py |
| `experiments/01-feasibility/` | 기능 개통(bringup)·SSD 정당성 게이트(prefix-gate)·cuFile 마이크로벤치·expfs 스모크·A~E 비교군 매트릭스 |
| `experiments/02-bailian/` | Bailian coder trace 리플레이(replay600)와 축소 윈도 실험(window150): 원인 규명·staging·비차단·V1 b256 |
| `experiments/03-leval/` | LEval 실제 텍스트 워크로드, I/O 스케줄러 비교, open-loop, 혼합 admission 워크로드 |
| `experiments/04-admission/` | Prefix Value Admission: 오프라인 시뮬레이션(sim)과 단계별 게이트 기록 |
| `experiments/05-upstream/` | 종료 race·/dev/shm 누출 upstream 회귀 검증 |
| `experiments/06-weight-offload/` | OPT-66B 가중치 3단 스트리밍(GPU·host·SSD)의 cuFile 대 POSIX, prefetch 깊이와 ring, host 비율 스윕 |
| `experiments/07-combined/` | 가중치 스트리밍 위에 KV를 SSD로 두는 결합 실험, pinned 정확 등록, 메모리 워치독 |
| `experiments/08-cufile-bounce/` | cuFile 미등록 버퍼 경로의 조각 크기, 검증 읽기, 읽기 패턴 마이크로벤치 |
| `experiments/09-kv-policy/` | 배치 구성, 구간 계측 러너, 비용 분해, 스케줄러 게이트 A/B, 저장 정책, KV int8 |
| `experiments/10-model-host-baseline/` | 모델 크기 × host 비율 기준표 캠페인(실제 문서, 배치 2, KV는 SSD) |
| `results/` | 실험별 원자료(bailian·leval·leval-openloop·admission·weight-offload·combined·cufile-bounce·kv-policy) |
| `docs/detailed-log.md` | 설계 근거, 전체 측정표, 실패와 정정의 상세 기록 |

대용량 재생성 데이터(trace 원본, kvroot, 프로파일러 산출물)는 .gitignore로 제외.

수치의 근거와 전체 기록: [docs/detailed-log.md](docs/detailed-log.md)
