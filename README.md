### vLLM + GDS KV 오프로드 실험

## 핵심 결론

- SSD KV 오프로드는 유효하다. opt-2.7b에서는 prefix 1,024토큰 이상부터 재계산보다 2배 이상 빨랐다.
- GDS 직행 자체의 재현 가능한 성능 우위는 확인하지 못했다. 같은 control plane에서 transport만 바꾼 대조(cuFile 대 POSIX bounce)에서 cuFile 단독 효과는 중립이었다.
- 실제 병목은 SSD→GPU 전송 방식보다 store 작업의 스케줄링이었다. store를 요청 사이 gap으로 미루자 tail latency 5배, CPU 사용량 12배 개선이 변동계수 0으로 재현됐고, 이 효과는 cuFile과 POSIX에서 동일했다.
- 간섭을 제거한 뒤에는 cuFile의 transport 우위가 나타났다. LEval 실제 텍스트 기반, load-dominated prefix 재사용 워크로드에서 deferred store를 적용한 cuFile 경로가 동일 control plane의 POSIX 경로를 5/5회 안정적으로 이겼다(R2 p95 평균 35% 격차). 다만 store가 지속 유입되는 Bailian W1에서는 기존 Tiering이 우세했으므로 GDS의 효과는 workload-dependent하다.
- tail latency의 함수 수준 원인을 규명했다. store 동시 실행의 CPU 폭증은 store 스레드의 CUDA event spin wait(구현 문제, blocking event로 9배 감소)였고, tail은 store 작업이 GPU 원본 KV 블록을 SSD 쓰기 완료까지 붙잡아 요청 경계를 침범하는 것이었다. GDS는 홉이 하나라 원본 블록을 느린 쓰기 끝까지 보존해야 하는데, 이 수명 결합이 tail의 원인이고 transport와 무관했다(순수 sleep으로 재현).
- CPU staging의 수명 분리를 GPU staging ring으로 재현하고, 포화 시 원본을 붙잡지 않는 비차단 admission(skip·CPU fallback)을 붙이자 tail이 5초에서 1.3초로, CPU가 1/5로 떨어졌다. 이때 cuFile과 POSIX staging은 전 지표가 동일했다. 즉 이득의 원천은 transport가 아니라 store 실행 구조(수명 분리와 비차단 admission)다.
- 무엇을 저장할지(admission)의 최적해는 workload의 재사용 구조에 의존한다. 반복형 재사용(Bailian coding)에서는 빈도 필터(seen-twice)가 arrival-order random 대비 같은 tail에서 useful hit을 늘리며 write를 절반으로 줄였다. 그러나 먼 거리 1회 재사용(cold tail, Mooncake가 SSD의 표적으로 지목한 구간)에서는 seen-twice가 구조적으로 실패했고(2번째 등장에서 저장하므로 2회 재사용을 못 잡음, useful hit 0), 1번째 등장에서 저장하는 정책만 cold tail을 잡았다. 단일 최적 정책은 없다.

한 줄 목적: 재사용 prefix가 GPU·CPU 메모리를 넘는 워크로드에서 SSD KV 오프로드의 유효 구간과, vLLM 기존 경로(SSD→CPU→GPU) 대비 GPUDirect Storage 직행(SSD→GPU)의 손익을 단일 노드에서 실측.

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

#### 해석과 한계

- 첫 표의 V2 GDS 수치는 최초 측정값. 재현 3회에서 미유지. 원인은 함수 수준까지 규명: CPU 폭증은 store 스레드의 CUDA event 스핀 대기(blocking 플래그로 9배 제거, 3/3), tail은 store 점유의 요청 경계 침범(순수 sleep으로 재현, 지연 store가 제거책).
- CPU 시간 = 전 스레드 누적 CPU time ÷ 요청 수. DRAM 왕복 = 2×(read+store bytes) 유도치. 반복은 조건당 n=3~5, 단일 빠른 실행 불인정.
- W1(store 지속 유입)과 W2(load 중심 재사용)의 우열 차이는 워크로드 경계이지 모순 아님.
- W3(open-loop): closed 동시성은 cuFile 지연 store 우위, Poisson 지속 부하는 처리량 열위가 큐 대기를 증폭해 3~9배 역전 — gap 전용 배출은 단일 스트림 전용.
- 기존 tiering 수치는 종료 race를 가드로 우회한 측정. race는 최신 main #49671로 해결 확인. /dev/shm 누출은 #52596 이후에도 Tiering 경로에서 재현(후속 보고 대상), 로컬 `offload-shm-leak-fix`는 이 경우까지 처리.
- 미해결: cuFile Batch API 엔진 통합.

#### Directory

| 경로 | 내용 |
| --- | --- |
| `env.sh` | 공통 실행 환경 |
| `phase0/` | 기능 개통: fs 티어 스모크, FS-hit 분리 증명, GPU 레이아웃 판정 |
| `phase05/` | SSD KV 정당성 게이트: 재계산 / CPU hit / SSD hit 벤치 |
| `phase1/` | vLLM 밖 cuFile 마이크로벤치와 경로 분류기 |
| `phase2/` | out-of-tree GDS filesystem 스펙 expfs.py |
| `phase3/` | 비교군 벤치, nsys 프로파일, GIL 진단 |
| `w1/` | Bailian coder trace 리플레이와 causal-closure 하네스 |
| `sched/` | 스케줄링 레짐 실험 (store 시간 분리 발견) |
| `w2_leval/` | LEval 실제 텍스트 워크로드와 I/O 스케줄러 비교 (W2a 완료, W2b 진행 중) |
| `docs/detailed-log.md` | 설계 근거, 전체 측정표, 실패와 정정의 상세 기록 |

대용량 재생성 데이터(trace 원본, kvroot, 벤치 바이너리)는 .gitignore로 제외.

수치의 근거와 전체 기록: [docs/detailed-log.md](docs/detailed-log.md)
