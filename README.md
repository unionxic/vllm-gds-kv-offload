### vLLM + GDS KV 오프로드 실험

한 줄 목적: 재사용 prefix가 GPU·CPU 메모리를 넘는 워크로드에서 SSD KV 오프로드의 유효 구간과, vLLM 기존 경로(SSD→CPU→GPU) 대비 GPUDirect Storage 직행(SSD→GPU)의 손익을 단일 노드에서 실측.

## 핵심 결론

- SSD KV 오프로드는 유효하다. opt-2.7b에서 prefix 1,024토큰 이상부터 재계산보다 2배 이상 빨랐다.
- GDS 직행 transport 자체는 결정 요인이 아니다. 같은 control plane에서 cuFile과 POSIX bounce를 교체한 대조에서 두 경로의 foreground 성능은 동일했다.
- tail latency의 원인은 전송 방식이 아니라 store 실행 구조다. store 스레드의 CUDA event spin wait가 CPU를 태웠고(blocking event로 9배 감소), store 작업이 GPU 원본 블록을 SSD 쓰기 완료까지 붙잡아 요청 경계를 침범한 것이 tail을 만들었다(순수 sleep으로 재현). CPU staging의 수명 분리를 GPU staging ring으로 재현하고 포화 시 원본을 붙잡지 않는 비차단 admission을 붙이자 tail이 5초에서 1.3초로, CPU가 1/5로 떨어졌다.
- 무엇을 저장할지(admission)의 최적해는 workload의 재사용 구조에 의존한다. 반복형 재사용에서는 빈도 필터(seen-twice)가 random 대비 같은 tail에서 write를 절반으로 줄이며 이겼으나, 먼 거리 1회 재사용(cold tail)에서는 seen-twice가 구조적으로 실패했고(2번째에 저장하므로 2회 재사용을 못 잡음) 1번째에 저장하는 정책만 cold tail을 잡았다. 단일 최적 정책은 없다.

즉 vLLM에서 GDS의 실용성은 전송 API가 아니라 KV 블록 수명 분리, 저장 admission, 캐시 적중률을 함께 설계하느냐로 결정된다.

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
| `results/` | 실험별 원자료(bailian·leval·leval-openloop·admission) |
| `docs/detailed-log.md` | 설계 근거, 전체 측정표, 실패와 정정의 상세 기록 |

대용량 재생성 데이터(trace 원본, kvroot, 프로파일러 산출물)는 .gitignore로 제외.

수치의 근거와 전체 기록: [docs/detailed-log.md](docs/detailed-log.md)
