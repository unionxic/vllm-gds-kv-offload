### vLLM + GDS KV 오프로드 실험

## 핵심 결론

- SSD KV 오프로드는 유효하다. opt-2.7b에서는 prefix 1,024토큰 이상부터 재계산보다 2배 이상 빨랐다.
- GDS 직행 자체의 재현 가능한 성능 우위는 확인하지 못했다. 같은 control plane에서 transport만 바꾼 대조(cuFile 대 POSIX bounce)에서 cuFile 단독 효과는 중립이었다.
- 실제 병목은 SSD→GPU 전송 방식보다 store 작업의 스케줄링이었다. store를 요청 사이 gap으로 미루자 tail latency 5배, CPU 사용량 12배 개선이 변동계수 0으로 재현됐고, 이 효과는 cuFile과 POSIX에서 동일했다.
- 간섭을 제거한 뒤에는 cuFile의 transport 우위가 나타났다. LEval 실제 텍스트 기반, load-dominated prefix 재사용 워크로드에서 deferred store를 적용한 cuFile 경로가 동일 control plane의 POSIX 경로를 5/5회 안정적으로 이겼다(R2 p95 평균 35% 격차). 다만 store가 지속 유입되는 Bailian W1에서는 기존 Tiering이 우세했으므로 GDS의 효과는 workload-dependent하다.

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

#### 해석과 한계

- 첫 표의 V2 GDS 수치는 최초 측정값. 재현 3회에서 tail 2배 악화로 미유지. 원인은 store 실행과 foreground의 시간 중첩(GIL 콘보이)으로 규명, read/write 순서와 NVMe 동시성 가설은 기각.
- CPU 시간 = 프로세스 전체 누적 CPU time(전 스레드, 요청 밖 store 포함) ÷ 요청 수. Host DRAM 왕복 = 2×(read+store bytes) 유도치, 직행 경로는 0.
- 반복 조건: W1 재현 n=3, 스케줄링·W2 정책당 n=3~5, CV = 모표준편차/평균. 단일 빠른 실행은 불인정.
- W1(Bailian)과 W2(LEval)의 우열 차이는 워크로드 차이. store 지속 유입 대 load 중심 재사용. 모순이 아니라 적용 조건의 경계.
- W2의 기존 tiering 수치는 vLLM 종료 시점 race를 가드로 우회한 측정(발화 4~14회/런). 가드 없이는 64문서에서 전 런 크래시.
- 미완: 지연 store의 backlog 안정성은 closed-loop에서만 확인(요청 경계 gap 존재 전제). open-loop 동시 부하에서는 gap이 사라져 굶을 수 있어 동시성 검증이 다음 순서. cuFile Batch API는 standalone 통과·엔진 통합 미해결, 종료 race 버그는 우회만 하고 미수정.
- 발견 버그 2건: 종료 race(위), /dev/shm mmap 누출(issue #51579 계열, 로컬 브랜치 `offload-shm-leak-fix`에서 flock 회수로 근본 수정).

#### 재현 환경·방법

- 하드웨어: rain 단일 노드. Quadro RTX 5000 16GB(Turing, BAR1 256MB), 로컬 NVMe(ext4), i9-10980XE, RAM 128GB
- 소프트웨어: Ubuntu 20.04, kernel 5.15.0-97, NVIDIA driver 570.211.01(open), CUDA 12.8, nvidia-fs 2.25.7(GDS 1.14.0.33, libcufile 1.13.1), vLLM commit 568afb3a13(tag v0.26.0) editable, torch 2.11 cu128, 모델 facebook/opt-2.7b fp16
- 공통 준비: `source env.sh` (conda libstdc++ 선로드, PYTHONHASHSEED 고정)
- 최소 재현(SSD 정당성 게이트):

```bash
source env.sh && python phase05/bench.py --model facebook/opt-2.7b --prefixes 1024,2032 --repeats 3
```

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
