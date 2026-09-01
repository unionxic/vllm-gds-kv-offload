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

load-dominated 재사용(라운드 2)의 비교. D1 대 E1은 cuFile transport 효과, D1 대 C는 레이아웃·block size·control plane까지 포함한 시스템 비교.

| 구성 | n | R2 p50 (s) | R2 p95 (s, CV) | R1 cold p95 (s) | CPU (s/런) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A 재계산 | 3 | 1.090 | 1.161 (0.001) | 1.141 | 5 |
| C 기존 tiering (block 16) | 3 | 0.977 | 4.908 (0.023) | 4.474 | 144 |
| D0 cuFile 동시 store | 3 | 0.661 | 0.850 (0.044) | 4.380 | 1,180 |
| E0 POSIX 동시 store | 3 | 0.884 | 1.246 (0.048) | 4.254 | 1,183 |
| D1 cuFile deferred store | 5 | 0.637 | 0.786 (0.046) | 1.169 | 114 |
| E1 POSIX deferred store | 5 | 0.920 | 1.206 (0.056) | 1.168 | 123 |

#### 해석과 한계

- 표의 V2 GDS 수치는 최초 측정값이며 재현 시도 3회(연속 재측정, 빈 디스크 단독 재검증 포함)에서 유지되지 않았다(p95가 4.6~4.9초로 악화, 현행 tiering 이하). p50은 전 측정에서 일치하고 tail만 갈렸다.
- 그 tail 변동의 원인은 후속 스케줄링 실험으로 규명됐다. store 실행과 foreground 엔진 구간의 시간 중첩(엔진 busy loop과 IO 스레드의 GIL 콘보이)이 핵심 변수이며, read/write 제출 순서(read-priority)와 NVMe read/write 동시성(strict phase)은 각각 5회 반복에서 무효였다.
- CPU 시간은 프로세스 전체 누적 CPU time(process_time, 모든 스레드와 요청 밖 store 구간 포함)을 요청 수로 나눈 값이다. wall time이 아니다.
- Host DRAM 왕복은 유도치다. CPU 경유 경로는 SSD read/write 바이트가 pinned 스테이징을 왕복하므로 2×(read+store bytes)로 계산했고, 직행 경로는 설계상 0이다.
- 반복 조건: W1 재현 검증 n=3, 스케줄링 정책 실험은 정책당 D/E 각 n=5(변동계수 = 모표준편차/평균). 단일 빠른 실행은 우위로 인정하지 않았다.
- W1과 W2의 우열 차이는 워크로드 성격 차이다. Bailian W1은 store가 지속 유입되고 재사용 거리가 길며, LEval W2a는 load 중심 재사용이다. 두 결과는 모순이 아니라 적용 조건의 경계다.
- W2의 C(tiering) 수치는 vLLM 종료 시점 race 버그를 가드로 우회한 상태의 측정이다(발화 4~14회/런, 해당 요청의 마지막 chunk store만 생략). 가드 없이는 V2 러너에서도 64문서 스케일에서 3런 전부 크래시했다.
- W2a는 max_tokens=1의 TTFT isolation이다. decode 부하가 있는 조건(W2b)에서 deferred store의 backlog 안정성은 진행 중.
- 발견한 vLLM 버그 2건: 오프로드 요청 종료 시점 레이스(가드로 우회, 미수정), /dev/shm mmap 누출(upstream issue #51579 계열, 로컬 브랜치 `offload-shm-leak-fix` 커밋 07bb9458eb·77c4033078로 근본 수정 — flock 기반 고아 회수, SIGKILL E2E 검증).
- 진행 중: LEval 실제 텍스트 워크로드(W2)에서 위 결론의 교차 검증과 I/O 스케줄러 비교(foreground-aware store admission, cuFile Batch API). Batch API는 standalone 검증을 통과했으나 엔진 통합은 미해결 상태.

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
