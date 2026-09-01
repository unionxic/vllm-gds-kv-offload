### vLLM + GDS KV 오프로드 실험

같은 prefix를 반복하는 워크로드(에이전트, RAG, 장기 세션)에서 재사용 KV 캐시가 GPU와 CPU 메모리를 넘는 상황을 대상으로 한 실측 기록. 검증 대상은 두 가지. SSD에 저장한 KV를 읽는 것이 prefix 재계산보다 유리한 구간의 존재 확인, 그리고 그 구간에서 vLLM 기존 경로(SSD→CPU→GPU)를 GPUDirect Storage 직행(SSD→GPU)으로 바꿀 때의 성능·자원 손익 측정.

측정 환경: rain 서버. Quadro RTX 5000 16GB, 로컬 NVMe(ext4), nvidia-fs native GDS 개통 상태의 단일 노드. 작업 소스는 `~/vllm`(commit 568afb3a13, tag v0.26.0 기준), conda 환경 `gdsllm`에 editable 설치. 모든 실행 전 `source env.sh` 필요.

#### Directory

| 경로 | 내용 |
|---|---|
| `env.sh` | 공통 실행 환경 |
| `phase0/` | 기능 개통: fs 티어 스모크, FS-hit 분리 증명, GPU 레이아웃 판정 |
| `phase05/` | SSD KV 정당성 게이트: 재계산 / CPU hit / SSD hit 벤치 |
| `phase1/` | vLLM 밖 cuFile 마이크로벤치와 경로 분류기 |
| `phase2/` | out-of-tree GDS filesystem 스펙 expfs.py |
| `phase3/` | 비교군 벤치, nsys 프로파일, GIL 진단 |
| `w1/` | Bailian coder trace 리플레이와 causal-closure 하네스 |
| `sched/` | 스케줄링 레짐 실험 (store 시간 분리 발견) |
| `w2_leval/` | LEval 실제 텍스트 워크로드와 I/O 스케줄러 비교 (진행 중) |
| `docs/detailed-log.md` | 설계 근거, 전체 측정표, 발견한 버그와 함정의 상세 기록 |

대용량 재생성 데이터(trace 원본, kvroot, 벤치 바이너리)는 .gitignore로 제외. vLLM 쪽 수정(/dev/shm mmap 누출 근본 해결)은 `~/vllm`의 `offload-shm-leak-fix` 브랜치에 별도 커밋으로 보존.

#### 검증 항목

| 항목 | 결과 |
|---|---|
| SSD hit와 prefill 재계산의 속도 비교 | opt-2.7b 기준 prefix 1024토큰 이상에서 재계산 대비 2배 이상 우위. store 비용은 재사용 한두 번으로 회수 |
| 실제 워크로드에서 filesystem hit의 자연 발생 여부 | Bailian production trace 600요청의 unique KV 약 200GB. GPU와 CPU를 수십 배 초과해 인위적 압박 없이 SSD 티어 강제 확인 |
| GDS 직행과 기존 CPU 경유의 우열 | 재현 가능한 우위 미확립. 아래 측정 참조 |

#### 측정

production trace(Bailian coder, 600요청) end-to-end, 각 설계의 최선 구성 비교. 러너 간 수치 비교는 하지 않음.

| | TTFT p50 / p95 / p99 | tok/s | CPU 초/요청 | host DRAM 왕복 |
|---|---|---|---|---|
| V2 기존 tiering (block 16) | 1.160 / 3.107 / 4.043 | 1392 | 0.67 | 498GiB |
| V2 GDS expfs (block 64) | 1.142 / 2.090 / 2.205 | 1596 | 8.23 | 0 |
| V1 기존 tiering (block 16) | 1.165 / 3.641 / 5.080 | 1140 | 0.94 | 496GiB |
| V1 GDS expfs (block 256) | 1.880 / 4.133 / 5.828 | 830 | 4.44 | 0 |

표의 V2 GDS 수치는 재현 시도 세 번에서 미유지(tail 두 배 악화, 현행 tiering 이하). transport 대조에서 cuFile 단독 효과는 중립. tail 변동의 원인은 이후 규명 — store 실행과 foreground 엔진 구간의 시간 중첩(GIL 콘보이)이 핵심 변수. store를 요청 사이 gap으로 미루는 것만으로 tail 5배·CPU 12배 개선이 변동계수 0으로 재현, 효과는 cuFile과 POSIX 동일. 결론: SSD 티어는 유효, 승부처는 transport가 아니라 store 스케줄링. 부수 성과: vLLM 버그 2건 발견(요청 종료 레이스, /dev/shm mmap 누출), 후자는 flock 기반 회수로 근본 수정.

수치의 근거, 설계 결정의 이유, 실패와 정정의 기록은 `docs/detailed-log.md` 참조.
