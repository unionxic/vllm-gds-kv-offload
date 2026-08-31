### vLLM + GDS KV 오프로드 실험

에이전트, RAG, 장기 세션처럼 같은 prefix를 반복하는 워크로드에서는 재사용 가능한 KV 캐시가 GPU와 CPU 메모리를 넘어설 수 있다. 이 실험은 두 가지를 차례로 검증한다. SSD에 저장해 둔 KV를 읽는 것이 prefix를 다시 계산하는 것보다 유리한 구간이 실제로 존재하는지 확인하고, 그 구간에서 vLLM의 기존 경로(SSD에서 CPU를 거쳐 GPU로)를 GPUDirect Storage 직행(SSD에서 GPU로)으로 바꾸면 성능과 자원 비용이 어떻게 달라지는지 잰다.

측정 장비는 rain 서버다. Quadro RTX 5000 16GB, 로컬 NVMe(ext4), nvidia-fs로 native GDS가 개통된 단일 노드. 작업 소스는 `~/vllm`(commit 568afb3a13, tag v0.26.0 기준)이고 conda 환경 `gdsllm`에 editable로 설치되어 있다. 모든 실행 전에 `source env.sh`가 필요하다.

#### 저장소 구성

| 경로 | 내용 |
|---|---|
| `env.sh` | 공통 실행 환경 |
| `phase0/` | 기능 개통: fs 티어 스모크, FS-hit 분리 증명, GPU 레이아웃 판정 |
| `phase05/` | SSD KV 정당성 게이트: 재계산 / CPU hit / SSD hit 벤치 |
| `phase1/` | vLLM 밖 cuFile 마이크로벤치와 경로 분류기 |
| `phase2/` | out-of-tree GDS filesystem 스펙 expfs.py |
| `phase3/` | 비교군 벤치, nsys 프로파일, GIL 진단 |
| `w1/` | Bailian coder trace 리플레이와 causal-closure 하네스 |
| `docs/detailed-log.md` | 설계 근거, 전체 측정표, 발견한 버그와 함정의 상세 기록 |

대용량 재생성 데이터(trace 원본, kvroot, 벤치 바이너리)는 .gitignore로 제외된다. vLLM 쪽 수정(/dev/shm mmap 누출의 근본 해결)은 `~/vllm`의 `offload-shm-leak-fix` 브랜치에 별도 커밋으로 있다.

#### 질문과 답

| 질문 | 답 |
|---|---|
| SSD hit가 prefill 재계산보다 빠른가 | 그렇다. opt-2.7b에서 prefix 1024토큰 이상이면 재계산 대비 2배 이상 빠르고, store 비용은 재사용 한두 번이면 회수된다 |
| 실제 워크로드에서 filesystem hit가 자연 발생하는가 | 그렇다. Bailian production trace 600요청의 unique KV가 약 200GB로 GPU와 CPU를 수십 배 초과해, 인위적 압박 없이 SSD 티어가 강제된다 |
| GDS 직행이 기존 CPU 경유보다 나은가 | 조건부다. 아래 요약 참조 |

#### 핵심 결과

production trace(Bailian coder, 600요청) end-to-end에서 각 설계의 최선 구성을 붙였다. 러너 간 수치는 비교하지 않는다.

| | TTFT p50 / p95 / p99 | tok/s | CPU 초/요청 | host DRAM 왕복 |
|---|---|---|---|---|
| V2 기존 tiering (block 16) | 1.160 / 3.107 / 4.043 | 1392 | 0.67 | 498GiB |
| V2 GDS expfs (block 64) | 1.142 / 2.090 / 2.205 | 1596 | 8.23 | 0 |
| V1 기존 tiering (block 16) | 1.165 / 3.641 / 5.080 | 1140 | 0.94 | 496GiB |
| V1 GDS expfs (block 256) | 1.880 / 4.133 / 5.828 | 830 | 4.44 | 0 |

기본 러너(V2)에서는 GDS 직행이 이긴다. 중앙값은 동률이지만 tail latency가 1.5~1.8배 좋고 처리량이 15% 높으며 host DRAM 왕복이 사라진다. 대가는 store 쪽 CPU다. cross-layer 러너(V1)에서는 synthetic 벤치의 승자였던 GDS가 실전에서 뒤집혔는데, 거친 chunk가 hit를 깎고 연속 store의 CPU와 GIL 간섭이 전체 분포를 오염시켰기 때문이다. synthetic이 못 보던 것을 production trace가 보여 준 사례다.

과정에서 나온 부수 성과: naive per-chunk GDS가 느린 진짜 원인이 transport가 아니라 엔진 busy loop과 IO 스레드 사이의 GIL 콘보이임을 nsys로 규명했고(chunk 확대로 처방), vLLM에서 버그 둘(오프로드 요청 종료 시점 레이스, /dev/shm mmap 누출)을 찾아 후자는 flock 기반 회수로 근본 수정했다.

수치의 근거, 설계 결정의 이유, 실패와 정정의 기록은 전부 `docs/detailed-log.md`에 있다.
