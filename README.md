### vLLM + GDS KV 오프로드 실험

같은 prefix를 반복하는 워크로드에서 KV 캐시를 SSD까지 내리는 것이 언제 이득인지, 그리고 vLLM의 기존 경로(SSD→CPU→GPU)를 GPUDirect Storage 직행(SSD→GPU)으로 바꾸는 것이 정당한지 단일 노드(RTX 5000 16GB, 로컬 NVMe, native GDS)에서 실측한 기록이다. vLLM v0.26 위에 out-of-tree GDS filesystem 백엔드(expfs)를 구현해 synthetic, production trace(Bailian coder), 실제 텍스트(LEval)로 검증했다.

#### 결론 요약

SSD 티어 자체는 유효하다. prefix 1024토큰 이상이면 SSD hit가 재계산보다 2배 이상 빠르고, production trace의 working set은 인위적 압박 없이 GPU와 CPU를 수십 배 넘는다.

GDS 직행의 재현 가능한 우위는 확립되지 않았다. 같은 control plane에서 transport만 바꾼 대조(cuFile 대 POSIX bounce)에서 cuFile 단독 효과는 중립이었고, 한 번 관찰됐던 GDS의 tail 우위는 재현되지 않았다.

대신 tail을 지배하는 진짜 변수를 찾았다. store 실행이 foreground 엔진 구간과 시간적으로 겹치는 것 자체가 원인이며(엔진 busy loop과 IO 스레드의 GIL 콘보이), store를 요청 사이 gap으로 미루면 tail 5배, CPU 12배 개선이 변동계수 0으로 재현된다. 이 효과는 cuFile과 POSIX에서 동일하므로 transport가 아니라 스케줄링의 문제다.

부수 성과로 vLLM 버그 둘(오프로드 요청 종료 레이스, /dev/shm mmap 누출)을 찾았고 후자는 flock 기반 회수로 근본 수정했다.

#### 저장소 구성

| 경로 | 내용 |
|---|---|
| `phase0/`~`phase3/` | 개통, 정당성 게이트, cuFile 마이크로벤치, expfs, 비교군과 진단 |
| `w1/` | Bailian coder trace 리플레이 |
| `sched/` | 스케줄링 레짐 실험 (store 시간 분리 발견) |
| `w2_leval/` | LEval 실제 텍스트 워크로드와 I/O 스케줄러 비교 (진행 중) |
| `docs/detailed-log.md` | 수치의 근거, 설계 결정, 실패와 정정의 전체 기록 |

대용량 재생성 데이터(trace 원본, kvroot, 바이너리)는 커밋하지 않는다. vLLM 쪽 수정은 `~/vllm`의 `offload-shm-leak-fix` 브랜치에 있다.
