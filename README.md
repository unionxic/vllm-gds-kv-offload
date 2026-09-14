### vLLM + GDS KV 오프로드 실험

목적: 재사용 프리픽스가 GPU와 CPU 메모리를 넘는 워크로드에서 SSD KV 오프로드가 언제 이득인지, 그리고 가중치까지 GPU와 RAM을 넘쳐 SSD에서 스트리밍되는 조건에서는 그 손익이 어떻게 바뀌는지를 단일 노드에서 실측.

| 항목 | 값 |
| --- | --- |
| GPU | Quadro RTX 5000 16 GB. BAR1 256 MiB(카드 최대, Resizable BAR 레지스터로 확인) |
| RAM, SSD | 125 GiB, Samsung 970 EVO 500 GB(OS와 같은 디스크). cuFile 읽기 3.2 GiB/s, 지속 쓰기 0.3~0.7 GB/s(빈 공간에 좌우), pinned H2D 12.3 GB/s |
| 소프트웨어 | CUDA 12.8, 드라이버 570, cuFile 1.13, nvidia-fs 2.25.7, vLLM 0.26.1 기반 포크(~/vllm, weight-ssd-offload) |
| 모델 | opt-2.7b, 6.7b, 13b, 30b, 66b, Qwen2.5-72B-Instruct. KV 경로는 expfs(01~10) 또는 포크 in-tree CuFileFsSpec(11) |

#### 핵심 결론

가중치가 GPU에 들어가는 조건(opt-2.7b, 01~05)

- SSD KV 적중은 프리픽스 1,024토큰 이상에서 재계산보다 빠름. 2,032토큰에서 재계산 1.155 s, SSD 0.483 s, CPU 0.110 s.
- 전송 API(cuFile 대 POSIX)는 결정 요인이 아님. 같은 제어 구조에서 foreground 성능 동일.
- tail의 원인은 store 실행 구조. CUDA event 스핀 대기(blocking event로 CPU 9배 감소)와 store가 GPU 원본 블록을 쓰기 완료까지 붙잡는 것. GPU staging ring과 비차단 admission으로 p95 5 s에서 1.3 s.
- 저장 admission의 최적해는 워크로드 의존. 반복형은 seen-twice가 이기고 먼 거리 1회 재사용은 1번째 저장만 잡음.

가중치가 SSD에서 스트리밍되는 조건(OPT-66B, 06~11)

- forward 하나는 가중치 이동이고 CPU 티어/12.3 GB/s + SSD 티어/3.44 GB/s로 실측과 1~3% 안. layer가 0.5 GiB 미만이면 SSD 2.9 GB/s.
- SSD KV 적중이 아끼는 것은 prefill의 토큰 계산 몫뿐. 66B 3,940토큰에서 15.5 s, host 비율과 무관. 모델 크기에 비례(6.7b 2 s, 13b 4 s, 30b 8 s).
- 저장 라운드 손해는 KV 쓰기와 가중치 SSD 읽기의 디스크 공유. 쓰기 중 가중치 읽기가 3.2에서 0.3 GB/s로 떨어짐(nvme 1초 샘플). SSD 가중치 몫에 비례해 RAM 0.7에서 +6 s, 0.1에서 +66 s.
- KV 쓰기 속도는 backend가 아니라 SSD의 지속 쓰기 상한이 정함. 970 EVO는 SLC 캐시 약 4 GB 뒤 0.33 GB/s(디스크 94% 사용) 또는 0.70 GB/s(36% 사용). 같은 경로의 읽기는 3.5 GB/s. 66B 배치당 KV 8.4 GiB가 캐시보다 커서 쓰기가 forward 여러 개에 걸침.
- write-behind(SSD 티어 layer 읽는 동안 KV 쓰기 정지, host 티어 구간에 재개)는 기본값 손해를 +19%에서 +9.1%로 줄임. 쓰기를 decode forward에서 빼내 다음 배치의 prefill forward로 옮긴 것이며 쓰는 양은 그대로.
- 한 사이클 순이익(저장 + 적중 대 재계산 2라운드)은 RAM 0.5 이상에서 1~3%. 0.3 이하는 0.
- vLLM 스케줄러는 KV 로드가 끝난 요청부터 승격하므로 먼저 온 요청이 혼자 forward를 돌아 배치가 쪼개짐. 게이트(승격 대기)로 forward 수가 기준선으로 복귀.
- 오프로더 버퍼를 두 세트로 두면(prefetch_step 2) prefill 계산이 전송 아래 숨어 재계산 비용이 0에 가까워지고 KV 적중이 아낄 몫이 사라짐. 16 GB에서는 배치 2와 같이 못 넣음.
- 손대지 않은 기본값(게이트 없음, 1 MiB 조각, KV 자동)은 RAM 0.5에서 두 단계 합계 +19% 손해. 같은 조건에 게이트와 4 MiB 조각만 넣으면 −1.3%.
- Qwen2.5-72B-Instruct(GQA, 토큰당 KV 0.33 MB) RAM 0.5 기본값, LongBench-v2 8건 × 8k 토큰: SSD 적중이 두 단계 합계 3,340 → 2,896 s(−13.3%). 저장 단계 손해 0(문서 KV 2.6 GB가 SSD 쓰기 캐시 안), 적중 단계 prefill forward 67 → 29 s. 출력 토큰열 동일.
- LMCache 0.5.5 GDS L1은 이 카드에서 불성립. staging 버퍼 등록이 BAR1을 넘고 cuFileReadAsync가 적중에서 멈춤. LMCache는 저장 시점·admission·가중치 층 인식이 없어 위 문제의 설계 바깥.

#### 측정

프리픽스 길이별 SSD 정당성(opt-2.7b, 단일 프로세스)

| 프리픽스 | 재계산 | SSD 읽기 | CPU 읽기 |
| --- | ---: | ---: | ---: |
| 2,032 | 1.155 s | 0.483 s | 0.110 s |
| 1,024 | 0.365 s | 0.256 s | 0.070 s |
| 512 | 0.145 s | 0.146 s | 0.049 s |

LEval 실제 텍스트 64문서, 재사용 라운드(W2a)

| 구성 | 재사용 p50 | 재사용 p95 | cold p95 | CPU (s/런) |
| --- | ---: | ---: | ---: | ---: |
| 재계산 | 1.090 s | 1.161 s | 1.141 s | 5 |
| 기존 tiering (block 16) | 0.977 s | 4.908 s | 4.474 s | 144 |
| cuFile 지연 store | 0.637 s | 0.786 s | 1.169 s | 114 |
| POSIX 지연 store | 0.920 s | 1.206 s | 1.168 s | 123 |

OPT-66B, host memory 비율(RAM 대비), LEval 문서 8개, 프리픽스 1,920, 배치 2, 게이트 켬, 4 MiB 조각

| RAM 비율 | CPU / SSD layer | forward 실측 / 모형 | 적중 라운드 | 저장 추가 | 두 라운드 합계 |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 6 / 58 | 35.7 / 35.4 s | −4.3% | +66 s | +0.6% |
| 0.3 | 19 / 45 | 29.8 / 29.8 s | −4.4% | +45 s | +0.1% |
| 0.5 | 33 / 31 | 23.9 / 23.8 s | −6.1% | +28 s | −1.3% |
| 0.6 | 39 / 25 | 21.4 / 21.3 s | −6.9% | +15 s | −2.4% |
| 0.7 | 46 / 18 | 18.5 / 18.3 s | −7.9% | +24 s | −2.1% |
| 0.8 | 52 / 12 | 16.1 / 15.7 s | −8.8% | +14 s | −3.2% |

같은 66B RAM 0.5, 설정 차이

| 조건 | 두 단계 합계(저장 + 적중 대 재계산) |
| --- | --- |
| 기본값(게이트 없음, 1 MiB 조각, KV 자동 10.4 GiB, gpu_util 0.85) | 1,674 → 1,992 s (+19%). forward 64 → 70, 저장 단계 prefill 32 → 41.5 s |
| 기본값 + write-behind(cufile_fs_store_window=host, 상한 30 s) | 1,674 → 1,827 s (+9.1%). decode forward 최대 53.3 → 24.6 s, 저장 단계 prefill 41.5 → 36.5 s |
| 게이트 + 4 MiB 조각 + KV 10.4 GiB | 1,653 → 1,631 s (−1.3%) |

Qwen2.5-72B-Instruct RAM 0.5, 기본값, 8건 × 8k 토큰(저장 + 적중, 재계산 대비)

| 조건 | cold_fill(저장) | reverse_retrieve(적중) | 두 단계 합계 |
| --- | --- | --- | --- |
| 재계산 | 1,800 s | 1,539 s | 3,340 s |
| SSD 적중 | 1,795 s | 1,101 s | 2,896 s (−13.3%) |
| SSD 적중 + write-behind 30 s | 1,773 s | 1,076 s | 2,849 s (−14.7%) |

이중 버퍼(66B RAM 0.7, 배치 1, 재계산)

| prefetch_step | prefill forward | decode forward | 8요청 wall clock |
| --- | ---: | ---: | ---: |
| 1 | 25.5 s | 18.3 s | 1,233 s |
| 2 | 17.8 s | 17.6 s | 1,128 s |

작은 모델(6.7b, 13b, 30b × host 비율 5점)과 09의 배치·정책·양자화 결과, nsys 관측은 docs/detailed-log.md.

#### 구현

| 위치 | 내용 |
| --- | --- |
| 포크 offloader/prefetch.py, ssd_tier.py | 가중치 3단 스트리밍(GPU 정적 버퍼, pinned host, SSD cuFile). 정확 pinned 등록(VLLM_OFFLOAD_PIN_EXACT), 등록 총량 상한(VLLM_OFFLOAD_SSD_REGISTER_MAX_MB), cuFile 캐시 워밍업, SSD 창 신호(IoWindow) |
| 포크 v1/core/sched/scheduler.py | 승격 대기 게이트 VLLM_KV_LOAD_WAVE_GATE(1: 로드 완료 요청, 2: 신규 요청도), VLLM_KV_LOAD_WAVE_WAIT_S |
| 포크 csrc/kv_offload/cufile_fs.cpp, v1/kv_offload/cufile_fs/ | CuFileFsSpec. GPU KV 블록을 cuFile로 파일에 직접 저장·로드하는 C++ backend. 쓰기 일시정지(cufile_fs_store_window=host), admission(cufile_fs_admission=all/never/profile). 출력이 재계산과 동일함을 확인 |
| lib/obs | 관측 계층. host 지표, nvidia-fs·프로세스·캐시 파일 1초 샘플, 이벤트 jsonl(wall과 monotonic), 환경 기록, 용량 검사, nsys 래퍼, 요약 |
| experiments/11-observability/run_obs.py | cold_fill → settle → reverse_retrieve 러너. kv-transport cufile/none/lmcache, 기본값 런(--pure), 출력 토큰열 기록 |
| tools/compare_results.py, baseline_table.py | 전 결과를 고정비 모형과 대조, 기준 런 대비 변화율, 순이익 표 |
| lib/expfs.py | 01~10 결과 재현용 파이썬 backend. 새 실험에는 쓰지 않음 |

#### 한계와 미해결

- 카드 한 장, SSD 한 장(OS와 공유), BAR1 256 MiB. KV 텐서 등록이 안 되어 KV 경로는 cuFile bounce 두 홉. 데이터센터 GPU에서는 같은 코드가 등록 직접 DMA.
- 이중 버퍼 조건에서 KV 오프로드 손익, write-behind와 게이트를 같이 켠 조합, host KV 층과 host 배분(layer 1개 = forward당 0.43 s 환율)은 미측정. OPT-66B 가중치는 삭제해 66B 추가 런은 없음.
- write-behind가 보는 SSD 구간 표시는 prefetch 발행 시점이라 실제 SSD 읽기보다 약 5.5 s 앞섬(72B nsys). 실제 읽기 시작에 맞추는 수정은 미적용.
- 게이트의 일반성은 forward가 비싼 조건에서만 검증. GPU 상주 모델에서는 이득이 ms 단위.
- 다음 모델은 Qwen2.5-72B-Instruct(GQA, 토큰당 KV 0.33 MB). 입력은 LongBench-v2 32건과 Bailian 트레이스. SSD 쓰기 상한 때문에 디스크는 20% 이상 비워 둠.

#### Directory

| 경로 | 내용 |
| --- | --- |
| `env.sh` | 공통 실행 환경 |
| `lib/` | obs(관측 계층), expfs.py(옛 backend), gdslib.py, scheduler.py, policies.py, value_admission.py, snapshot.py |
| `harness/` | run_bench.py |
| `experiments/01-feasibility/` | 개통, 프리픽스 정당성, cuFile 마이크로벤치, A~E 비교군 |
| `experiments/02-bailian/` | Bailian coder trace 리플레이, staging과 비차단 admission |
| `experiments/03-leval/` | LEval 실제 텍스트, I/O 스케줄러, open-loop, 혼합 admission |
| `experiments/04-admission/` | Prefix Value Admission 시뮬레이션과 게이트 |
| `experiments/05-upstream/` | 종료 race와 /dev/shm 누출 upstream 검증 |
| `experiments/06-weight-offload/` | 66B 가중치 3단 스트리밍, cuFile 대 POSIX, prefetch, host 비율 |
| `experiments/07-combined/` | 가중치 스트리밍 + KV SSD 결합, pinned 정확 등록, 워치독 |
| `experiments/08-cufile-bounce/` | cuFile 미등록 경로의 조각 크기 |
| `experiments/09-kv-policy/` | 배치 구성, 구간 계측, 게이트 A/B, 저장 정책, int8 |
| `experiments/10-model-host-baseline/` | 모델 크기 × host 비율 기준표 |
| `experiments/11-observability/` | 관측 계층 러너, 기본값 런, LMCache 비교 시도 |
| `tools/` | compare_results.py, baseline_table.py |
| `results/` | 원자료 |
| `docs/detailed-log.md` | 설계 근거, 전체 측정표, 정정 기록 |

작업 규칙은 CLAUDE.md와 .claude/skills/compare-results/SKILL.md. 상세는 [docs/detailed-log.md](docs/detailed-log.md).
