# 06 최종 보고서 — vLLM 가중치 3단 오프로드(GPU → pinned CPU → SSD)와 GDS 경로 비교

날짜: 2026-09-07 ~ 09-08(캠페인 24런 + 보강 2런, 06:41 완료) · 환경: rain(Quadro RTX 5000 16 GB, DRAM 125 GiB, NVMe 로컬, MOFED 23.10 + nvidia-fs 2.25.7, 드라이버 570) · 모델: **facebook/opt-66b fp16 132 GB**

---

## 0. 요약

GPU(16 GB)에도 RAM(125 GiB)에도 다 들어가지 않는 모델에서 "SSD에서 가중치를 매 forward 스트리밍"이 실제로 필요해지는 상황을 만들고, 그 SSD 층을 GPU로 옮기는 두 경로를 같은 코드 안에서 비교했다.

- **POSIX**: SSD → O_DIRECT pread → pinned bounce → H2D memcpy → GPU (호스트 경유)
- **cuFile(GDS)**: SSD → cuFileRead → GPU (호스트 미경유, nvidia-fs DMA)

> **답: SSD 층이 80 GiB/step인 기본 구성(host 30 %)에서 decode step은 cuFile 28.5 s vs POSIX 68.0 s(2.4배), 최선 정책(step 2 + 8 스레드 + 등록 ring)에서는 26.7 s vs 68.0 s(2.5배).**
> - 두 경로 모두 매 forward 정확히 같은 877 GiB를 읽고(ssd_stats), 생성 토큰이 모든 런에서 bit-동일하다. 차이는 순수 전송 경로 비용.
> - cuFile 최속 arm은 decode 중 NVMe 읽기 **2.96 GiB/s, util 90 %** = 디스크 한계. POSIX는 util 94 %인데 **1.2 GiB/s**: 스레드마다 pread → H2D → 동기 대기가 직렬화되어 디스크 큐가 비는 구조적 병목.
> - 경로 증명 3중: cuFile TRACE 분류 DIRECT / nvidia-fs 카운터(cuFile만 877 GiB 증가, POSIX 0) / **nsys memcpy 집계(cuFile: SSD 몫이 H2D 0·D2D 863 GB, POSIX: H2D +941 GB)**.
> - 부산물: **upstream vLLM prefetch 오프로더 버그**(모듈 수가 prefetch_step으로 안 나뉘면 패스 경계에서 슬롯 충돌 → garbage 토큰)를 발견·재현·수정(vllm `3fc4433b62`).

## 1. 왜 이 실험인가

앞선 KV 오프로드 실험(01~05)은 모델(opt-2.7b, 5 GiB)이 GPU에 통째로 올라가고 CPU 티어를 인위적으로 8 GiB로 제한해 SSD를 "필요하게 만든" 설계였다. SSD 당위성은 **가중치가 GPU + RAM을 넘을 때** 생긴다. OPT-66B fp16 132 GB는 RAM 125 GiB를 살짝 넘고(비gated, native fp16, sm75에서 bf16 불가) 디스크에도 들어가는 유일한 대중 모델이었다.

실험 파라미터(사용자 결정): host DRAM 예산 = **MemTotal의 30 %**(다른 작업 보호), 성능 런과 nsys 런 분리, 정책 축(prefetch 깊이·IO 스레드·GPU ring)은 KV 실험의 아이디어를 이식.

## 2. 구현 (vLLM `~/vllm` 브랜치 `weight-ssd-offload`)

vLLM의 가중치 오프로드는 UVA(`cpu_offload_gb`, zero-copy pinned)와 Prefetch(`offload_group_size/num_in_group/prefetch_step`, 정적 GPU 버퍼 풀에 H2D prefetch) 둘뿐이고 SSD 티어가 없다. Prefetch 백엔드에 세 번째 티어를 넣었다.

| 커밋 | 내용 |
|---|---|
| `2fbceeb103` | `offloader/ssd_tier.py` 신설: ctypes cuFile 바인딩(시스템 libcufile), 파일 백업 mmap 텐서(`torch.from_file`)로 로더가 디스크에 직접 씀 → fsync + fadvise DONTNEED → O_DIRECT 재오픈; transport `cufile`(cuFileRead → 정적 버퍼) / `posix`(O_DIRECT preadv → 4 KiB 정렬 pinned bounce → non_blocking H2D); 코디네이터 1 + IO 스레드 풀; `prefetch.py` `_layer_mode` 첫맞춤(host 예산 초과분부터 SSD), `_SsdParamOffloader`, `wait_host()`; config/CLI `--offload-ssd-path/--offload-host-fraction/--offload-ssd-transport/--offload-ssd-io-threads` |
| `1c86373b60` | `device_loading_context`가 CPU 파라미터를 pageable 새 텐서로 되돌려 오프로드 저장소와 이중 존재(anon RSS 72 GB → OOM kill) → 원래 저장소에 제자리 `copy_` |
| `fbfc637cb0` | `--offload-ssd-ring-mb`: 등록된 GPU ring(2 × io_threads 슬롯)에 cuFileRead 직접 DMA → D2D. 정적 버퍼(680 MB fc1/fc2)가 BAR1 256 MB에 등록 불가한 GPU 우회 |
| `3fc4433b62` | **prefetch 경계 슬롯 충돌 수정**(§6) |

제약: V1 model runner(`VLLM_USE_V2_MODEL_RUNNER=0`, V2에는 prefetch 오프로더가 연결돼 있지 않음) + `enforce_eager`(호스트 구동 읽기는 CUDA graph 캡처 불가, 캡처 시 RuntimeError).

OPT-66B 배치(group 64 / num_in_group 61): GPU 상주 3층, 오프로드 61 모듈(층당 fp16 1.90 GiB). host 30 % = 37.6 GiB 예산 → host 19층(36.1 GiB), SSD 42층(79.7 GiB, 168 파일). 정적 버퍼 풀 1.9 GiB(step 1) / 3.8 GiB(step 2). KV 5.55 GiB(2,512 tok).

## 3. 검증

- **QA(opt-2.7b, `run_qa.sh`)**: baseline / cpu / ssd-posix / ssd-cufile 4 arm 토큰 완전 일치. cuFile 경로 TRACE 분류 DIRECT(px_io 0, bounce 0). ring 모드(`smoke_ring.py`)도 PASS.
- **66B 전 런 토큰 일치**: `summarize_66b.py`가 모든 json의 생성 id를 기준 런(c-h0.3-r1)과 비교. 수정 전 step 2 런(§6) 하나를 제외하고 전부 일치.
- **읽기량 일치**: 모든 arm에서 `ssd_stats` 877.08 GiB(forward 11회 × 79.74 GiB). cuFile arm은 nvidia-fs `Reads.readMiB` 델타가 이와 일치, POSIX arm은 0.
- **nsys(§5.4)**: 경로별 memcpy 종류·총량이 설계와 정확히 일치.

## 4. 결과 (OPT-66B, prefill 4 × 256 tok, decode 8 tok, 중앙값)

`python3 summarize_66b.py` 출력. 이상치(같은 arm 최소 load 대비 8 %+ 느린 런 = 런 전체가 느려진 시스템 지연, 2건)는 제외.

### 4.1 기본 구성(host 0.3) — transport × 정책

| arm | n | load s | prefill s | decode step s | tok/s | CPU s | nvfs reads | 평균 IO |
|---|---|---|---|---|---|---|---|---|
| cuFile step1 thr4 | 3 | 508 | 31.1 | **28.5** | 0.14 | 159 | 826,056 | 1.1 MiB |
| cuFile step1 thr4 ring16 | 3 | 516 | 31.4 | 29.8 | 0.13 | 133 | 112,728 | 8.2 MiB |
| cuFile step2 thr8 | 2 | 503 | 29.9 | 29.7 | 0.14 | 158 | 863,016 | 1.1 MiB |
| cuFile step2 thr8 ring8 | 3 | 517 | 27.4 | **26.7** | 0.15 | 137 | 112,728 | 8.2 MiB |
| POSIX step1 thr4 | 3 | 582 | 72.0 | **68.0** | 0.06 | 146 | 0 | – |
| POSIX step2 thr8 | 3 | 574 | 70.0 | 68.2 | 0.06 | 142 | 0 | – |

- 반복 편차: cuFile 28.1/29.1/28.5, POSIX 68.0/68.0/69.0 → 3 % 이내.
- step2 ring8 3반복: 26.7/26.7/26.7 s(prefill 27.3/27.5/27.4) — 가장 안정.

### 4.2 host fraction 스윕(step1 thr4, 1회)

| host | host / SSD 층 | SSD GiB/step | cuFile decode s | POSIX decode s | 배수 | decode 중 NVMe(cuFile) |
|---|---|---|---|---|---|---|
| 0.1 | 6 / 55 | 104.4 | 56.5 | 87.8 | 1.6× | 2.07 GiB/s, util 96 % |
| 0.3 | 19 / 42 | 79.7 | 28.5 | 68.0 | 2.4× | 2.96 GiB/s, util 90 % |
| 0.5 | 33 / 28 | 53.2 | 22.8 | 49.4 | 2.2× | 2.40 GiB/s, util 74 % |

## 5. 해석

### 5.1 cuFile은 디스크 한계, POSIX는 bounce 직렬화 한계
sysmon(30 s 스냅샷)으로 decode 구간의 NVMe를 보면 cuFile 최속 arm은 2.96 GiB/s·util 90 %(80 GiB / 27 s = 2.96 GiB/s와 일치)로 **SSD 읽기 대역폭에 도달**했다. POSIX는 util 94 %인데도 1.2 GiB/s에 그친다. 스레드마다 파일 전체를 pread한 뒤 H2D를 `copy_stream.synchronize()`로 기다리므로 그동안 그 스레드 몫의 디스크 큐가 비고, 4~8 스레드로는 큐를 채우지 못한다. 그래서 POSIX는 prefetch 깊이·스레드 수를 바꿔도 68 s에 고정된다(4.1).

### 5.2 정책 축: step 2와 ring은 "함께"일 때만 이득
- ring 단독(step1 thr4): DMA 횟수 1/7, CPU −17 %지만 벽시계는 같다(병목이 디스크).
- step2 thr8 단독: 오히려 29.7 s. 8 스레드가 cuFile 내부 1 MiB GPU 캐시(미등록 버퍼 경로)를 두고 경합.
- step2 thr8 + ring8: 26.7 s(−6 %), prefill 27.4 s(−12 %). 두 슬롯으로 읽기/계산이 한 층 더 겹치고, ring이 8 스레드의 1 MiB 캐시 경합을 8 MiB 직접 DMA로 치환. 이 arm에서 디스크가 포화되므로 여기가 이 하드웨어의 상한.
- ring 크기는 BAR1(256 MiB)이 정한다: 16 MiB × 2 × 8 스레드 = 256 MiB는 13/16만 등록(dmesg `no space for BAR1 mappings`) → 8 MiB.
- step 2는 정적 버퍼 풀 3.8 GiB가 vLLM 메모리 프로파일에 잡히지 않아 KV 산정 후 OOM → `gpu_memory_utilization` 0.75로 여유 확보(KV 1,456 tok, 필요 1,056).

### 5.3 host fraction
decode는 SSD 층 수에 거의 비례한다(0.5 → 0.3 → 0.1: 22.8 → 28.5 → 56.5 s). "GPU를 넘친 만큼을 RAM에 얼마나 두느냐"가 1차 변수이고 그 위에서 cuFile 배수(2.2~2.4×)가 유지된다. 0.1의 1.6×는 디스크 85 % 점유 상태에서 104 GiB를 막 쓴 직후 NVMe 읽기가 2.07 GiB/s(util 96 %)로 떨어져 디스크 의존적인 cuFile이 더 깎인 결과 — 단서를 달아야 한다.

### 5.4 nsys memcpy 집계(결정적 증거, forward 11회)

| 종류 | cuFile | POSIX |
|---|---|---|
| Host-to-Device | 560 GB / 7,405회 | 1,501 GB / 9,253회 |
| Device-to-Device | 863 GB / 824,233회(1 MiB) | 0 |
| Device-to-Host | 163 GB(로드 시 오프로드) | 163 GB |

H2D 560 GB는 공통(초기 로드 132 GB + host tier 19층 × 1.9 GiB × 11). POSIX에만 SSD 877 GiB(=941 GB)가 H2D로 더해지고, cuFile에서는 같은 양이 GPU 안의 D2D(cuFile 내부 캐시 → 정적 버퍼)로만 나타난다. 즉 GDS 경로는 호스트 메모리를 거치지 않았다. nsys 런 자체의 시간(cuFile 45.5 s, POSIX 70.7 s)은 프로파일러 오버헤드가 커서 참고용.

### 5.5 CPU
cuFile step1 159 s vs POSIX 146 s로 POSIX가 오히려 낮다. cuFile 미등록 버퍼 경로는 1 MiB 단위 호출 82만 번을 호스트가 돌려야 하고, POSIX는 4 스레드가 큰 preadv를 기다리는 시간이 대부분이다. ring(8 MiB DMA)을 쓰면 cuFile CPU가 133 s로 내려간다.

## 6. 발견한 upstream 버그: prefetch 경계 슬롯 충돌

step 2 첫 런의 토큰이 처음부터 garbage("\n," 반복)였다. 원인은 SSD 티어가 아니라 upstream `PrefetchOffloader` 스케줄이었다.

- 슬롯 배정 `idx % step`, prefetch 대상 `(i + step) % n`. **n % step ≠ 0**(61 모듈, step 2)이면 layer 59가 layer 0을 slot 0에 prefetch하는 동안 아직 실행 안 된 layer 60(slot 0)이 덮어써진 가중치로 계산된다.
- 재현: opt-2.7b group 32 / num 31(31 모듈) × step 2 → host tier만 써도 FAIL(`repro_wrap.sh`). 기존 QA는 24 모듈(짝수)이라 잡히지 않았다.
- 수정(`3fc4433b62`): post_init에서 레이어별 정적 prefetch 계획(`_build_prefetch_plan`)을 만들고, 경계를 넘는 대상은 같은 슬롯을 쓰는 뒷 레이어가 남아 있으면 마지막 레이어로 미룬다. n % step == 0이면 기존 순환 스케줄과 동일. 수정 후 31 모듈 ALL PASS, step 1 회귀 PASS, 66B step 2 토큰 일치.
- upstream PR 후보(작성 시 요약: 최소 재현 = 임의 모델 + `offload_group_size` 로 홀수 모듈 + `offload_prefetch_step 2`).

## 7. 함정·교훈

- **BAR1 256 MiB**: cuFileBufRegister는 정적 버퍼 4개 중 1개만 성공. 나머지는 cuFile 내부 캐시(여전히 native DMA, 1 MiB 단위). ring 슬롯 총량도 BAR1 안이어야 한다.
- **정적 버퍼 풀은 프로파일 밖**: step ≥ 2는 gpu_util을 낮춰야 한다.
- **시스템 지연 이상치**: 24 런 중 2 런이 load까지 포함해 런 전체가 2배 느렸다(시작 메모리 상태는 정상과 동일, cron/timer 없음, sysstat 미수집). 반복 3회 + 중앙값 + load 기준 이상치 제외가 필요했다.
- **SSD 상태**: 디스크 점유율·직전 대량 쓰기가 읽기 대역폭을 바꾼다(0.1 스윕). 런마다 80~104 GiB를 다시 쓰는 설계라 디스크 여유 ≥ 90 G 가드를 두었다.
- **cuFile compat 폴백**: 9/4 정비로 MOFED 패치 nvme가 사라져 cuFile이 POSIX로 조용히 폴백했었다. TRACE 분류 + nvidia-fs 카운터로 매 런 경로를 확인해야 한다(`rain-server-state` 참조).
- `device_loading_context`의 pageable 되돌림(§2)과 `run_66b.sh` 실행 중 편집 금지(bash가 읽는 중), `pkill -f`가 자기 셸을 죽이는 문제(패턴에 `[x]` 사용).

## 8. 재현·파일

```
source ~/experiments/vllm-gds-kv/env.sh; export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
cd ~/experiments/vllm-gds-kv/experiments/06-weight-offload
./run_qa.sh                         # opt-2.7b 4 arm QA
python smoke_ring.py 16 1 4         # ring QA
./repro_wrap.sh after-fix           # §6 재현/검증
./campaign.sh                       # 66B 전체(phase A~F, 런당 14~30분), 결과 results/weight-offload/opt66b/<tag>.json
python3 summarize_66b.py            # arm별 중앙값 표 + 토큰 일치
```

- 결과: `results/weight-offload/opt66b/` (json·log·campaign.log·sysmon.log·nsys csv), 이상치 런은 표에서 자동 제외.
- vLLM: `~/vllm` 브랜치 `weight-ssd-offload`(커밋 `2fbceeb103`, `1c86373b60`, `fbfc637cb0`, `3fc4433b62`). 주의: 그 아래 `77c4033078`이 requirements의 torch 라인을 지워 놓음(sed 사고, 런타임 무관, upstream diff 전 `git checkout 568afb3a13 -- pyproject.toml requirements/`).
