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

### 2.1 정적 버퍼와 BAR1 — 왜 4개 중 1개만 등록되는가

정적 버퍼는 체크포인트의 일부가 아니라 prefetch 오프로더가 GPU에 한 번 잡아 두고 계속 재사용하는 **한 층 분량의 착지 공간**이다. 오프로드된 61개 층의 가중치 원본은 pinned CPU(19층)와 SSD 파일(42층)에 있고, 층 L을 계산하기 직전에 그 층의 행렬들을 이 버퍼로 읽어 넣은 뒤 파라미터의 `.data`가 버퍼를 가리키게 한다. 계산이 끝나면 다음 층이 같은 자리를 덮어쓴다(prefetch_step = 슬롯 수, step 2면 두 벌 3.8 GiB).

버퍼는 행렬(파라미터 텐서) 단위다. 행렬곱 커널이 행렬 전체를 연속 메모리로 읽으므로 쪼갤 수 없고, 크기는 모델 형상이 정한다. OPT-66B 디코더 층 하나(hidden 9216)의 구성:

| 행렬 | 역할 | 모양(fp16) | 크기 | BAR1 256 MiB 등록 |
|---|---|---|---|---|
| qkv_proj | 어텐션 입력 투영(Q·K·V 세 행렬을 이어 붙인 것) | 27648 × 9216 | 486 MiB | 실패 |
| out_proj | 어텐션 출력 투영 | 9216 × 9216 | 162 MiB | **성공** |
| fc1 | MLP 첫 층(hidden → 4×, ReLU) | 36864 × 9216 | 648 MiB | 실패 |
| fc2 | MLP 둘째 층(4× → hidden) | 9216 × 36864 | 648 MiB | 실패 |

LayerNorm·bias 같은 1 MiB 미만 파라미터는 오프로드해도 pinned CPU에 두고 매번 복사한다(SSD 파일로 만들 가치가 없음).

`cuFileBufRegister`는 GPU 버퍼를 nvidia-fs가 DMA 대상으로 쓰도록 BAR1 창에 고정 매핑(`nvidia_p2p_get_pages_persistent`)한다. BAR1은 PCIe에서 GPU 메모리를 직접 보는 창으로, 데이터센터 GPU(A100 등)는 VRAM 전체 크기지만 Turing 워크스테이션 카드(Quadro RTX 5000)는 256 MiB 고정이며 Resizable BAR도 없다. 162 MiB인 out_proj만 들어가고 나머지 셋은 각각 단독으로도 256 MiB를 넘어 순서와 무관하게 실패한다. dmesg 증거:

```
NVRM: RmThirdPartyP2PBAR1GetPages: no space for BAR1 mappings, length: 0x1000000
nvidia-fs: nvfs_pin_gpu_pages: Error ret -12 invoking nvidia_p2p_get_pages_persistent
```
(ring 16 MiB × 16 슬롯을 등록하다 13개째 이후 실패한 기록: 13 × 16 = 208 MiB + 기존 매핑 ≈ 256 MiB.)

등록에 실패해도 POSIX로 떨어지지는 않는다. cuFile은 미등록 목적지에 대해 자기 내부의 등록된 GPU 캐시(1 MiB 조각)로 DMA한 뒤 D2D로 옮긴다. nsys의 D2D 863 GB / 1 MiB × 824k회, nvidia-fs 읽기 826,056회가 그 흔적이다. **ring 모드는 이 우회를 명시적으로 크게 만든 것**이다: BAR1에 들어가는 8 MiB × 16 슬롯(128 MiB)만 등록해 거기로 DMA한 뒤 정적 버퍼로 D2D 복사하면 1 MiB 82만 번이 8 MiB 11만 번이 되어 CPU −17 %, 8 스레드 경합 해소 → 최속 arm(26.7 s).

조정 가능한 축: 슬롯 수(`offload_prefetch_step`), 오프로드 대상(`offload_params`), ring 슬롯 크기(`offload_ssd_ring_mb`, 슬롯 × 2 × 스레드 ≤ BAR1). 조정 불가: 버퍼 하나의 크기(모델 형상), BAR1 크기(하드웨어). BAR1이 큰 GPU라면 정적 버퍼를 그대로 등록해 ring 없이 직접 DMA가 된다.

### 2.2 vLLM에서 무엇을 어디에 고쳤나 (GDS 경로가 생기기까지)

기본 vLLM(v0.26.1.dev, 568afb3a13)에는 SSD 티어가 없다. 가중치 오프로드는 `vllm/model_executor/offloader/` 아래 두 백엔드뿐이다.

- **UVA**(`uva.py`): `cpu_offload_gb`만큼 파라미터를 pinned CPU에 두고 GPU가 zero-copy로 읽는다. 매 커널이 PCIe를 건너므로 느리고 디스크 개념이 없다.
- **Prefetch**(`prefetch.py`, SGLang의 `offloader.py`를 가져온 것): `offload_group_size` / `offload_num_in_group`으로 층을 골라 pinned CPU에 두고, 정적 GPU 버퍼 풀(§2.1)에 `copy_stream`으로 H2D prefetch한다. 각 층 forward를 감싸 `torch.ops.vllm.wait_prefetch(idx)` → forward → `start_prefetch(idx+step)` 순서로 호출한다(`prefetch_ops.py`의 custom op, torch.compile 호환용).

**수정의 성격.** 별도 스크립트를 얹은 것이 아니라 `~/vllm`에 editable 설치된 vLLM 소스 트리 자체를 고쳤다(브랜치 `weight-ssd-offload`, 커밋 4개). 그래서 사용자 쪽에서는 공식 진입점을 그대로 쓴다: `LLM(model=..., offload_ssd_path=..., offload_ssd_transport="cufile")` 또는 `vllm serve --offload-ssd-path ... --offload-ssd-transport cufile`. 실험 repo의 `run_66b.py`·`qa_ssd.py`는 그 진입점을 호출해 측정만 하는 얇은 스크립트다. 변경은 전부 파이썬이며 C++/CUDA 커널은 건드리지 않았다. cuFile은 새 확장을 컴파일한 것이 아니라 ctypes로 시스템 `libcufile.so`를 직접 호출한다. upstream에는 아직 올리지 않았다(로컬 브랜치, 경계 버그 수정은 PR 후보).

**삽입 지점은 Prefetch 백엔드의 세 곳**이었다. (1) 파라미터를 CPU로 내리는 순간(`_CpuParamOffloader._offload_to_cpu_internal`), (2) forward 직전에 정적 버퍼를 채우는 한 줄(`start_onload_to_static`의 `gpu_buffer.copy_(cpu_storage)`), (3) 그 완료를 기다리는 곳(`_wait_for_layer`). 바뀐 파일과 규모:

| 파일 | 변경 | 무엇을 |
|---|---|---|
| `offloader/ssd_tier.py` | +410 (신규) | cuFile 바인딩, 파일 티어, 읽기 경로 두 개, ring |
| `offloader/prefetch.py` | +253 | 층별 티어 배정, SSD 파라미터 오프로더, 호스트 대기, prefetch 계획(§6) |
| `config/offload.py`, `engine/arg_utils.py` | +31, +25 | 설정 필드 5개와 CLI 플래그 |
| `model_loader/utils.py` | +17 | 로더가 CPU 파라미터를 제자리 갱신하도록 |
| `offloader/base.py` | +5 | 새 필드를 PrefetchOffloader 생성자에 전달 |

(`v1/kv_offload/cpu/shared_offload_region.py`의 변경은 앞선 KV 실험의 shm 누수 수정이며 이 실험과 무관하다.)

**데이터 흐름을 시간순으로 보면:**

1. **모델 구성 시 티어 배정** — `PrefetchOffloader.wrap_modules`가 오프로드 대상 층을 고를 때 `_layer_mode()`가 층 크기를 누적해 host 예산(`offload_host_fraction × MemTotal`, /proc/meminfo)을 넘기 전까지는 `"cpu"`, 넘긴 뒤부터는 `"ssd"`를 준다(앞 층부터 채우는 첫맞춤). `"ssd"` 층이라도 1 MiB 미만 파라미터(LayerNorm, bias)는 `"cpu"`로 남긴다(`SSD_MIN_PARAM_BYTES`).
2. **가중치 로드 — 파일에 직접 쓰기** — `"ssd"` 파라미터는 `_SsdParamOffloader._offload_to_cpu_internal`이 pinned 텐서 대신 `SsdTier.new_file_tensor()`가 만든 **파일 백업 mmap 텐서**(`torch.from_file(shared=True)`, 파일은 `<ssd_path>/rank0/<layer>/<param>.bin`)를 파라미터에 꽂는다. 체크포인트 로더는 평소처럼 `param.copy_()`를 하는데 그 목적지가 mmap이라 **가중치가 DRAM을 거치지 않고 디스크에 내려간다**. 132 GB 모델을 125 GiB RAM에서 한 번도 통째로 들지 않는 이유다.
3. **로드 후 처리와 제자리 갱신** — vLLM의 `device_loading_context`는 `process_weights_after_loading`을 위해 CPU 파라미터를 GPU로 올렸다가 되돌리는데, 원래 코드는 되돌릴 때 **새 pageable 텐서**를 만든다. 오프로더가 쥔 원본(pinned/mmap)과 이중으로 존재해 anon RSS 72 GB → OOM kill이 났다. `model_loader/utils.py`를 고쳐 모양·dtype·stride가 같으면 원본 저장소에 `copy_()`로 제자리 갱신하게 했다(커밋 1c86373b60).
4. **post_init: 파일 확정과 등록** — `_SsdParamOffloader.assign_static_buffer`가 정적 버퍼를 배정한 뒤 `SsdTier.finalize()`를 부른다: `fsync` → `posix_fadvise(DONTNEED)`로 페이지 캐시에서 내보냄 → mmap 해제 → **O_DIRECT로 재오픈** → `cuFileHandleRegister`. 이어서 `register_buffers()`가 정적 버퍼에 `cuFileBufRegister`를 시도한다(§2.1, best-effort). ring 모드면 대신 ring 슬롯을 등록한다.
5. **forward마다 읽기** — `start_onload_to_static`에서 `"cpu"` 파라미터는 기존대로 `copy_stream`에 H2D를 넣고, `"ssd"` 파라미터는 `(SsdFile, gpu_buffer)` 목록으로 모아 `SsdTier.submit_layer(items, fork_event, done_event)`에 넘긴다. 코디네이터 스레드 1개가 층 단위 job을 직렬로 받아, `fork_event.synchronize()`로 **이 슬롯을 쓰던 이전 층의 커널이 끝날 때까지 호스트에서 기다린 뒤** IO 스레드 풀(`offload_ssd_io_threads`)로 파라미터별 읽기를 병렬 실행하고, 끝나면 `done_event.record(copy_stream)`을 찍는다.
   - **cufile**: `cuFileRead(fh, gpu_ptr, nbytes, 0)` 한 번. 등록된 버퍼면 직접 DMA, 아니면 cuFile 내부 GPU 캐시(1 MiB) 경유 후 D2D.
   - **cufile + ring**: 파일을 `ring_mb` 조각으로 나눠 등록된 ring 슬롯(큐로 관리)에 `cuFileRead` → 스레드별 스트림에서 정적 버퍼로 D2D → 슬롯 반납.
   - **posix**: 스레드별 4 KiB 정렬 pinned bounce에 `os.preadv`(O_DIRECT) → `copy_stream`에 non_blocking H2D → `copy_stream.synchronize()` 후 bounce 재사용.
6. **forward 직전 대기** — `_wait_for_layer`가 먼저 `offloader.wait_host()`(job Future 완료)를 부르고, 그 다음 기존처럼 `current_stream.wait_event(done_event)`를 건다. 호스트 구동 IO라 이벤트만으로는 순서를 보장할 수 없어서 넣은 단계다.

**왜 V1 러너와 enforce_eager인가.** V2 model runner는 `set_offloader`를 아예 호출하지 않아 prefetch 오프로더 자체가 붙지 않는다(`VLLM_USE_V2_MODEL_RUNNER=0`). 그리고 cuFileRead는 동기 호스트 호출이라 CUDA graph 캡처 안에 들어갈 수 없다. `start_onload_to_static`이 캡처 중이면 RuntimeError를 내고, 실험은 `enforce_eager=True`로 돈다.

**cuFile을 어떻게 붙였나.** `cuda-python`의 cufile 모듈은 시스템 libcufile과 이중 로드되어 충돌했으므로 `ctypes`로 `/usr/local/cuda/.../libcufile.so.0`을 직접 열고 `cuFileDriverOpen / HandleRegister / BufRegister / Read`만 바인딩했다(`CuFile` 싱글턴, 약 80줄). 경로가 진짜 GDS인지는 코드가 판정할 수 없으므로 매 런 `CUFILE_ENV_PATH_JSON`으로 TRACE 로그를 켜 `cufio-px`(compat POSIX) / `read_through_bounce_buffer` / 둘 다 없음(DIRECT)으로 분류하고, `/proc/driver/nvidia-fs/stats`의 `Reads.readMiB` 델타가 `ssd_stats`와 맞는지 확인한다.

**설정 표면.** `PrefetchOffloadConfig`에 `offload_ssd_path`, `offload_host_fraction`(0.3), `offload_ssd_transport`(`cufile`|`posix`), `offload_ssd_io_threads`(4), `offload_ssd_ring_mb`(0) 다섯 필드를 추가하고 `--offload-ssd-*` CLI 플래그와 `LLM(...)` kwargs로 노출했다. `offload_ssd_path`는 `offload_group_size > 0`(prefetch 백엔드)일 때만 유효하도록 validator를 두었다.

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

## 9. 선행 사례와 이 실험의 위치

"GPU에 안 들어가는 LLM 가중치를 SSD에 두고 GPU로 스트리밍"은 여러 팀이 했고, 그중 GDS(cuFile)로 호스트를 건너뛴 사례도 있다. 우리가 한 것과의 차이는 **vLLM 안에 넣었다는 점**과 **같은 코드에서 POSIX bounce와 cuFile을 스위치 하나로 바꿔 경로 비용만 분리해 측정했다는 점**이다.

| 사례 | 무엇을 | SSD→GPU 경로 | 우리와의 관계 |
|---|---|---|---|
| **FlexGen** (ICML 2023, [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)) | 단일 T4로 OPT-175B. 가중치·KV·활성화를 GPU/CPU/디스크에 배치하는 선형계획 + 4-bit 압축 | 디스크 → CPU → GPU(호스트 경유) | 배치 크기를 키워 처리량을 노리는 설계. 경로 비교는 없음 |
| **DeepSpeed ZeRO-Inference / DeepNVMe** ([2022 블로그](https://www.deepspeed.ai/2022/09/09/zero-inference.html), [DeepNVMe 2025-06](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepnvme/06-2025/README.md), [PyTorch 블로그](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/)) | 가중치를 DRAM/NVMe에 두고 층 단위로 가져옴. **AIO(CPU bounce)와 GDS 두 모드** 제공, SGLang에 통합 | 둘 다 | 가장 가까운 선행. H200 + Gen5 NVMe 4~8장으로 Llama-3-70B에서 GDS 모드 7 → 17 → 26 tok/s(디스크 수에 비례) 보고. 우리 실험(01~05, gds-llm-demo)에서도 ZeRO-Inference GDS로 decode +40 %를 재현했었다. 다만 GPU/host 비율 knob이 없어(전량 NVMe) "host 30 %" 같은 3단 배치는 못 한다 |
| **Endor** ([arXiv 2406.11674](https://arxiv.org/pdf/2406.11674)) | 오프로드 추론용 희소 압축 포맷. **OPT-66B·Llama2-70B**로 HF Accelerate 대비 1.70×, 여기에 SSD→GPU 직접 전송(GDS)을 더해 2.25× | GDS | 같은 모델(OPT-66B)에서 "직접 전송만으로 약 1.3×"를 보고. 우리는 압축 없이 경로만 바꿔 2.4~2.5× |
| **I/O 특성 연구** (CHEOPS 2025, [PDF](https://atlarge-research.com/pdfs/2025-cheops-llm.pdf)) | DeepSpeed(OPT-13B)·FlexGen(OPT-30B)의 NVMe 오프로드 I/O를 계측 | 호스트 경유만 | 결론이 우리 5.1과 같다: **CPU bounce buffer와 작은 I/O 때문에 SSD 피크에 한참 못 미침**, 직접 GPU-스토리지 경로가 해법일 것이라 제안. 우리는 그 제안을 실측으로 확인한 셈 |
| **LLM in a flash** (Apple, ACL 2024, [링크](https://machinelearning.apple.com/research/efficient-large-language)) | 플래시에 가중치를 두고 FFN 희소성·윈도잉·row-column bundling으로 읽는 양과 횟수를 줄임 | 플래시 → DRAM(모바일/Mac) | 접근 패턴 최적화 쪽. GPU DMA 경로는 아님 |
| **DAK** ([arXiv 2604.26074](https://arxiv.org/pdf/2604.26074)) | prefetch 대신 GPU TMA로 원격 메모리에서 SMEM으로 직접 가져옴 | 원격 메모리(NVLink-C2C/PCIe) 직접 접근 | prefetch 자체를 없애는 방향. 우리는 prefetch 계열 |
| **TERAIO** ([arXiv 2506.06472](https://arxiv.org/abs/2506.06472)), **SSDTrain** ([arXiv 2408.10013](https://arxiv.org/pdf/2408.10013)) | 학습 시 텐서/활성화를 GDS로 SSD에 오프로드 | GDS | 추론이 아니라 학습. GDS 사용법은 동일 |
| **MoE SSD 오프로드 에너지 분석** ([arXiv 2508.06978](https://arxiv.org/pdf/2508.06978)), llama.cpp [Expert-Aware SSD Streaming 논의](https://github.com/ggml-org/llama.cpp/discussions/27149) | MoE 전문가 가중치만 SSD에서 토큰마다 스트리밍 | 다양 | dense 모델 전체를 스트리밍하는 우리와 달리 활성 전문가만 읽어 양 자체를 줄이는 쪽 |
| **vLLM 쪽 논의** ([RFC #38256 MoE expert offloading](https://github.com/vllm-project/vllm/issues/38256), [vllm-omni #754 layerwise CPU offloading](https://github.com/vllm-project/vllm-omni/issues/754)) | CPU pinned + GPU 캐시 기반 전문가 오프로드, 층 단위 CPU 오프로드 | CPU까지만 | upstream에는 아직 SSD/GDS 티어가 없다. 우리 `weight-ssd-offload`가 그 자리를 채운 형태 |

정리하면: SSD 스트리밍 자체(FlexGen, ZeRO-Inference)와 GDS 적용(DeepNVMe, Endor, TERAIO)은 선례가 있고, "bounce buffer가 병목"이라는 진단(CHEOPS)도 있다. 새로운 부분은 ① vLLM prefetch 오프로더 위에 host 비율이 파라미터인 3단 티어를 구현한 것, ② 동일 코드·동일 읽기량·동일 토큰을 보장한 상태에서 transport만 바꿔 nsys memcpy 집계로 경로를 증명한 것, ③ BAR1 256 MiB짜리 워크스테이션 GPU에서 ring으로 직접 DMA를 살린 것, ④ 그 과정에서 upstream prefetch 스케줄 버그를 찾은 것이다.
