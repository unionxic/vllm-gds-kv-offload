### vLLM 가중치 스트리밍 + GDS KV 오프로드

16 GB GPU 한 장으로 145 GB 모델(Qwen2.5-72B-Instruct)을 돌리는 조건에서, 가중치를 host memory와 NVMe에서 매 forward 스트리밍하고 KV를 NVMe에 저장·적중시키는 실험. vLLM 포크(~/vllm, weight-ssd-offload) 안에 구현하고 외부 코드는 쓰지 않음.

| 항목 | 값 |
| --- | --- |
| GPU | Quadro RTX 5000 16 GB. BAR1 256 MiB(카드 최대)라 cuFile은 드라이버 bounce 경로 |
| RAM, SSD | 125 GiB, Samsung 970 EVO 500 GB(OS와 공유). host→GPU 12.3 GB/s, SSD→GPU 3.4 GB/s, 둘을 동시에 쓰면 host 8.5 GB/s |
| 소프트웨어 | CUDA 12.8, 드라이버 570, cuFile 1.13, nvidia-fs 2.25.7, vLLM 0.26.1 기반 포크, nsys 2025.3 |
| 모델 | Qwen2.5-72B-Instruct fp16(145 GB, 80 layer × 1.63 GiB, KV 토큰당 328 KB). 소형 검증은 Qwen2.5-3B |
| 입력 | Bailian 트레이스 32건(프리픽스 공유 56%, 8k 상한), LongBench-v2 32건(8k), 트레이스 시간 순서 리플레이 128건 |

#### 핵심 결론

- forward 하나는 가중치 이동이다. 오프로드된 131 GiB가 매 forward GPU로 들어오며, 기본 배치(host layer 앞, SSD layer 뒤, prefetch_step 1)에서는 host 복사 5.7 s와 SSD 읽기 23.3 s가 차례로 일어나 decode forward 28.4 s.
- 가중치 티어 교차 배치 + prefetch_step 2: host layer를 SSD layer 사이에 고르게 끼워 넣고 다음 두 layer를 동시에 가져오면 host 복사(PCIe)와 SSD 읽기(NVMe)가 같이 흘러 forward = 둘 중 긴 쪽. RAM 0.5에서 22.2 s.
- host 비율을 전송 대역폭 비율(8.5 : 3.4)에 맞춤: RAM 0.72(host 55 : SSD 25 layer)에서 host 복사 11.3 s와 SSD 읽기 13.7 s가 같이 끝나 decode forward 14.0 s. 16 GB 링크로 131 GiB를 옮기는 하한이 11.7 s.
- KV는 GPU에서 NVMe로 바로 저장하고 적중 시 바로 읽음. 적중이 아끼는 것은 prefill 계산뿐이며, 요청 하나의 KV 2.6 GB 쓰기가 SSD 쓰기 캐시(약 4 GB) 안에 들어가 저장 비용은 0. 스케줄러 wave gate가 적재 완료 요청을 한 forward에 묶어 적중 단계 forward 67 → 50.
- 정책 조합(교차 배치 + prefetch_step 2 + RAM 0.72 + wave gate + write-behind + 전부 저장): Bailian 32건 두 단계 합계 4,350 → 2,448 s(−44%), LongBench 32건 12,652 → 6,527 s(−48%). 출력 토큰열은 재계산과 동일(LongBench 64건 중 1건은 근소 차 토큰 순서 뒤바뀜).
- 이득이 없던 것: host memory를 KV에 주는 것(가중치가 SSD로 밀려 forward +2 s), SSD 용량 상한 축출(LRU/LFU, 역순 재방문에서 쓰기 2배), seen_twice admission(첫 재사용을 잃음), LMCache 방식 host 8 GB 층(SSD 전부 저장과 4 s 차이), compute/load split(적재가 재계산보다 20배 빨라 최적 k=0).

#### 측정

Qwen2.5-72B, RAM 0.5, Bailian 32건, vLLM 기본값(chunk 8192, GPU KV 자동 21.4k)

| 조건 | 저장 단계 | 적중 단계 | 두 단계 합계 |
| --- | --- | --- | --- |
| 재계산 (기준) | 2,280 s / 54 fwd | 2,070 s / 53 fwd | 4,350 s |
| SSD 전부 저장 | 2,355 s / 59 fwd | 1,677 s / 59 fwd | 4,032 s (−7.3%) |
| LFU 8 GiB 상한 | 2,262 s | 1,934 s | 4,196 s |
| LRU 8 GiB 상한 | 2,260 s | 1,989 s | 4,249 s |
| seen_twice admission | 2,251 s | 1,919 s | 4,169 s |
| host memory 8 GB를 KV에 (가중치 54.7 GB) | 2,374 s | 2,104 s | 4,478 s (+2.9%) |

Qwen2.5-72B, Bailian 32건, chunk 4096·GPU KV 18.8k 고정(prefetch_step 2가 들어가는 조건)

| 조건 | 저장 단계 | 적중 단계 | 두 단계 합계 | decode forward |
| --- | --- | --- | --- | --- |
| 재계산, 교차 배치 + prefetch_step 2, RAM 0.5 | 2,000 s / 69 fwd | 1,880 s / 68 fwd | 3,880 s | 22.2 s |
| SSD 전부 저장, 기본 배치 | 2,630 s / 69 fwd | 1,910 s / 67 fwd | 4,541 s | 28.3 s |
| LMCache 방식 카피(host 8 GB LRU + SSD write-through) | 2,630 s / 69 fwd | 1,907 s / 67 fwd | 4,537 s | 28.3 s |
| 정책 조합, RAM 0.5 | 1,963 s / 69 fwd | 1,115 s / 50 fwd | 3,078 s | 22.1 s |
| 재계산, 교차 배치 + prefetch_step 2, RAM 0.72 | 1,910 s / 69 fwd | 1,657 s / 68 fwd | 3,567 s | 14.7 s |
| 정책 조합, RAM 0.72 | 1,635 s / 69 fwd | 812 s / 50 fwd | 2,448 s | 14.0 s |

Qwen2.5-72B, LongBench-v2 32건(실제 문서, 8k), chunk 4096·KV 18.8k, RAM 0.72

| 조건 | 저장 단계 | 적중 단계 | 두 단계 합계 | decode forward |
| --- | --- | --- | --- | --- |
| 재계산, 기본 배치, prefetch_step 1 | 6,446 s / 161 fwd | 6,206 s / 159 fwd | 12,652 s | 21.5 s |
| 정책 조합 | 4,686 s / 161 fwd | 1,841 s / 128 fwd | 6,527 s (−48%) | 14.0 s |

전송 대역폭(experiments/12-channels, 4 GiB 버퍼)

| 경로 | 단독 | 동시 실행 |
| --- | --- | --- |
| host→GPU pinned | 12.3 GB/s | SSD 읽기와 같이 쓰면 8.0~8.5 GB/s(같은 PCIe 링크를 bounce가 공유) |
| SSD→GPU cuFile | 2.9~3.6 GB/s | 유지 |
| SSD 읽기 + 쓰기 | | 읽기 0.28~0.54배, 쓰기 0.33배 |
| GPU 계산 | 69 TFLOPS | 어느 전송과도 독립 |

Qwen2.5-3B compute/load split parameter sweep(적중 단계 wall clock, s): 같은 재계산 비율 k에서 겹침(split)이 직렬(serial)보다 8~16 s 빠르나 최적은 k=0. 표는 docs/detailed-log.md.

#### 구현

| 위치 | 내용 |
| --- | --- |
| 포크 model_executor/offloader/prefetch.py, ssd_tier.py | 가중치 3단 스트리밍(GPU 정적 버퍼 prefetch_step, pinned host 티어, SSD 티어 cuFile). VLLM_OFFLOAD_TIER_LAYOUT=interleave(교차 배치), IoWindow(SSD 읽기 진행 신호) |
| 포크 v1/core/sched/scheduler.py | VLLM_KV_LOAD_WAVE_GATE(적재 완료 요청 묶기), compute/load split의 앞 계산·뒤 적재 동시 진행, 비동기 적재 admit의 head-of-line 교착 수정 |
| 포크 csrc/kv_offload/cufile_fs.cpp, v1/kv_offload/cufile_fs/ | CuFileFsSpec: GPU KV 블록 ↔ 파일 cuFile C++ backend(읽기·쓰기 스레드, 쓰기·읽기 정지/재개), manager(admission all/seen_twice/profile, 용량 상한 LRU/LFU, write-behind cufile_fs_store_window) |
| 포크 v1/kv_offload/hybrid/ | HybridSpec: pinned host KV 층 + GDS SSD 층(write-through), LMCache 방식 비교용 |
| 포크 v1/kv_offload/split_policy.py | VLLM_KV_SPLIT off/fixed/serial/model |
| lib/obs | 관측: hostmon(nvidia-smi, vmstat, diskstats 1 s, bpftrace 블록 I/O), observe(nvidia-fs 카운터), events, run_nsys(gds trace, 캡처 구간, nsys.done), memguard |
| experiments/11-observability/run_obs.py | 러너: cold_fill → settle → reverse_retrieve, 또는 --mode stream(트레이스 시간 순서 open-loop). 입력 longbench/bailian/leval. 산출물 result.json, steps.jsonl, requests.jsonl, events.jsonl |
| experiments/11-observability/campaign_qwen72.sh, campaign_grid3b.sh | 조건 조합 캠페인(RATIOS, CONDS, LAYOUT, PSTEP, GATE, SPLIT, MNBT, KVB, MODE, NSYS) |
| experiments/12-channels/bench_channels.py | 전송 대역폭 단독·동시 실행 측정 |
| tools/ | compare_results.py(고정비 모형 대조), qwen72_policy_table.py, grid3b_table.py, nsys_overlap.py(forward 안 겹침 분석) |

#### 한계

- 카드 한 장, SSD 한 장(OS와 공유), BAR1 256 MiB. KV·가중치 버퍼 등록이 안 되어 cuFile은 bounce 두 홉이고 SSD 읽기가 GPU PCIe 링크를 같이 씀. 등록 직접 DMA는 BAR1 8 GiB 이상(사실상 VRAM 전체) 카드에서만.
- prefetch_step 2의 정적 버퍼 1.6 GiB 때문에 chunk 8192가 안 들어가 chunk 4096·GPU KV 18.8k로 맞춤. forward 수가 기준선보다 많음.
- 재방문이 역순인 2단계 워크로드는 저장소 적중을 최대로 만드는 조건. 트레이스 시간 순서 리플레이(--mode stream)로 재측정 중.
- 출력 토큰열 검사는 Qwen + chunked prefill에서 근소 차 토큰이 뒤바뀌는 경우가 있어 완전 일치를 기준으로 못 씀.

#### Directory

| 경로 | 내용 |
| --- | --- |
| `env.sh` | 공통 실행 환경 |
| `lib/obs/` | 관측 계층 |
| `experiments/11-observability/` | 러너, 캠페인, 입력 데이터(data/) |
| `experiments/12-channels/` | 전송 대역폭 측정 |
| `tools/` | 결과 대조·표 |
| `results/qwen72b/`, `results/grid3b/`, `results/channels/` | 결과(result.json, steps, requests, events, nsys stats). nsys 원본과 티어·KV 파일은 git 제외 |
| `docs/detailed-log.md` | 상세 기록 |
| git 태그 `archive/opt-era-2026-09-18` | 이전 세대(OPT, expfs, 01~10 실험) 전체 |
