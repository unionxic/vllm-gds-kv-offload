### vLLM 가중치 스트리밍 + GDS KV 오프로드

16 GB GPU 한 장에서 145 GB 모델(Qwen2.5-72B-Instruct)을 돌린다. 가중치는 host memory와 NVMe에서 매 forward 스트리밍하고, KV는 NVMe에 저장했다가 적중 시 읽는다. 구현은 vLLM 포크(~/vllm, weight-ssd-offload) 안에 있고 외부 코드는 없다.

| 항목 | 값 |
| --- | --- |
| GPU, RAM, SSD | Quadro RTX 5000 16 GB(BAR1 256 MiB), 125 GiB, 970 EVO(OS와 공유) |
| 전송 대역폭 | host→GPU 12.3 GB/s, SSD→GPU 3.4 GB/s, 동시엔 host 8.5 GB/s(bounce가 PCIe 링크 공유) |
| 모델 | Qwen2.5-72B fp16, 80 layer × 1.63 GiB, KV 토큰당 328 KB |

#### 핵심 결론

- forward 하나는 가중치 131 GiB를 GPU로 옮기는 시간이다. 기본 배치는 host 복사 5.7 s 뒤 SSD 읽기 23.3 s가 차례로 일어나 decode forward 28.4 s.
- host layer를 SSD layer 사이에 고르게 끼워 넣고(교차 배치) 다음 두 layer를 동시에 가져오면(prefetch_step 2) 두 경로가 같이 흘러 forward가 22.2 s.
- host 비율을 대역폭 비율에 맞추면(RAM 0.72, host 55 : SSD 25 layer) 두 경로가 같이 끝나 forward 14.0 s. 이 링크의 하한은 11.7 s.
- KV는 GPU와 NVMe 사이를 직접 오간다. 적중은 prefill 계산을 건너뛰고, 요청당 KV 2.6 GB 쓰기는 SSD 쓰기 캐시 안이라 저장 비용이 0. wave gate가 적재 완료 요청을 한 forward에 묶는다.
- 위를 합친 정책 조합이 Bailian 32건에서 4,350 → 2,448 s(−44%), LongBench 32건에서 12,652 → 6,527 s(−48%). 출력 토큰열은 재계산과 동일.
- 이득이 없던 것: host memory를 KV에 주기, SSD 용량 축출(LRU/LFU), seen_twice admission, LMCache 방식 host 층, compute/load split.

#### 측정 (Qwen2.5-72B, Bailian 32건, chunk 4096·GPU KV 18.8k)

| 조건 | 두 단계 합계 | decode forward |
| --- | --- | --- |
| 재계산, 기본 배치 (chunk 8192, KV 21.4k 기준선) | 4,350 s | 28.4 s |
| SSD 전부 저장, 기본 배치 | 4,541 s | 28.3 s |
| LMCache 방식 카피(host 8 GB LRU + SSD) | 4,537 s | 28.3 s |
| 재계산, 교차 배치 + prefetch_step 2, RAM 0.5 | 3,880 s | 22.2 s |
| 정책 조합, RAM 0.5 | 3,078 s | 22.1 s |
| 정책 조합, RAM 0.72 | 2,448 s | 14.0 s |

LongBench-v2 32건(실제 문서, RAM 0.72): 재계산 12,652 s → 정책 조합 6,527 s.

#### 구성

| 경로 | 내용 |
| --- | --- |
| 포크 offloader/prefetch.py, ssd_tier.py | 가중치 3단 스트리밍, 교차 배치, SSD 읽기 진행 신호 |
| 포크 v1/core/sched/scheduler.py | wave gate, compute/load split, 비동기 적재 admit 교착 수정 |
| 포크 csrc/kv_offload/cufile_fs.cpp, v1/kv_offload/cufile_fs/, hybrid/, split_policy.py | cuFile C++ backend, KV manager(admission, LRU/LFU, write-behind), host+SSD 두 층, split 제어기 |
| lib/obs | 관측(hostmon, nvidia-fs, events, nsys 캡처 구간, memguard) |
| experiments/11-observability | 러너 run_obs.py(2단계 또는 트레이스 시간 순서 stream), 캠페인, 입력 data/ |
| experiments/12-channels, tools/ | 전송 대역폭 측정, 결과 대조·표·nsys 겹침 분석 |
| results/qwen72b, grid3b, channels | 결과. nsys 원본·티어·KV 파일은 git 제외 |
| docs/detailed-log.md | 상세 기록. 이전 세대(OPT, 01~10)는 git 태그 archive/opt-era-2026-09-18 |

#### 한계

- BAR1 256 MiB라 cuFile은 bounce 두 홉이고 SSD 읽기가 GPU PCIe 링크를 같이 쓴다. 직접 DMA는 BAR1이 VRAM 전체인 카드에서만.
- prefetch_step 2의 버퍼 1.6 GiB 때문에 chunk 8192가 안 들어가 chunk 4096·KV 18.8k로 맞췄다.
- 역순 재방문 워크로드는 적중을 최대로 만드는 조건. 트레이스 시간 순서 open-loop 리플레이 128건(동시 상한 4)에서는 vLLM 기본값이 도착을 못 따라가 3시간에 93건 완료·대기 중앙값 903 s였고, 정책 조합은 128건 완료·대기 12 s·TTFT 169 → 70 s. SSD KV cache hit은 토큰의 19%(트레이스 최대 38%, 나머지는 GPU prefix cache hit).
