### 세 경로 KV 공급: host DRAM + 로컬 NVMe GDS + 원격 NVMe-oF RDMA

긴 문맥 서빙에서 KV cache가 DRAM을 넘쳐 SSD까지 써야 할 때, host DRAM(H2D)·로컬 NVMe(GDS)·원격 DRAM/SSD(NVMe-oF RDMA + GDS) 세 소스로 GPU를 동시에 공급하는 방식이 Mooncake(원격 DRAM 풀 + SSD 오프로드, GPU 직접 RDMA)보다 나은지 본다. 구현은 vLLM 포크(vllm/), 워크로드는 Bailian 트레이스 200건(16k 상한), 모델은 Llama-3.1-8B GPTQ. 기준선과 우리 조건은 같은 GPU(RTX A4000 16 GB)에서 돌린다.

#### 확인된 것

- 세 소스의 대역폭은 더해지지 않는다. GPU로 들어오는 P2P(로컬 NVMe, NVMe-oF, RDMA 합)는 약 10.3 GB/s에서 멈추고, H2D를 섞어야 11–11.9 GB/s(PCIe x16 상한 12.3). H2D 단독 대비 3경로의 대역폭 이득은 2–3%라 원래 가설(빈 링크 구간을 세 소스로 채운다)은 이 플랫폼에서 성립하지 않는다.
- 남는 이득은 용량 넘침 상황의 SSD 경로다. DRAM 8 GiB 예산에서 Mooncake의 SSD 복원은 DRAM 복원보다 요청당 약 2 s 느리고(SSD→DRAM→RDMA 경유), 우리 GDS 경로는 저하가 없다. 공통 적중 요청의 엔진 내부 지연은 2.3 vs 3.5 s.
- 그러나 전체 지표에서 앞서지 못한다. 적중 건수는 Mooncake 70 vs 57로 항상 뒤지고, 도착 기준 TTFT와 대기는 8 GiB 예산에서 같은 수준(런 간 편차 6–7 s 안), DRAM이 넉넉한 46 GB 예산에서는 Mooncake가 대기 14 vs 23 s로 앞선다.
- 적중 손실의 원인은 write-behind다. 저장이 커밋되기 전에 재사용 요청이 도착해 miss가 나고, 커밋 지연은 디스크가 아니라 블록을 만든 GPU 작업의 이벤트 대기다. 저장이 decode에 주는 간섭 자체는 작다.

| 8 GiB 예산 | queue 중앙 | TTFT(도착) 중앙 | 엔진지연 중앙 | 적중 |
| --- | --- | --- | --- | --- |
| mooncake | 30.0 / 36.4 s | 33.7 / 39.9 s | 2.79 / 2.88 s | 70 / 69 |
| ours | 29.3 / 36.0 s | 31.8 / 38.9 s | 1.72 / 2.01 s | 57 / 56 |

두 값은 같은 설정의 1차 / 반복 런. 출력 토큰열은 조건 사이에 동일.

#### 하고 있는 것

- 적중 손실 줄이기: 재사용 가능성 높은 앞쪽 블록부터 저장, 저장 job의 GPU 이벤트 분리. pending-wait(쓰기 중인 키를 기다리기)는 그 다음.
- 발행 설정을 실측에 맞추기: 원격은 4 MiB 요청·8스레드·NVMe-oF 큐 8개가 최선(7.5 → 9.3 GiB/s, 큐는 적용), 로컬은 1 MiB·4스레드면 장치 상한 3.3 GiB/s. 워커 읽기 스레드를 4에서 8로 올리고 배치 비율을 P2P 예산 안에서 다시 잡는다. 루트별 in-flight 상한과 읽기 우선 정책은 이득이 없었다.
- 판정 방법: 8 GiB 예산에서 반복 3회로 대기·처리량을 다시 비교. 원격 티어를 host 경유로 강제한 조건을 추가해 GDS 경로의 이득을 같은 예산에서 직접 확인.

#### 구성

- vllm/: 포크. KV 파일 티어(다중 루트·용량 spill·IO 발행 정책), host+파일 hybrid 티어, cuFile C++ 워커
- experiments/11-observability: 러너 run_obs.py, 캠페인 스크립트, 트레이스
- experiments/12-link-layer: 대역폭 상한과 발행 설정 측정(PCIe, RoCE, 경로 조합)
- lib/obs, tools/: 요청·키 계측(kvtrace), 결과 대조(check_stream, check_tier_reads, check_replay)
- results/: 캠페인별 result.json·requests.jsonl·campaign.log
- docs/: interim-summary.md(전체 표), detailed-log.md(상세 기록)
