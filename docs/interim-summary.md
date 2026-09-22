# GDS KV 3경로 공급 실험 중간 정리

### 연구 질문

- 긴 문맥 서빙에서 KV cache가 GPU는 물론 host DRAM과 원격 DRAM까지 넘쳐 SSD 계층이 실제로 쓰여야 하는 상황에서, 세 소스(host DRAM H2D, 로컬 NVMe GDS, 원격 NVMe-oF RDMA + GDS)로 GPU를 동시에 공급하는 방식이 기준선 Mooncake(원격 DRAM 풀 + SSD 오프로드, GPU 직접 RDMA)보다 적중 TTFT·대기·처리량에서 나은가
- 그 이득이 어느 계층(PCIe·RoCE·GPU 수신 경로·저장 정책)에서 나오고 어디서 막히는가

### 환경

#### 호스트

| 항목 | rain (원격 메모리·SSD 노드) | sunny (추론 노드) |
|---|---|---|
| CPU/메모리 | Skylake-SP, DRAM 읽기 31.9 GB/s | Skylake-SP, DRAM 125 GB, 읽기 59.2 GB/s |
| GPU | Quadro RTX 5000 (Turing, 16 GB, BAR1 256 MiB 고정) | RTX A4000 (Ampere, 16 GB, BAR1 16 GB로 확장) |
| GPU PCIe | Gen3 x16, MaxPayload 128 B (TU104 USB-C 기능이 128 B까지라 링크 전체가 128) | Gen3 x16, MaxPayload 256 B |
| NIC | ConnectX-6 100G, mlx5_1, 30.0.0.3 | ConnectX-6 100G, mlx5_0, 30.0.0.4 |
| 로컬 NVMe | Samsung PM981 500 GB (root) | Samsung PM981 500 GB, p3 313 GB → /mnt/local-ssd (xfs) |
| IOMMU | 없음 | iommu=pt (passthrough 전 H2D 8.1 → 12.35 GB/s) |
| 커널/OFED | 5.15, MOFED 23.10 | 6.8, DOCA-OFED 3.2.1 (MOFED 25.10; nvme_rdma에 nvidia-fs 훅 있는 마지막 계열) |
| GDS | nvidia-fs 2.30.2, CUDA 12.8 | 동일 |

- 링크: 100G RoCE 직결(1홉), MTU 9000 / RoCE 4096, PFC 꺼짐, 802.3x global pause 켜짐, ECN 전 우선순위 on
- NIC와 GPU가 서로 다른 루트 포트(0000:16 / 0000:64) → NIC→GPU P2P는 루트 컴플렉스를 지남. 로컬 NVMe(0000:b2)도 별도 루트 포트

#### 저장 계층(우리 조건, sunny 기준)

| 층 | 위치 | 경로 | 용량 |
|---|---|---|---|
| host pinned | sunny DRAM | cudaMemcpy H2D | 4–8 GB, LRU, write-through |
| 원격 DRAM | rain brd 램디스크 | NVMe-oF RDMA(nqn…rain:dram0) + GDS | 40 GB, 용량 상한 spill |
| 원격 SSD | rain NVMe 파일 namespace(150 GB sparse) | NVMe-oF RDMA(nqn…rain:ssd0) + GDS | 150 GB |
| 로컬 SSD | sunny NVMe p3 | GDS | 313 GB |

- 파일 티어 3루트는 블록 해시로 가중치(로컬 25 : 원격 DRAM 82 : 원격 SSD 32) 비례 배치. 용량이 있는 루트가 차면 남은 루트끼리 다시 비례 배치(spill)
- Mooncake 기준선: rain mooncake_master + mooncake_client(DRAM 세그먼트, RDMA), SSD 오프로드 켬(file_per_key 백엔드, 하트비트 1 s, 150 GB). sunny vLLM은 MooncakeStoreConnector, GPU 텐서 직접 등록(BAR1 16 GB)

#### 소프트웨어

- vLLM 포크 v0.26.1.dev0 (weight-ssd-offload): CuFileFsSpec(C++ cuFile 워커, 읽기 4·쓰기 4 스레드), HybridSpec(host 티어 + 파일 티어), MultiRootFileMapper(가중치·용량 spill), shadow store, IO 발행 정책(root_inflight, read_priority)
- mooncake-transfer-engine 0.3.13.post1
- 모델 Llama-3.1-8B-Instruct GPTQ INT4(토큰당 KV 128 KB), 스모크는 Qwen2.5-3B
- 워크로드: Alibaba Bailian 트레이스 offset 1240, 200건, hash 블록 16토큰, KV 블록 64토큰(8 MiB), 프롬프트 상한 16k, 도착 시각 TSCALE 6(압박), 동시 4
- 러너 experiments/11-observability/run_obs.py (모드 phases / forced_hit / stream / stream_replay), 계측 lib/obs/kvtrace.py, 분석 tools/check_stream.py · check_tier_reads.py · check_replay.py

#### 지표 정의

- TTFT(도착): 도착 → 첫 토큰(사용자가 겪는 시간, 외부 대기 포함), 요청별 시각으로 계산
- 엔진지연: 엔진 제출 → 첫 토큰(외부 대기 제외)
- queue: 도착 → 엔진 제출(동시 4 제한 대기)
- 적중 토큰: 요청별 커넥터 lookup 최댓값(matched_of), 적중 요청: matched_of > 0
- 층별 적재 바이트: 요청별로 실제 읽은 층(host / 각 루트 / mooncake memory·disk), kvtrace 필요
- 두 중앙값을 더하지 않음. 계측을 켠 런은 계측 런끼리만 비교

### 경로 대역폭

#### 경로 조합 매트릭스(12 s, h2d_loop 64 MiB 2 스트림 + gdsio 8스레드 1 MiB, GB/s)

| 조합 | rain | sunny(직접 경로) |
|---|---|---|
| H2D | 12.31 | 12.34 |
| GDS 로컬 | 3.51 | 2.47 |
| GDS 원격(램디스크) | 9.95 | 8.2 |
| H2D + 로컬 | 10.99 + 1.17 = 12.16 | 9.54 + 2.54 = 12.08 |
| H2D + 원격 | 8.02 + 3.86 = 11.88 | 8.21 + 3.79 = 12.0 |
| 로컬 + 원격 | 3.53 + 6.74 = 10.27 | 2.53 + 6.44 = 8.97 |
| 셋 | 7.94 + 2.14 + 1.98 = 12.06 | 7.90 + 2.09 + 1.94 = 11.93 |

- H2D가 섞이면 합은 PCIe 한 방향 상한 12.3 GB/s, P2P만(로컬+원격)은 9–10에서 멈춤. 복사 엔진(H2D)이 우선권을 가져 GDS 몫이 눌림

#### CPU·메모리

| 항목 | rain | sunny |
|---|---|---|
| DRAM 읽기 | 31.9 GB/s | 59.2 GB/s |
| memcpy 단일 스레드 | 8.3 | 9.3 |
| pageable H2D | 7.8–7.9 | 7.8–7.9 |
| pinned H2D / D2H / 양방향 | 12.25 / 13.15 / – | 12.35 / 13.17 / 22.66 |

- 12.3은 PCIe Gen3 x16 한 방향 실효 상한(양방향 합 22.5). CPU 병목은 CPU 경유 경로에만 해당

### 링크 계층

#### PCIe

| 실험 | 결과 |
|---|---|
| GPU MaxReadReq 128–4096 스윕(rain·sunny) | H2D 12.25/12.37, D2H 13.15/13.17 GB/s 전 구간 동일 |
| MaxPayload 128(rain) vs 256(sunny) | 결과 차이 없음 |
| IOMMU | sunny passthrough 전 8.1 → 후 12.35 GB/s |

- TLP 페이로드·읽기 요청 크기는 12.3의 원인이 아님. D2H가 H2D보다 7% 높은 것은 읽기 요청 왕복 대 게시 쓰기 차이
- 남는 후보는 GPU 복사 엔진과 루트 컴플렉스. "루트 컴플렉스가 원인"은 확정 아님, "GPU 수신 경로의 병목"까지가 안전

#### RoCE perftest(rain 서버, sunny 클라이언트, Gb/s)

| 조건 | host 메모리 | GPU 메모리 |
|---|---|---|
| ib_read_bw MTU 4096, QP 4–16 | 98.0 | 82.1 |
| ib_read_bw MTU 1024 | 92.5 | 82.1 |
| ib_read_bw QP 1 | 66–96 | 66–82 |
| ib_write_bw | 98.0 | 74–78 (4 MiB 다중 QP 50) |
| ib_write_lat 64 B / 1 MiB | 0.84 / 90.5 us | – |

- GPU 목적지는 MTU·QP·메시지 크기와 무관하게 82.1 Gb/s = 10.26 GB/s에서 멈춤. host 목적지는 98 Gb/s = 12.25 GB/s
- GPU로 받을 때만 sunny NIC가 pause 프레임 송신(4 s에 3–4만 개). host 메모리 수신은 QP ≥ 4에서 거의 0

#### 혼합 링크(같은 7 s 창, sunny NIC rx_bytes_phy 차분)

| 조합(목적 버퍼) | NVMe-oF→GPU | RDMA | 링크 수신 합 | pause |
|---|---|---|---|---|
| NVMe-oF 단독 | 8.94 GiB/s | – | 9.74 GB/s | 52k |
| RDMA→GPU 단독 | – | 82.1 Gb/s | 10.42 | 67k |
| NVMe-oF + RDMA→GPU | 3.89 GiB/s | 47.5 Gb/s | 10.41 | 66k |
| RDMA→host 단독 | – | 98.0 Gb/s | 12.43 | 0 |
| NVMe-oF→GPU + RDMA→host | 4.29 GiB/s | 61.6 Gb/s | 12.33 | 0 |

- 두 전송이 모두 GPU를 향하면 합이 10.4 GB/s에서 멈춤(이전의 11.5는 측정 창 어긋남). RDMA를 host로 보내면 링크는 12.4까지 나가고 pause 0 → 상한 10.4와 pause는 링크가 아니라 GPU 수신 경로의 것
- 요청자 주도인 RDMA read가 NVMe-oF를 단독의 절반 아래로 밀어냄
- NVMe-oF io 큐 수: 36개 7.5, 8개 9.3, 2개 6.3 GiB/s. 큐 2개는 pause 거의 없음
- 물리 계층: rx_err_lane(FEC 전 원시 오류)·rx_corrected_bits_phy(FEC 교정)만 증가, 미교정 카운터 0 → 링크 정상

### 밑단 5단계

| 단계 | 확인할 것 | 결과 | 결론 |
|---|---|---|---|
| 1 | 로컬 SSD 경로 불안정 | 순차 기록 파일: dd·GDS·CPU 경유 모두 3.31 GiB/s 3회 동일. 조각 파일(936–1,018 extent): GDS 2.39, CPU 경유 1.00–2.39 | 파일 배치 문제. 장치 상한 3.3 GiB/s(원시 파티션 3.19) |
| 2 | GPU 중간 복사 | 등록/미등록 처리량 동일, nvidia-fs readMiB 전량 = 항상 직접 경로. 비동기 -x 5는 원격 4.4로 손해 | 줄일 복사 없음 |
| 3 | 요청 크기·동시성 | 원격 io 64K 3.65 → 1M 7.47 → 4M 9.49 → 16M 8.38 GiB/s. threads 1/2/4/8 = 1.29/2.22/5.15/7.26. NVMe-oF 큐 2/4/8/16/36 = 7.82/8.50/9.14/7.94/7.53. 로컬은 io ≥ 1M·4스레드면 상한 | 원격은 4 MiB·8스레드·큐 8이 최선 |
| 4 | RDMA 발행 | perftest tx-depth ≥ 4면 크기·QP 무관 82.1 Gb/s. transfer_engine_bench 스레드 ≥ 4면 block 64K–4M·batch 1–128 전부 10.16–10.21 GB/s | GPU 직접 수신 10.2 GB/s는 발행 방식과 무관한 플랫폼 상한 |
| 5 | 세 경로 조합 | 아래 표 | P2P 합은 조합과 무관하게 9.5–9.7 GiB/s, H2D는 별도 예산 |

#### 5단계 조합(같은 11 s 창, NIC 수신 바이트·nvidia-fs readMiB 차분, GiB/s)

| 조합 | 로컬 GDS | 원격 GDS | RDMA→GPU | H2D | GPU 유입 합 | pause 지속 |
|---|---|---|---|---|---|---|
| 로컬 | 3.32 | | | | 3.3 | 0 |
| 원격 | | 9.52 | | | 9.5 | 1.7M |
| RDMA | | | 9.70 | | 9.7 | 1.8M |
| H2D | | | | 10.7 | 10.7 | 0 |
| 로컬+원격 | 3.31 | 6.2 | | | 9.56 | 4.9M |
| 원격+RDMA | | 2.1 | 7.4 | | 9.7 | 1.7M |
| H2D+로컬 | 3.32 | | | 7.5 | 10.8 | 0 |
| H2D+원격 | | 3.6 | | 7.3 | 10.9 | 7.3M |
| H2D+로컬+원격 | 1.1 | 2.5 | | 7.4 | 11.05 | 8.5M |
| 넷 다 | 2.3 | 1.9 | ≤1.7 | 6.8 | 약 10.3–11 | 9.4M |

- GPU P2P 수신 총량은 소스 조합과 무관하게 9.5–9.7 GiB/s(10.2–10.3 GB/s)이고 튜닝으로 움직이지 않음
- H2D는 P2P와 다른 예산: 합이 11–11.05 GiB/s(11.6–11.9 GB/s)까지 가되 H2D가 7.3–8 GB/s를 선점하고 P2P가 나머지를 나눔
- 세 소스의 대역폭 단순 합산(3.3 + 9.5 + 10.7)은 성립하지 않음. H2D 단독 대비 3경로 이득은 대역폭 기준 2–3%
- 동시 P2P 소스가 늘수록 pause 지속이 1.7M → 9.4M

#### GDR과 NVMe-oF RDMA의 관계

- GDR은 NIC가 host DRAM을 거치지 않고 GPU 메모리에 DMA하는 하드웨어 경로. NVMe-oF RDMA + nvidia-fs 훅(우리 원격 티어)과 Mooncake transfer engine(verbs 직접) 둘 다 이 경로 위에 있으며 같은 10.2 GB/s 상한에 닿음
- 차이는 소프트웨어 층: 주도권(타깃 push vs 요청자 read), 요청 단위·큐, 등록 방식(cuFile 내부 vs ibv_reg_mr 통째, 후자는 BAR1이 커야 함)
- GDR의 의미는 "더 빠르게"가 아니라 "DRAM이 넘칠 때 H2D 예산을 축내지 않고 SSD·원격을 붙이는" 것

### Mooncake 기준선 구성에서 확인한 사실

| 항목 | 내용 |
|---|---|
| GPU 직접 경로 | rain(BAR1 256 MiB)은 KV 텐서 등록 실패 → host staging 우회 필요. sunny(BAR1 16 GB)는 직접 등록 성공(transfer_engine_bench VRAM 10.20 GB/s, host 12.23) |
| SSD 오프로드 | 0.3.13에 있음(master --enable_offload, client --enable_offload + MOONCAKE_OFFLOAD_FILE_STORAGE_PATH). GDS 없음: SSD 사본 읽기는 소유자 SSD→DRAM 버퍼→RDMA→요청자 DRAM→GPU |
| 기본 bucket 백엔드 결함 | 256 MB 버킷이 찰 때까지 객체를 미루는 사이 DRAM 축출이 먼저 와 유실(2 GB 세그먼트·8 MiB 객체 640개 중 372개 miss, 오프로드 20–55 MB/s) |
| 채택 설정 | file_per_key 백엔드 + 하트비트 1 s → 640/640 회수, GPU batch_get_into도 SSD 사본에서 성공 |
| 적재 쪼개기 결함 | SSD 층 적재 묶음(171–208키, ≥1.4 GB)이 소유자 staging 버퍼 1.25 GB 초과로 BUFFER_OVERFLOW(-10) 통째 실패 → 긴 요청이 첫 토큰 못 냄. 커넥터 설정 enable_offload=true로 예산(1.25 GB × 0.9) 단위 분할 |
| NoF(NVMe-oF SSD 풀) | SPDK + USE_NOF 빌드 전용이라 pip 휠에서 불가 |

### KV 실험 결과

#### 워크로드의 재사용 거리(Bailian 1240, 200건, KV 블록 8 MiB)

| DRAM 총량(LRU 가정) | SSD에서 읽는 재사용 | 비율 |
|---|---|---|
| 46 GiB | 0.4 GiB | 1% |
| 16 GiB | 6.4 GiB | 8% |
| 8 GiB | 15.1 GiB | 19% |
| 4 GiB | 24.9 GiB | 31% |

- 고유 KV 73 GiB, 재사용 블록 10,275개(80 GiB), 재사용 거리 p50 2.8·p75 5.5·p90 15.3 GiB
- 우리 원격 DRAM 루트는 LRU가 아니라 먼저 온 블록이 차지하는 spill이라 우리 SSD 몫이 항상 Mooncake 이상(우리에게 보수적)

#### DRAM이 넘치기만 하는 조건(mooncake 46 GB vs ours host 8 + 원격 38 GiB 상한, 계측 없음)

| 조건 | queue 중앙/p95 | TTFT(도착) 중앙/p95 | 엔진지연 중앙/p95 | 적중 | 저장 |
|---|---|---|---|---|---|
| mooncake | 14.3 / 24.4 s | 17.7 / 29.0 s | 2.42 / 8.20 s | 73 | DRAM 39.8/46 GB, SSD 사본 73 GB, 축출 4,260키 |
| ours-ratio | 23.1 / 33.7 | 26.4 / 37.2 | 1.71 / 7.25 | 58 | 원격 DRAM 38 GiB 꽉 참(spill 677), 로컬 15, 원격 SSD 20 GiB |

- 요청별 층 계측 런에서 mooncake 적중 복원 45 GiB 중 SSD에서 읽은 것은 1.6 GiB(재사용이 최근 블록에 몰려 DRAM에서 처리). DRAM이 넘쳤다는 사실만으로 SSD 조건이 성립하지 않음
- mooncake만 적중한 17건은 한 블록(64토큰)짜리. 진짜 차이는 적중 요청 decode(1.21 vs 1.92 s)와 queue 누적
- 토큰열 199/200(근소 차 1건)

#### 저장 간섭(재사용 프리픽스 1,642블록 12.8 GiB를 미리 저장한 뒤 같은 stream 3회)

| 조건 | 적중 | queue 중앙/p95 | TTFT(도착) 중앙/p95 | 엔진지연 중앙/p95 | decode 중앙/p95 | cuFile 쓰기 |
|---|---|---|---|---|---|---|
| store-off | 80 | 2.51 / 11.4 s | 4.43 / 14.6 s | 1.12 / 6.02 s | 0.78 / 5.06 s | 0 |
| store-shadow | 80 | 3.60 / 12.4 | 5.94 / 15.3 | 1.20 / 5.87 | 0.84 / 5.27 | 86 GiB |
| store-on | 79 | 4.26 / 12.7 | 6.65 / 15.9 | 1.25 / 5.98 | 0.90 / 5.37 | 73 GiB |

- 세 조건 토큰열 200건 동일, 적중 토큰 동일. host 티어 적재 바이트는 다름(저장 끄면 prefill 블록이 host에 남아 26.8 GiB, 켜면 11.2 GiB)
- paired 차: decode 중앙 +0.02–0.03 s(p95 +1.6), 엔진지연 +0.06–0.08, queue +0.6–0.9. decode 지연과 겹친 쓰기 job 수 상관 0.19–0.24
- 저장 간섭 자체는 꼬리에만 작게 남음. 실제 런의 손해(적중 58 vs 80, queue 23 s)는 write-behind로 커밋 전에 재사용이 도착한 miss 22건의 재계산
- 쓰기 스레드 시간 723 s 중 io 280 s, 나머지는 cudaEventSynchronize(블록을 만든 GPU 작업 대기) → 커밋 지연은 디스크가 아니라 GPU 큐

#### DRAM 예산 축소, 요청별 층 계측 런(적재 쪼개기 수정 뒤, 계측 켬)

| 예산 | 조건 | 적중 | 복원 합 | SSD에서 읽음 | 절반 이상 SSD인 요청 | queue 중앙 |
|---|---|---|---|---|---|---|
| 16 GiB | mooncake 16 GB | 70 | 44.7 GiB | 17.4 GiB | 34 | 28.0 s |
| 16 GiB | ours host 8 + 원격 8 | 58 | 43.7 | 18.7 | 25 | 22.7 |
| 8 GiB | mooncake 8 GB | 71 | 44.0 | 32.8 | 54 (적중 토큰 중앙 2,304) | 27.3 |
| 8 GiB | ours host 4 + 원격 4 | 57 | 45.5 | 28.7 | 35 | 26.4 |

- 8 GiB에서 Mooncake도 복원의 75%를 SSD에서 읽고 11.7k 토큰 프리픽스 3건도 SSD가 섞임 → SSD 조건 충족, 비교 예산은 8 GiB
- Mooncake 안에서 SSD를 읽은 적중은 DRAM만 읽은 적중보다 적중량을 맞춰도 엔진지연 +2.13 s(32/40 느림)

#### 공통 적중 요청 짝 비교(8 GiB, 54건)

| 지표 | mooncake | ours-ratio |
|---|---|---|
| 적중 토큰 중앙 | 5,056 | 7,616 |
| SSD 바이트 합 / SSD>0 요청 | 32.7 GiB / 47 | 28.1 GiB / 54 |
| TTFT(도착) 중앙 | 32.0 s | 32.4 s |
| 엔진지연 중앙 | 3.53 s | 2.32 s (ours 빠른 39/54) |
| 둘 다 SSD 읽은 47건 엔진지연 | 3.97 | 2.45 |

- 16 GiB에서는 TTFT(도착)도 33.8 vs 28.1 s로 ours가 51/54건 빠름

#### 8 GiB 예산 성능 비교(계측 없음)와 IO 발행 정책

| 런 | 조건 | queue 중앙 | TTFT(도착) 중앙 | 엔진지연 중앙 | 적중 | wall |
|---|---|---|---|---|---|---|
| 1차 | mooncake | 30.0 s | 33.7 s | 2.79 s | 70 | 264.6 s |
| 1차 | ours-ratio | 29.3 | 31.8 | 1.72 | 57 | 263.4 |
| 1차 | io-inflight 2/8/4 | 33.6 | 36.4 | 2.05 | 57 | 267.7 |
| 1차 | io-inflight-rp K=1 | 35.8 | 38.9 | 1.71 | 56 | 271.8 |
| 2차 | io-inflight 4/16/8 | 25.0 | 28.1 | 1.85 | 57 | 258.7 |
| 2차 | io-inflight-rp K=2 | 34.5 | 37.0 | 1.85 | 56 | 269.5 |
| 반복 | mooncake | 36.4 | 39.9 | 2.88 | 69 | 272.9 |
| 반복 | ours-ratio | 36.0 | 38.9 | 2.01 | 56 | 271.0 |

- 네 조건 출력 토큰열 200건 전부 일치
- 런 간 편차 6–7 s(같은 설정의 반복이 둘 다 느려짐). TSCALE 6 도착 조건에서 queue가 누적되는 구조라 단일 런으로 5 s 이하 차이는 판정 불가
- 런 사이에 안정한 지표: 적중 건수(mooncake 69–70 vs ours 56–57), 엔진지연(ours 1.7–2.0 vs mooncake 2.8–2.9 s)
- 루트별 상한은 거의 안 걸림(읽기·쓰기 스레드 4/4라 in-flight 최댓값 4). K=2는 쓰기 대기 687회 18.8 s에 이득 없음 → 발행 깊이는 루트 상한이 아니라 스레드 수와 요청 크기로 정해야 함

#### 강제 적중(같은 KV를 미리 저장하고 저장 차단 뒤 재생, sunny)

| 지표 | mooncake(직접 경로) | ours-ratio |
|---|---|---|
| replay 적중 | 32/32 | 32/32 |
| TTFT 중앙 | 9.23 s | 4.16 s (paired 32/32 ours 빠름) |
| replay wall | 13.8 s | 6.98 s |
| decode 중앙 | 0.52 s | 0.18 s |
| 적재 완료(전송 job) 중앙 | 0.11 s | 0.19 s |

- 전송 job 자체는 Mooncake 직접 RDMA가 빠르고(ours는 로컬 NVMe 몫이 느림), 요청 전체는 ours가 빠름

### 현재 결론

- 대역폭 상한: GPU로 들어오는 P2P(로컬 NVMe·NVMe-oF·RDMA) 총량은 발행 튜닝과 무관하게 약 10.3 GB/s에서 멈추는 플랫폼 제약이고, H2D가 7.3–8 GB/s를 따로 얹어 PCIe 링크 12.3 GB/s 근처까지 채우는 것이 3경로 동시 공급의 실제 상한. H2D 단독 대비 대역폭 이득은 2–3%
- 3경로의 의미: 대역폭이 아니라 용량. DRAM이 넘쳐 SSD·원격에서 읽어야 할 때 H2D 예산과 host 메모리를 축내지 않고 붙이는 것. 실측으로는 Mooncake의 SSD 복원이 DRAM 복원보다 요청당 +2.1 s인 반면 우리 경로는 저하가 없고, 같은 요청에서 엔진지연 2.32 vs 3.53 s
- 여전한 손해: 적중 건수(56–57 vs 69–71). write-behind로 커밋 전에 재사용이 도착한 miss이며, 커밋 지연의 정체는 디스크가 아니라 GPU 이벤트 대기
- 대기·처리량: 46 GB 예산에서는 Mooncake 우세(queue 14 vs 23 s), 8 GiB 예산에서는 동률(30 vs 29, 반복 36 vs 36) 이내이나 편차 6–7 s
- 경로별 최선 발행: 로컬 io ≥ 1 MiB·4스레드, 원격 io 4 MiB·8스레드·NVMe-oF 큐 8(적용함, 9.18 GiB/s), RDMA depth ≥ 4. 버퍼 등록·비동기·배치 모드·MaxReadReq는 효과 없음. 워커 cuFile 호출은 호출당 약 3 MiB로 이미 정점 근처, 남은 차이는 스레드 수(읽기 4 vs gdsio 8)

### 남은 과제

- 8 GiB 예산에서 반복 3회로 mooncake / ours-ratio / 발행 정책 재비교(편차 대비 판정 확보)
- 워커 읽기 스레드 8, 배치 비율을 실측 예산(P2P 9.6 GiB/s 안에서 로컬 ≤ 3.3, H2D 7–8 GB/s)에 맞춰 재설정
- write-behind miss 축소: 재사용 가능성 높은 앞쪽 블록 우선 저장, 저장 job의 GPU 이벤트 분리(pending-wait보다 우선)
- 원격 티어 compat 모드(host 바운스) 조건 추가로 GDR/GDS 이득을 같은 예산에서 직접 확인
- 3B 스모크·46 GB 런의 근소 차 토큰열 1건은 재현·이분 실험으로 데이터 손상이 아닌 수치 분기로 확인됨(기록만)

### 결과 폴더와 스크립트

| 내용 | 위치(실험 저장소 vllm-gds-kv) |
|---|---|
| 러너·캠페인 | experiments/11-observability/run_obs.py, campaign_phase3.sh, campaign_forced_hit.sh, smoke_*.sh |
| 계측·분석 | lib/obs/kvtrace.py, tools/check_stream.py, check_tier_reads.py(--pair), check_replay.py, check_forced_hit.py |
| 링크·밑단 | experiments/12-link-layer/{pcie_probe, mrrs_sweep, roce_sweep, p2p_sweep, mixed_link2, step1–5}.sh, h2d_sweep.py |
| 46 GB 예산 | results/stream-long-8b-sunny, stream-long-8b-sunny-trace |
| 저장 간섭 | results/store-interference-8b-sunny |
| DRAM 16/8 계측 | results/stream-long-8b-sunny-dram16-trace, -dram8-trace (mooncake-failed-nosplit는 수정 전) |
| 8 GiB 성능·정책 | results/stream-long-8b-sunny-dram8, -dram8-io16, -dram8-rep |
| 강제 적중 | results/forced-hit-8b-sunny, forced-hit-8b |
| 링크·밑단 결과 | results/link-layer-rain, link-layer-sunny, lowlevel-sunny |
| Mooncake 스모크 | scratchpad mc/mc_offload_smoke.py, mc_offload_rate.py; rain ~/bin/mooncake_up.sh(MC_OFFLOAD=1, MC_SEG) |
| 포크 | vllm/ (subtree): vllm/v1/kv_offload/cufile_fs/{multi_root.py, spec.py}, hybrid/, csrc/kv_offload/cufile_fs.cpp |

- 운영상 함정: 실험 파일은 dd 순차 기록본으로, pkill -f 패턴은 앞을 고정(^), 원격 프로세스는 fd를 끊은 bash -c로, transfer_engine_bench 타깃은 --use_vram=false, 캠페인 로그는 append-only라 마지막 줄로 판정, nsys 분석은 nsys.done 뒤에만
