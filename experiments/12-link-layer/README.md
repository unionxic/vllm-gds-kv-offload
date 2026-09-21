### 링크 계층 측정

PCIe(TLP·MPS/MRRS·IOMMU·P2P 경로)와 RoCE(MTU·QP 수·pause/ECN·NVMe-oF 큐)를 변수로 둔 마이크로벤치. 모델 없이 경로 대역폭만 본다.

#### 스크립트

- pcie_probe.sh: lspci·nvidia-smi·IOMMU·NIC 카운터를 있는 그대로 기록(설정 변경 없음)
- h2d_sweep.py: pinned host↔GPU 복사 대역폭을 전송 크기·스트림 수별로 측정(torch)
- mrrs_sweep.sh: GPU DevCtl MaxReadReq를 128~4096으로 바꿔 가며 h2d_sweep 실행, 끝나면 원복(setpci, sudo)
- roce_sweep.sh: perftest(ib_read_bw/ib_write_bw/ib_write_lat) 메시지 크기×QP 수×MTU 매트릭스, 양쪽 NIC 카운터 차분. rain이 서버, sunny가 클라이언트(host 메모리와 GPU 메모리)
- p2p_sweep.sh: sunny에서 gdsio 로컬 NVMe / 원격 NVMe-oF(램디스크) / 둘 / H2D 동시 조합을 IO 크기별로, nvidia-smi PCIe 처리량과 NIC 카운터 같이 기록
- mixed_link.sh: NVMe-oF 읽기(gdsio)와 RDMA 읽기(ib_read_bw)를 한 링크에서 동시에 흘려 몫과 pause 카운터를 보고, NVMe-oF io queue 수를 바꿔 반복

결과는 results/link-layer-<host>/ 아래 텍스트·jsonl.
