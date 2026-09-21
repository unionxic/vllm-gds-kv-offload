### HybridSpec

KV 블록을 host(pinned memory) 티어와 SSD(GPUDirect Storage) 티어 두 단으로 오프로드하는 OffloadingSpec.
티어 1은 vLLM in-tree CPU 티어(`cpu/manager.py`, `cpu/gpu_worker.py`)를 그대로 쓰고, 티어 2는 포크의
cuFile 백엔드(`cufile_fs/spec.py`, `csrc/kv_offload/cufile_fs.cpp`)를 그대로 쓴다. 이 패키지는 두 티어를
묶는 스케줄러 쪽 관리자와 워커 쪽 job 분할만 담당.

#### 구성

- `spec.py` — HybridSpec. 내부적으로 CPUOffloadingSpec과 CuFileFsSpec을 인스턴스화해 재사용
- `manager.py` — HybridManager. 스케줄러 쪽 조회·배치·축출 판정
- `worker.py` — HybridWorker. 바깥 job 하나를 티어별 하위 job으로 쪼개 제출하고 합쳐 보고
- `common.py` — HybridLoadStoreSpec. 티어별 하위 spec과 키 위치(cpu_pos / file_pos)
- `test_hybrid_cpu_only.py` — GPU 없이 도는 단위 테스트

#### 설정 키

kv_connector_extra_config에 넣는 키.

- `spec_name` — "HybridSpec"
- `hybrid_host_gb` — host 티어 크기(GB, 10^9 바이트). 블록 수는 CPUOffloadingSpec과 같은 산식
- `hybrid_host_policy` — lru(기본) | arc
- `hybrid_placement` — host_first(기본) | profile
- `hybrid_profile` — placement=profile일 때 읽을 JSON 경로. `{"hashes": {"<block hash hex>": reuse}, "min_reuse": N}`
- `hybrid_write_through` — true(기본)면 store 때 host와 SSD 양쪽에 모두 씀
- `cufile_fs_*` — SSD 티어에 그대로 전달. `cufile_fs_root_dir` 필수이고 read/write threads,
  register_tensors, store_window, capacity_gb, policy, admission은 cufile_fs/spec.py와 동일

실행 예: `{"spec_name": "HybridSpec", "hybrid_host_gb": 8, "cufile_fs_root_dir": "/mnt/nvme/kv"}`

#### 데이터 흐름

store는 GPU에서 두 티어로 동시에 나간다. write-through(기본)에서는 한 블록이 host 쓰기와 SSD 쓰기
두 하위 job으로 갈라지고, 같은 GPU 블록을 CPUOffloadingWorker와 CuFileFsWorker가 각각 읽는다.
placement=profile이면 재사용 횟수가 min_reuse 이상인 키만 host 후보가 되고 나머지는 SSD만 간다.
host 쪽 할당이 실패하면(축출 가능한 블록 부족) 그 배치는 SSD만으로 진행.

load는 티어별로 한 번씩만 간다. 조회 우선순위는 host > SSD > miss이고, host에 있으면 host→GPU,
없고 SSD에 파일이 있으면 SSD→GPU로 GDS가 직접 올린다. SSD 적중분을 host로 올리는 승격 단계는 없다.
한 요청의 키 목록이 두 티어에 섞여 있으면 HybridWorker가 청크 단위로 GPU 블록 목록을 잘라
두 하위 job으로 보내고, 둘 다 끝났을 때만 바깥 job을 완료로 보고한다. 하나라도 실패하면 실패.

축출 보고는 실제 손실만 센다. write-through에서 host가 축출한 키는 SSD 사본이 있으므로
evicted_keys에 넣지 않고, SSD가 축출한 키도 host에 남아 있으면 넣지 않는다.

#### 구현하지 않은 것

- write-back demotion. host 축출분을 SSD로 내리는 경로는 없다. CPUOffloadingManager.prepare_store가
  축출된 블록을 같은 호출 안에서 free하고 곧바로 재할당하기 때문에, evicted_keys를 받은 시점에는
  그 슬롯이 이미 다른 키 몫이고 GPU→host 쓰기가 예약돼 있을 수 있다. 스케줄러 쪽에서 /dev/shm
  공유 영역을 읽어 파일로 내리려 해도 읽는 사이에 덮일 수 있어 정확성을 보장하지 못하고, 고치려면
  cpu/manager.py 수정이 필요. `hybrid_write_through=false`는 demotion이 아니라 배치 정책이 고른
  한 티어에만 쓰는 모드이고, 이때 host 축출은 진짜 손실로 보고
- SSD→host 승격(promotion). 적중한 SSD 블록은 GPU로 직접 올라가고 host에 남지 않음
- 다중 KV cache group. cufile 백엔드와 같은 제약으로 단일 그룹만
- 워커 쪽 GPU 경로의 자동 테스트. 테스트는 스케줄러 쪽 로직과 GPU 블록 분할 계산까지만 확인
