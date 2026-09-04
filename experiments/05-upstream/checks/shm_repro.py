# 최신 main 회귀 검증 2: /dev/shm 누출 — #52596(barrier unlink) 적용 상태에서
#   TieringOffloadingSpec + TP=1 + SIGKILL 재시험.
# 정적 관찰: main의 tiering/spec.py는 barrier 인자를 넘기지 않는다(두 생성부 모두).
# 예상: CPUOffloadingSpec은 unlink되어 누출 없음, Tiering은 여전히 누출.
# usage: python shm_repro.py {tiering|cpu}   (child: 엔진 띄우고 대기)
#        판정은 러너(run_shm_check.sh)가 SIGKILL 후 /dev/shm 잔재로 수행.
import glob
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
MODE = sys.argv[1]

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

if MODE == "tiering":
    extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 1 << 30,
             "block_size": 16,
             "secondary_tiers": [{"type": "fs", "root_dir": "/tmp/shmrepro-kvroot"}]}
else:
    extra = {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": 1 << 30,
             "block_size": 16}

llm = LLM(model="facebook/opt-125m",
          kv_transfer_config=KVTransferConfig(
              kv_connector="OffloadingConnector", kv_role="kv_both",
              kv_connector_extra_config=extra),
          gpu_memory_utilization=0.5, max_model_len=2048, enforce_eager=True)
llm.generate(["hello " * 200], SamplingParams(max_tokens=4), use_tqdm=False)
print("READY files_during_run=",
      sorted(os.path.basename(f) for f in glob.glob("/dev/shm/vllm_offload_*.mmap")),
      flush=True)
while True:
    time.sleep(60)  # 러너가 SIGKILL
