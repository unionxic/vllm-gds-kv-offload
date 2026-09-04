# in-engine load 프로파일 (nsys -c cudaProfilerApi): warm+drain은 캡처 밖,
# load gen 구간만 캡처. usage: python prof_engine.py {expfs-cufile|expfs-posix|tiering}
import os
import random
import shutil
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib"))

CONFIG = sys.argv[1]

import torch

# tiering C의 load_block에 NVTX (expfs는 자체 계측)
import vllm.v1.kv_offload.tiering.fs.manager as fsm

_ol = fsm.load_block
def _nl(*a, **k):
    with torch.cuda.nvtx.range("tiering.load_block"):
        return _ol(*a, **k)
fsm.load_block = _nl

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

kvroot = os.path.join(HERE, f"kvroot-prof-{CONFIG}")
shutil.rmtree(kvroot, ignore_errors=True)

if CONFIG == "tiering":
    extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 8 << 30,
             "block_size": 16, "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]}
else:
    extra = {"spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
             "expfs_root_dir": kvroot, "expfs_transport": CONFIG.split("-")[1],
             "block_size": 16}

llm = LLM(model="facebook/opt-2.7b",
          kv_transfer_config=KVTransferConfig(
              kv_connector="OffloadingConnector", kv_role="kv_both",
              kv_connector_extra_config=extra),
          gpu_memory_utilization=0.7, max_model_len=2048, enforce_eager=True)
tok = llm.get_tokenizer()
rng = random.Random(7)
sp = SamplingParams(max_tokens=1, temperature=0.0)
prefix = [tok.bos_token_id or 2] + [rng.randrange(1000, tok.vocab_size - 1000)
                                    for _ in range(2031)]
llm.generate([{"prompt_token_ids": prefix + [11, 12]}], sp, use_tqdm=False)
for _ in range(40):  # store 캐스케이드 nudge-drain
    llm.generate([{"prompt_token_ids": [2] + [rng.randrange(1000, tok.vocab_size - 1000)
                                              for _ in range(8)]}], sp, use_tqdm=False)
    time.sleep(0.05)

if CONFIG == "tiering":
    assert llm.reset_prefix_cache(reset_connector=True)  # → C (SSD hit)
else:
    assert llm.reset_prefix_cache()  # → D/E

torch.cuda.profiler.start()
t0 = time.perf_counter()
llm.generate([{"prompt_token_ids": prefix + [13, 14]}], sp, use_tqdm=False)
dt = time.perf_counter() - t0
torch.cuda.profiler.stop()
print(f"ENGINE {CONFIG}: load TTFT {dt:.3f}s ({dt/127*1e3:.2f} ms/chunk)")
