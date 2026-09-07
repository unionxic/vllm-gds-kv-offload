"""ring 모드 QA(opt-2.7b): cufile+ring 16MiB → 토큰이 baseline과 일치하고 TRACE 분류가 native인지."""
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault(
    "CUFILE_ENV_PATH_JSON",
    os.path.join(HERE, "..", "01-feasibility", "cufile-microbench", "cufile_trace.json"),
)
ring = int(sys.argv[1]) if len(sys.argv) > 1 else 16
step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
root = os.path.join(HERE, "..", "..", "results", "weight-offload", "qa-ssd", f"ring{ring}-s{step}-t{threads}")
os.makedirs(root, exist_ok=True)
os.chdir(root)
if os.path.exists("cufile.log"):
    os.remove("cufile.log")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def nvfs():
    for line in open("/proc/driver/nvidia-fs/stats"):
        if line.startswith("Reads"):
            d = dict(kv.split("=") for kv in line.split(":")[1].split() if "=" in kv)
            return int(d.get("n", 0))
    return 0


n0 = nvfs()
llm = LLM(model="facebook/opt-2.7b", gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True,
          offload_backend="prefetch", offload_group_size=4, offload_num_in_group=3, offload_prefetch_step=step,
          offload_ssd_path=os.path.join(root, "ssd"), offload_host_fraction=0.005,
          offload_ssd_transport="cufile", offload_ssd_io_threads=threads, offload_ssd_ring_mb=ring)
base = json.load(open(os.path.join(HERE, "..", "..", "results", "weight-offload", "qa-ssd", "baseline.json")))
prompts = ["The capital of France is", "GPUDirect Storage lets",
           "In a distant galaxy, a small robot named", "The recipe for a perfect omelette starts with"]
sp = SamplingParams(max_tokens=24, temperature=0)
out = llm.generate(prompts, sp)
ids = [o.outputs[0].token_ids for o in out]
t = time.time(); llm.generate(["Hello"] * 4, SamplingParams(max_tokens=32, temperature=0)); dec = time.time() - t
n1 = nvfs()
text = open("cufile.log", errors="replace").read() if os.path.exists("cufile.log") else ""
px = len(re.findall(r"cufio-px.*(?:read|write)", text, re.I)) - text.count("px-pool")
bounce = len(re.findall(r"read_through_bounce_buffer completed", text))
cls = "COMPAT-POSIX" if px > 0 else ("INTERNAL-BOUNCE" if bounce else "DIRECT")
base_ids = base.get("token_ids") or base.get("ids")
match = ids == base_ids
res = dict(ring=ring, step=step, threads=threads, decode_4x32_s=round(dec, 2), nvfs_reads=n1 - n0, cls=cls,
           px=max(0, px), bounce=bounce, token_match=match)
json.dump(res, open("result.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
print("QA", "PASS" if (match and cls != "COMPAT-POSIX") else "FAIL", flush=True)
