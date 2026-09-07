"""SSD tier 첫 기능 테스트: opt-2.7b, host 예산을 극소로 잡아 대부분 층을 SSD로."""
import os, sys, time, json, glob
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams

transport = sys.argv[1] if len(sys.argv) > 1 else "cufile"
root = os.path.expanduser(f"~/experiments/vllm-gds-kv/results/weight-offload/smoke-ssd/{transport}")
def nvfs():
    for l in open("/proc/driver/nvidia-fs/stats"):
        if l.startswith("Reads"):
            return dict(kv.split("=") for kv in l.split(":")[1].split() if "=" in kv)
    return {}
r0 = nvfs()
t0 = time.time()
llm = LLM(model="facebook/opt-2.7b", gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True,
          offload_backend="prefetch", offload_group_size=4, offload_num_in_group=3, offload_prefetch_step=1,
          offload_ssd_path=root, offload_host_fraction=0.005, offload_ssd_transport=transport, offload_ssd_io_threads=4)
load = time.time() - t0
files = glob.glob(root + "/rank0/*/*.bin")
sp = SamplingParams(max_tokens=16, temperature=0)
out = llm.generate(["The capital of France is", "GPUDirect Storage lets"], sp)
ids = [o.outputs[0].token_ids for o in out]
t1 = time.time(); llm.generate(["Hello"] * 4, sp); dec = time.time() - t1
r1 = nvfs()
res = dict(transport=transport, load_s=round(load, 1), files=len(files), file_gb=round(sum(os.path.getsize(f) for f in files)/1e9, 2),
           ids=ids, decode_4x16_s=round(dec, 2), nvfs_reads_before=r0, nvfs_reads_after=r1,
           gpu_max_gib=round(torch.cuda.max_memory_allocated()/2**30, 2))
json.dump(res, open(f"{root}.json", "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
