"""Prefetch weight-offload 백엔드가 이 빌드(V1 러너, eager)에서 동작하는지 스모크.
opt-2.7b 32층 중 group 4 / num_in_group 2 → 절반(16층)을 pinned CPU로 내리고 토큰 일치·GPU 메모리 감소 확인."""
import os, sys, time
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams

mode = sys.argv[1] if len(sys.argv) > 1 else "prefetch"
kw = {}
if mode == "prefetch":
    kw = dict(offload_backend="prefetch", offload_group_size=4, offload_num_in_group=2, offload_prefetch_step=1)
elif mode == "uva":
    kw = dict(offload_backend="uva", cpu_offload_gb=2.5)
t0 = time.time()
llm = LLM(model="facebook/opt-2.7b", gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True, **kw)
print(f"[{mode}] load {time.time()-t0:.1f}s, gpu alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB, "
      f"max {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
sp = SamplingParams(max_tokens=16, temperature=0)
out = llm.generate(["The capital of France is", "GPUDirect Storage lets"], sp)
for o in out:
    print(f"[{mode}] {o.prompt!r} -> {o.outputs[0].token_ids}", flush=True)
t1 = time.time(); llm.generate(["Hello"] * 4, sp); print(f"[{mode}] 4x16tok decode {time.time()-t1:.2f}s", flush=True)
