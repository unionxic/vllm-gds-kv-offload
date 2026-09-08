"""pinned 호스트 티어 footprint 검증: Shmem 증가량 / host_tier_bytes (1.0이면 올림 없음). usage: VLLM_OFFLOAD_PIN_EXACT=0|1 python pin_exact_test.py"""
import os, sys, time
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
def mi(k):
    for l in open("/proc/meminfo"):
        if l.startswith(k + ":"): return int(l.split()[1]) * 1024
from vllm import LLM, SamplingParams
sh0, av0 = mi("Shmem"), mi("MemAvailable")
llm = LLM(model="facebook/opt-2.7b", dtype="float16", gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=True,
          offload_backend="prefetch", offload_group_size=64, offload_num_in_group=60, offload_prefetch_step=1,
          offload_ssd_path=os.path.expanduser("~/experiments/vllm-gds-kv/results/combined/pin-test-ssd"), offload_host_fraction=0.3,
          offload_ssd_transport="cufile", offload_ssd_io_threads=2)
from vllm.model_executor.offloader.base import get_offloader
off = get_offloader(); tier = off.host_tier_bytes
sh1, av1 = mi("Shmem"), mi("MemAvailable")
out = llm.generate([{"prompt_token_ids": [2] + list(range(100, 132))}], SamplingParams(max_tokens=4, temperature=0))
print("PINTEST exact=%s host_tier=%.2fGiB shmem_delta=%.2fGiB ratio=%.3f avail_drop=%.2fGiB ids=%s" % (
    os.environ.get("VLLM_OFFLOAD_PIN_EXACT", "0"), tier / 2**30, (sh1 - sh0) / 2**30, (sh1 - sh0) / tier, (av0 - av1) / 2**30, out[0].outputs[0].token_ids), flush=True)
