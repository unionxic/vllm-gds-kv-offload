"""OPT-66B 3단(GPU→pinned CPU→SSD) 가중치 오프로드 측정 러너.

한 프로세스 = 한 arm. 결과는 JSON 한 개.
  python run_66b.py --transport cufile --host-fraction 0.3 --tag c-h0.3-r1
측정: 로드 시간, tier 배치, prefill(TTFT), decode step 평균, RSS/pinned, nvidia-fs 카운터 델타,
      SSD tier 읽기 통계(ssd_tier.stats), GPU 최대 할당.
"""
import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="facebook/opt-66b")
ap.add_argument("--transport", choices=["cufile", "posix", "cpu", "none"], default="cufile")
ap.add_argument("--host-fraction", type=float, default=0.3)
ap.add_argument("--group-size", type=int, default=64)
ap.add_argument("--num-in-group", type=int, default=61)
ap.add_argument("--prefetch-step", type=int, default=1)
ap.add_argument("--io-threads", type=int, default=4)
ap.add_argument("--gpu-util", type=float, default=0.9)
ap.add_argument("--max-model-len", type=int, default=1024)
ap.add_argument("--prompt-tokens", type=int, default=256)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--decode-tokens", type=int, default=8)
ap.add_argument("--ssd-root", default=os.path.expanduser("~/experiments/vllm-gds-kv/results/weight-offload/ssd-66b"))
ap.add_argument("--out", default=None)
ap.add_argument("--tag", default="run")
args = ap.parse_args()

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def meminfo():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM", "VmLck")):
            k, v = line.split(":")
            d[k] = int(v.split()[0]) * 1024
    return d


def nvfs():
    out = {}
    for line in open("/proc/driver/nvidia-fs/stats"):
        if ":" in line:
            k, v = line.split(":", 1)
            for kv in v.split():
                if "=" in kv:
                    a, b = kv.split("=", 1)
                    try:
                        out[f"{k.strip()}.{a}"] = int(b)
                    except ValueError:
                        pass
    return out


def sysmem():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemTotal", "MemAvailable", "Cached", "Dirty", "Mlocked", "Unevictable"):
            d[k] = int(v.split()[0]) * 1024
    return d


kw = {}
if args.transport in ("cufile", "posix", "cpu"):
    kw.update(offload_backend="prefetch", offload_group_size=args.group_size,
              offload_num_in_group=args.num_in_group, offload_prefetch_step=args.prefetch_step)
if args.transport in ("cufile", "posix"):
    kw.update(offload_ssd_path=os.path.join(args.ssd_root, args.transport),
              offload_host_fraction=args.host_fraction,
              offload_ssd_transport=args.transport,
              offload_ssd_io_threads=args.io_threads)

res = dict(args=vars(args), sysmem_start=sysmem(), nvfs_start=nvfs())
t0 = time.time()
llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_util,
          max_model_len=args.max_model_len, enforce_eager=True, **kw)
res["load_s"] = round(time.time() - t0, 1)
res["mem_after_load"] = meminfo()
res["sysmem_after_load"] = sysmem()
res["gpu_after_load_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)

from vllm.model_executor.offloader.base import get_offloader  # noqa: E402
off = get_offloader()
tiers = {}
if hasattr(off, "module_offloaders"):
    tiers = dict(
        n_modules=len(off.module_offloaders),
        n_ssd=sum(1 for m in off.module_offloaders if m.mode == "ssd"),
        host_tier_gib=round(getattr(off, "host_tier_bytes", 0) / 2**30, 2),
        ssd_tier_gib=round(getattr(off, "ssd_tier_bytes", 0) / 2**30, 2),
        host_budget_gib=round(getattr(off, "host_budget_bytes", 0) / 2**30, 2),
        buffer_pool_gib=round(off.buffer_pool.total_bytes / 2**30, 2) if off.buffer_pool else 0,
    )
res["tiers"] = tiers

# 결정론적 prompt (토큰 id 직접): 배치 내 서로 다른 시드
import random  # noqa: E402
vocab_hi = 50000
prompts = []
for b in range(args.batch):
    r = random.Random(1000 + b)
    prompts.append({"prompt_token_ids": [2] + [r.randrange(4, vocab_hi) for _ in range(args.prompt_tokens - 1)]})

# 1) prefill만 (max_tokens=1) → TTFT
sp1 = SamplingParams(max_tokens=1, temperature=0)
cpu0 = resource.getrusage(resource.RUSAGE_SELF)
t1 = time.time(); out1 = llm.generate(prompts, sp1); pre = time.time() - t1
res["prefill_batch_s"] = round(pre, 2)
# 2) prefill + decode N → step 평균 = (총 - prefill)/(N-1)
spN = SamplingParams(max_tokens=args.decode_tokens, temperature=0, ignore_eos=True)
t2 = time.time(); outN = llm.generate(prompts, spN); tot = time.time() - t2
cpu1 = resource.getrusage(resource.RUSAGE_SELF)
res["gen_total_s"] = round(tot, 2)
res["decode_step_s"] = round((tot - pre) / max(args.decode_tokens - 1, 1), 2)
res["out_tok_per_s"] = round(args.batch * args.decode_tokens / tot, 3)
res["cpu_s"] = round((cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime), 1)
res["ids"] = [o.outputs[0].token_ids for o in outN]
res["ids_prefill"] = [o.outputs[0].token_ids for o in out1]
res["mem_end"] = meminfo()
res["sysmem_end"] = sysmem()
res["nvfs_end"] = nvfs()
res["nvfs_delta"] = {k: res["nvfs_end"].get(k, 0) - res["nvfs_start"].get(k, 0)
                     for k in res["nvfs_end"] if res["nvfs_end"].get(k, 0) != res["nvfs_start"].get(k, 0)}
res["gpu_max_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
if getattr(off, "ssd_tier", None) is not None:
    res["ssd_stats"] = dict(off.ssd_tier.stats)
    res["ssd_stats"]["read_gib"] = round(off.ssd_tier.stats["bytes"] / 2**30, 2)

out = args.out or os.path.expanduser(f"~/experiments/vllm-gds-kv/results/weight-offload/opt66b/{args.tag}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, "w"), indent=1)
summary = {k: res[k] for k in ("load_s", "prefill_batch_s", "decode_step_s", "out_tok_per_s", "cpu_s", "gpu_max_gib")}
summary["tiers"] = tiers
summary["rss_gib"] = round(res["mem_end"]["VmRSS"] / 2**30, 2)
summary["ssd_read_gib"] = res.get("ssd_stats", {}).get("read_gib")
print("RESULT", args.tag, json.dumps(summary), flush=True)
