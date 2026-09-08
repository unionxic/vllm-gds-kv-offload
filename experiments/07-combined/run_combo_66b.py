"""07: OPT-66B 가중치 3단(GPU 최대 → host fraction → SSD) + KV→SSD(expfs). 프리픽스 재사용 2라운드.
   round1 = cold(KV 저장), round2 = 프리픽스 적중(KV를 SSD에서 GPU로 로드). GPU KV는 --kv-cache-gib로 최소 고정.
   usage: python run_combo_66b.py --kv-transport cufile|posix|none --tag ..."""
import argparse, json, os, random, resource, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="facebook/opt-66b")
ap.add_argument("--weight-transport", default="cufile", choices=["cufile", "posix"])
ap.add_argument("--kv-transport", default="cufile", choices=["cufile", "posix", "none"])
ap.add_argument("--host-fraction", type=float, default=0.85)
ap.add_argument("--group-size", type=int, default=64)
ap.add_argument("--num-in-group", type=int, default=60)   # GPU 상주 4층(5층은 버퍼·KV와 함께 0.9에 안 들어감)
ap.add_argument("--prefetch-step", type=int, default=1)
ap.add_argument("--io-threads", type=int, default=4)
ap.add_argument("--ring-mb", type=int, default=0)
ap.add_argument("--gpu-util", type=float, default=0.9)
ap.add_argument("--kv-cache-gib", type=float, default=1.5)   # 활성 배치 몫만 GPU에
ap.add_argument("--max-model-len", type=int, default=640)
ap.add_argument("--n-prompts", type=int, default=8)
ap.add_argument("--prefix-tokens", type=int, default=448)
ap.add_argument("--tail-tokens", type=int, default=32)
ap.add_argument("--decode-tokens", type=int, default=8)
ap.add_argument("--kv-threads", type=int, default=4)
ap.add_argument("--kv-block", type=int, default=64)
ap.add_argument("--ssd-root", default=os.path.expanduser("~/experiments/vllm-gds-kv/results/weight-offload/ssd-66b"))
ap.add_argument("--kv-root", default=os.path.expanduser("~/experiments/vllm-gds-kv/results/combined/kv-66b"))
ap.add_argument("--tag", default="run")
args = ap.parse_args()
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "lib"))
OUT = os.path.join(ROOT, "results", "combined", "opt66b"); os.makedirs(OUT, exist_ok=True)

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def meminfo():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS", "VmHWM", "VmLck")):
            k, v = line.split(":"); d[k] = int(v.split()[0]) * 1024
    return d
def nvfs():
    out = {}
    for line in open("/proc/driver/nvidia-fs/stats"):
        if ":" in line:
            k, v = line.split(":", 1)
            for kv in v.split():
                if "=" in kv:
                    a, b = kv.split("=", 1)
                    try: out[f"{k.strip()}.{a}"] = int(b)
                    except ValueError: pass
    return out
def sysmem():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemTotal", "MemAvailable", "MemFree", "Cached", "Dirty", "AnonPages", "Mapped", "Shmem", "Unevictable", "Mlocked", "Slab"): d[k] = int(v.split()[0]) * 1024
    return d

kw = dict(offload_backend="prefetch", offload_group_size=args.group_size, offload_num_in_group=args.num_in_group,
          offload_prefetch_step=args.prefetch_step, offload_ssd_path=args.ssd_root, offload_host_fraction=args.host_fraction,
          offload_ssd_transport=args.weight_transport, offload_ssd_io_threads=args.io_threads, offload_ssd_ring_mb=args.ring_mb)
cnt = {"matched": 0, "kv_read_n": 0, "kv_read_b": 0, "kv_write_n": 0, "kv_write_b": 0}
if args.kv_transport != "none":
    import shutil; shutil.rmtree(args.kv_root, ignore_errors=True)
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
        kv_connector_extra_config={"spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
            "expfs_root_dir": args.kv_root, "expfs_transport": args.kv_transport,
            "expfs_read_threads": args.kv_threads, "expfs_write_threads": args.kv_threads, "block_size": args.kv_block})
    import expfs
    for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport):
        _r, _w = cls.read_chunk, cls.write_chunk
        def mk(fn, kn, kb):
            def inner(self, path, spans, cb):
                cnt[kn] += 1; cnt[kb] += sum(s[2] for s in spans); return fn(self, path, spans, cb)
            return inner
        cls.read_chunk = mk(_r, "kv_read_n", "kv_read_b"); cls.write_chunk = mk(_w, "kv_write_n", "kv_write_b")
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
    _gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
    def _gmw(self, request, n):
        r = _gm(self, request, n); cnt["matched"] += r[0] or 0; return r
    osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

res = dict(args=vars(args), sysmem_start=sysmem(), nvfs_start=nvfs())
t0 = time.time()
llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
          kv_cache_memory_bytes=int(args.kv_cache_gib * 2**30), enforce_eager=True, **kw)
res["load_s"] = round(time.time() - t0, 1); res["mem_after_load"] = meminfo(); res["sysmem_after_load"] = sysmem()
from vllm.model_executor.offloader.base import get_offloader
off = get_offloader()
res["tiers"] = dict(n_modules=len(off.module_offloaders), n_ssd=sum(1 for m in off.module_offloaders if m.mode == "ssd"),
                    host_tier_gib=round(off.host_tier_bytes / 2**30, 2), ssd_tier_gib=round(off.ssd_tier_bytes / 2**30, 2),
                    gpu_resident_layers=args.group_size - args.num_in_group)
vocab_hi = 50000
# 프롬프트마다 서로 다른 토큰열(세션 재방문 패턴). 공유 프리픽스는 GPU 프리픽스 캐시가 먼저 맞아 SSD 경로를 안 탄다.
prompts = [{"prompt_token_ids": [2] + [random.Random(500 + i).randrange(4, vocab_hi) for _ in range(args.prefix_tokens + args.tail_tokens - 1)]}
           for i in range(args.n_prompts)]
sp1 = SamplingParams(max_tokens=1, temperature=0)
spN = SamplingParams(max_tokens=args.decode_tokens, temperature=0, ignore_eos=True)
for rnd in (1, 2):
    for k in cnt: cnt[k] = 0
    c0 = resource.getrusage(resource.RUSAGE_SELF)
    t1 = time.time(); out1 = llm.generate(prompts, sp1); pre = time.time() - t1
    t2 = time.time(); outN = llm.generate(prompts, spN); tot = time.time() - t2
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    res[f"r{rnd}"] = dict(prefill_s=round(pre, 2), gen_total_s=round(tot, 2),
                          decode_step_s=round((tot - pre) / max(args.decode_tokens - 1, 1), 2),
                          cpu_s=round((c1.ru_utime - c0.ru_utime) + (c1.ru_stime - c0.ru_stime), 1),
                          ids=[o.outputs[0].token_ids for o in outN], **{k: v for k, v in cnt.items()})
res["ids_equal_rounds"] = res["r1"]["ids"] == res["r2"]["ids"]
res["mem_end"] = meminfo(); res["sysmem_end"] = sysmem(); res["nvfs_end"] = nvfs()
res["nvfs_delta"] = {k: res["nvfs_end"].get(k, 0) - res["nvfs_start"].get(k, 0) for k in res["nvfs_end"]
                     if res["nvfs_end"].get(k, 0) != res["nvfs_start"].get(k, 0)}
res["gpu_max_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
if getattr(off, "ssd_tier", None) is not None:
    res["weight_ssd_stats"] = dict(off.ssd_tier.stats); res["weight_ssd_stats"]["read_gib"] = round(off.ssd_tier.stats["bytes"] / 2**30, 2)
if args.kv_transport != "none":
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(args.kv_root) for f in fs]
    res["kv_files"] = len(files); res["kv_bytes_gib"] = round(sum(os.path.getsize(f) for f in files) / 2**30, 3)
json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
summary = {k: v for k, v in res.items() if k in ("load_s", "tiers", "ids_equal_rounds", "gpu_max_gib", "kv_files", "kv_bytes_gib")}
summary.update({f"r{i}": {k: v for k, v in res[f"r{i}"].items() if k != "ids"} for i in (1, 2)})
print("RESULT", args.tag, json.dumps(summary), flush=True)
try: llm.llm_engine.engine_core.shutdown()
except Exception as e: print("shutdown:", e)
