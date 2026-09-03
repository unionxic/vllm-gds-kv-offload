# 5단계: real-text synthetic-interleaving admission workload replay.
#   mixed_workload.json 스케줄을 따라가며 카테고리별 useful hit·저장 행동을 측정.
#   변별 가설: value_density(reuse-distance)가 near_reuse(GPU/CPU가 잡음)를 저장 안 하고
#   far_reuse(evict→SSD 필요)만 저장 → seen_twice보다 적은 write로 같은 far hit.
# usage: python replay_mixed.py --run-id <id> --arm {C|Sskip|Sseen|Svalue} [--runner v1]
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True, choices=["C", "Sskip", "Sseen", "Svalue"])
ap.add_argument("--note", default="")
args = ap.parse_args()

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "sched"),
          os.path.join(HERE, "..", "phase2"), os.path.join(HERE, "..", "phase1"),
          os.path.join(HERE, "..", "admission")):
    sys.path.insert(0, p)
import snapshot  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "admission", "mixed", args.run_id)
os.makedirs(OUT, exist_ok=True)
MW = json.load(open(os.path.join(HERE, "mixed_workload.json")))
W = json.load(open(os.path.join(HERE, "workload.json")))
docs = W["docs"]
delim = MW["delim_tokens"]
sched = MW["schedule"]

cnt = {"tp_read_n": 0, "tp_read_b": 0, "tp_write_n": 0, "tp_write_b": 0,
       "matched": 0, "guard_skips": 0}

# transport 설정
if args.arm == "C":
    transport, policy, vmode = None, None, None
elif args.arm == "Sskip":
    transport, policy, vmode = "cufile_staged", "skip", None
elif args.arm == "Sseen":
    transport, policy, vmode = "cufile_staged", "value", "seen_twice"
else:  # Svalue
    transport, policy, vmode = "cufile_staged", "value", "value_density"

if args.arm != "C":
    import expfs  # noqa: E402
    for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport):
        _r, _w = cls.read_chunk, cls.write_chunk
        def mk(fn, kn, kb):
            def inner(self, path, spans, cb):
                cnt[kn] += 1
                cnt[kb] += sum(s[2] for s in spans)
                return fn(self, path, spans, cb)
            return inner
        cls.read_chunk = mk(_r, "tp_read_n", "tp_read_b")
        cls.write_chunk = mk(_w, "tp_write_n", "tp_write_b")
    _ws0 = expfs.StagedCuFileTransport.write_slot
    def _ws_cnt(self, slot, path, kind="ring"):
        cnt["tp_write_n"] += 1
        cnt["tp_write_b"] += self.chunk_bytes
        return _ws0(self, slot, path, kind)
    expfs.StagedCuFileTransport.write_slot = _ws_cnt
    import vllm.v1.kv_offload.tiering.fs.manager as fsm  # noqa: E402
    _ol, _os_ = fsm.load_block, fsm.store_block
    fsm.load_block = lambda p, v, o, b: (cnt.__setitem__("tp_read_n", cnt["tp_read_n"] + 1),
                                         cnt.__setitem__("tp_read_b", cnt["tp_read_b"] + b),
                                         _ol(p, v, o, b))[-1]
    fsm.store_block = lambda p, bf, o, b: (cnt.__setitem__("tp_write_n", cnt["tp_write_n"] + 1),
                                           cnt.__setitem__("tp_write_b", cnt["tp_write_b"] + b),
                                           _os_(p, bf, o, b))[-1]
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched  # noqa: E402
_gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
def _gmw(self, request, nct):
    r = _gm(self, request, nct)
    if r[0]:
        cnt["matched"] += r[0]
    return r
osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

import shutil
kvroot = os.path.join(HERE, f"kvroot-{args.run_id}")
shutil.rmtree(kvroot, ignore_errors=True)
if args.arm == "C":
    ktc = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                           kv_connector_extra_config={
                               "spec_name": "TieringOffloadingSpec",
                               "cpu_bytes_to_use": 8 << 30, "block_size": 16,
                               "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]})
else:
    ktc = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                           kv_connector_extra_config={
                               "spec_name": "ExperimentalFilesystemSpec",
                               "spec_module_path": "expfs", "expfs_root_dir": kvroot,
                               "expfs_transport": transport, "block_size": 64,
                               "expfs_staging_slots": 6, "expfs_staging_writers": 2,
                               "expfs_staging_policy": policy,
                               "expfs_cpu_fallback_slots": 8})

meta = {"args": vars(args), "start": snapshot.collect("start")}
llm = LLM(model="facebook/opt-2.7b", kv_transfer_config=ktc,
          gpu_memory_utilization=0.7, max_model_len=2048, enforce_eager=True)

if policy == "value":
    import value_admission  # noqa: E402
    value_admission.install(expfs.LAST_WORKER.transport, vmode)

sp = SamplingParams(max_tokens=1, temperature=0.0)
rows = []
t0, c0 = time.perf_counter(), time.process_time()
for k, s in enumerate(sched):
    d = docs[s["doc"]]
    q = d["questions"][s["q"]]
    toks = d["prefix"] + delim + q["tokens"]
    before = dict(cnt)
    tg = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": toks}], sp, use_tqdm=False)[0]
    ttft = time.perf_counter() - tg
    dl = {kk: cnt[kk] - before[kk] for kk in cnt}
    rows.append(dict(k=k, doc=s["doc"], category=s["category"], occ=s["occ"],
                     ttft_s=round(ttft, 4), matched=dl["matched"],
                     ssd_read_ios=dl["tp_read_n"], write_ios=dl["tp_write_n"],
                     cached=getattr(out, "num_cached_tokens", None)))

wall = time.perf_counter() - t0
cpu = time.process_time() - c0
if args.arm != "C":
    _w = expfs.LAST_WORKER
    if hasattr(_w.transport, "flush"):
        _w.transport.flush()
        meta["staged_stats"] = dict(_w.transport.stats)
        meta["staged_write_errors"] = _w.transport.write_errors
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
meta["io_threads_schedstat"] = snapshot.thread_schedstat()
try:
    llm.llm_engine.engine_core.shutdown()
except Exception as e:
    print("shutdown:", e)
meta["end"] = snapshot.collect("end")
meta["totals"] = dict(wall_s=round(wall, 2), cpu_s=round(cpu, 2), **cnt)

# 카테고리별 useful hit (재등장=occ>0에서 matched>0)
cat = defaultdict(lambda: dict(reqs=0, hit_reqs=0, matched=0))
for r in rows:
    c = cat[r["category"]]
    c["reqs"] += 1
    if r["occ"] > 0:
        if r["matched"] > 0:
            c["hit_reqs"] += 1
        c["matched"] += r["matched"]
meta["by_category"] = {k: dict(v) for k, v in cat.items()}

with open(os.path.join(OUT, "raw.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

tt = sorted(r["ttft_s"] for r in rows)
p = lambda q: tt[min(len(tt) - 1, int(round(q * (len(tt) - 1))))]
wb = cnt["tp_write_b"] / 2**30
print(f"RESULT {args.run_id} arm={args.arm} p95={p(.95):.3f} "
      f"matched={cnt['matched']} write={wb:.2f}GB "
      f"far_hit={cat['far_reuse']['matched']} near_hit={cat['near_reuse']['matched']} "
      f"rep_hit={cat['repeated']['matched']} wall={wall:.0f}s cpu={cpu:.0f}s", flush=True)
shutil.rmtree(kvroot, ignore_errors=True)
