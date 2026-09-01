# W2 LEval 리플레이 (W2a: max_tokens=1, TTFT isolation).
#   Round 1: 문서 i 질문 1 (cold/store) → Round 2: seed shuffle된 문서의 질문 2 (reuse).
# usage:
#   python replay.py --run-id <id> --arm {A|C|Dsync|Dsched|Esync|Esched}
#                    [--sched S1|S2|S3] [--docs 24] [--max-w-bytes N] [...]
import argparse
import csv
import json
import os
import random
import shutil
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True,
                choices=["A", "C", "D0", "D1", "D2", "E0", "E1", "DS1", "DS2"])
ap.add_argument("--docs", type=int, default=24)
ap.add_argument("--max-w-bytes", type=int, default=None)
ap.add_argument("--write-quantum", type=int, default=2)
ap.add_argument("--note", default="")
args = ap.parse_args()

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "sched"),
          os.path.join(HERE, "..", "phase2"), os.path.join(HERE, "..", "phase1")):
    sys.path.insert(0, p)
import snapshot  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "w2_leval", "raw", args.run_id)
os.makedirs(OUT, exist_ok=True)
meta = {"args": vars(args), "start": snapshot.collect("start")}

W = json.load(open(os.path.join(HERE, "workload.json")))
docs = W["docs"][:args.docs]
delim = W["delim_tokens"]
PT = W["prefix_tokens"]

# ---- 용량 사전 계산 (opt-2.7b: 320KiB/token) ----
BPT = 327_680
prefix_kv = PT * BPT
meta["capacity"] = dict(
    bytes_per_token=BPT, prefix_kv_bytes=prefix_kv,
    unique_prefix_kv_gib=round(len(docs) * prefix_kv / 2**30, 2),
    gpu_kv_est_gib=5.5, cpu_budget_gib=8.0,
    ws_over_gpu=round(len(docs) * prefix_kv / 2**30 / 5.5, 2),
    ws_over_gpu_cpu=round(len(docs) * prefix_kv / 2**30 / 13.5, 2),
)
print("capacity:", meta["capacity"], flush=True)

ARM_MODE = {"D0": "S0", "D1": "DEF", "D2": "DEFB",
            "E0": "S0", "E1": "DEF", "DS1": "S1", "DS2": "S2", "C": None, "A": None}
mode = ARM_MODE[args.arm]
transport = ("cufile" if args.arm.startswith("D")
             else "posix" if args.arm.startswith("E") else None)

cnt = {"tp_read_n": 0, "tp_read_b": 0, "tp_write_n": 0, "tp_write_b": 0,
       "fs_load_n": 0, "fs_load_b": 0, "fs_store_n": 0, "fs_store_b": 0, "matched": 0, "guard_skips": 0}

if args.arm != "A":
    import expfs  # noqa: E402
    import scheduler as sch  # noqa: E402
    if mode:
        sch.install(mode, max_w_bytes=args.max_w_bytes,
                    write_quantum_chunks=args.write_quantum)
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
    import vllm.v1.kv_offload.tiering.fs.manager as fsm
    _ol, _os_ = fsm.load_block, fsm.store_block
    def _cl(path, view, off, bs):
        cnt["fs_load_n"] += 1
        cnt["fs_load_b"] += bs
        return _ol(path, view, off, bs)
    def _cs(path, buf, off, bs):
        cnt["fs_store_n"] += 1
        cnt["fs_store_b"] += bs
        return _os_(path, buf, off, bs)
    fsm.load_block, fsm.store_block = _cl, _cs
    import vllm.v1.kv_offload.tiering.manager as tmgr
    _ps = tmgr.TieringOffloadingManager.prepare_store
    def _ps_guard(self, keys, req_context):
        if req_context.req_id not in self._req_state:
            cnt["guard_skips"] += 1
            return None
        return _ps(self, keys, req_context)
    tmgr.TieringOffloadingManager.prepare_store = _ps_guard
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
    _gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
    def _gmw(self, request, nct):
        r = _gm(self, request, nct)
        if r[0]:
            cnt["matched"] += r[0]
        return r
    osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

kvroot = os.path.join(HERE, f"kvroot-{args.run_id}")
shutil.rmtree(kvroot, ignore_errors=True)
ktc = None
if args.arm == "C":
    ktc = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                           kv_connector_extra_config={
                               "spec_name": "TieringOffloadingSpec",
                               "cpu_bytes_to_use": 8 << 30, "block_size": 16,
                               "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]})
elif args.arm != "A":
    ktc = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                           kv_connector_extra_config={
                               "spec_name": "ExperimentalFilesystemSpec",
                               "spec_module_path": "expfs",
                               "expfs_root_dir": kvroot,
                               "expfs_transport": transport, "block_size": 64})

llm = LLM(model="facebook/opt-2.7b", kv_transfer_config=ktc,
          gpu_memory_utilization=0.7, max_model_len=2048, enforce_eager=True)
sp = SamplingParams(max_tokens=1, temperature=0.0)

# Gate 0 audit: PID/TID 맵
meta["audit_threads"] = [
    dict(name=t.name, tid=t.native_id) for t in threading.enumerate()]
meta["audit_pid"] = os.getpid()


def gen(token_ids):
    t0 = time.perf_counter()
    out = llm.generate([{"prompt_token_ids": token_ids}], sp, use_tqdm=False)
    return time.perf_counter() - t0, out[0]


def request_tokens(doc, q):
    return doc["prefix"] + delim + q["tokens"]


rows = []
gap_worker: list = []
t_all0, c_all0 = time.perf_counter(), time.process_time()
order1 = list(range(len(docs)))
rng = random.Random(W["seed"] + 1)
order2 = list(range(len(docs)))
rng.shuffle(order2)

for rnd, order, qidx in ((1, order1, 0), (2, order2, 1)):
    for di in order:
        d = docs[di]
        q = d["questions"][min(qidx, len(d["questions"]) - 1)]
        toks = request_tokens(d, q)
        before = dict(cnt)
        ttft, out = gen(toks)
        dl = {k: cnt[k] - before[k] for k in cnt}
        if args.arm in ("D1", "D2", "E1"):
            import scheduler as sch2
            sch2.release_gap(gap_worker)  # foreground idle 구간에서 store 방출·drain
        rows.append(dict(round=rnd, doc=di, subtask=d["subtask"], q=q["q_index"],
                         prompt_tokens=len(toks), ttft_s=round(ttft, 4),
                         cached=getattr(out, "num_cached_tokens", None),
                         matched=dl["matched"],
                         ssd_read_ios=dl["tp_read_n"] + dl["fs_load_n"],
                         ssd_read_b=dl["tp_read_b"] + dl["fs_load_b"],
                         write_ios=dl["tp_write_n"], write_b=dl["tp_write_b"]))
    print(f"[{args.run_id}] round {rnd} done", flush=True)

# 최종 drain: 남은 보류 store 전부 실행·완료 (wall/CPU에 포함 — 생략 아님 증명)
if args.arm in ("D1", "D2", "E1"):
    import scheduler as sch3
    n_final = sch3.release_gap(gap_worker)
    print(f"final drain released={n_final}", flush=True)
wall = time.perf_counter() - t_all0
cpu = time.process_time() - c_all0
# IO 스레드 생존 중에 schedstat 캡처 (shutdown 후엔 스레드 소멸 — Gate 2 교훈)
meta["io_threads_schedstat"] = snapshot.thread_schedstat()
try:
    llm.llm_engine.engine_core.shutdown()
except Exception as e:
    print("shutdown:", e)

meta["end"] = snapshot.collect("end")
meta["totals"] = dict(wall_s=round(wall, 2), cpu_s=round(cpu, 2), **cnt)
if args.arm != "A":
    import scheduler as sch
    meta["sched"] = sch.summary()
    assert meta["sched"]["deferred_pending"] == 0, "store backlog가 남으면 안 됨"
with open(os.path.join(OUT, "raw.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

r2 = [r for r in rows if r["round"] == 2]
tt = sorted(x["ttft_s"] for x in r2)
p = lambda qv: tt[min(len(tt) - 1, int(round(qv * (len(tt) - 1))))]
hits = sum(1 for r in r2 if r["ssd_read_ios"] > 0)
print(f"RESULT {args.run_id} arm={args.arm} mode={mode or '-'} "
      f"docs={len(docs)} R2 p50={p(.5):.3f} p95={p(.95):.3f} "
      f"fsHitReqs={hits}/{len(r2)} matched={cnt['matched']} "
      f"wall={wall:.1f}s cpu={cpu:.1f}s", flush=True)
shutil.rmtree(kvroot, ignore_errors=True)
