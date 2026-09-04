# W3-openloop: 동시성/open-loop에서 지연 store의 굶주림(starvation) 검증.
#   W2b의 backlog 안정성은 closed-loop(요청 경계 gap 보장)에서만 확인됨.
#   여기서는 sync LLMEngine의 add_request+step을 직접 구동해 continuous batching을
#   유지한 채(in-process, 몽키패치 유효) 동시성 N closed-loop과 Poisson open-loop을 잰다.
#   AsyncLLM은 항상 별도 프로세스(make_async_mp_client)라 쓸 수 없다.
# usage:
#   python openloop.py --run-id <id> --arm {C|D1|E1|DS4} --conc 4
#   python openloop.py --run-id <id> --arm D1 --arrival poisson --rate 0.5
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
                choices=["C", "D0", "D1", "E1", "DS4", "ST"])
ap.add_argument("--docs", type=int, default=64)
ap.add_argument("--conc", type=int, default=1, help="closed-loop 동시 스트림 수")
ap.add_argument("--arrival", choices=["closed", "poisson"], default="closed")
ap.add_argument("--rate", type=float, default=0.5, help="poisson 도착률 req/s")
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

OUT = os.path.join(HERE, "..", "results", "w2_openloop", "raw", args.run_id)
os.makedirs(OUT, exist_ok=True)
meta = {"args": vars(args), "start": snapshot.collect("start")}

W = json.load(open(os.path.join(HERE, "workload.json")))
docs = W["docs"][:args.docs]
delim = W["delim_tokens"]

ARM_MODE = {"D0": "S0", "D1": "DEF", "E1": "DEF", "DS4": "S4",
            "ST": "S0", "C": None}
mode = ARM_MODE[args.arm]
transport = ("cufile_staged" if args.arm == "ST"
             else "cufile" if args.arm in ("D0", "D1", "DS4")
             else "posix" if args.arm == "E1" else None)
if args.arm == "DS4" and args.max_w_bytes is None:
    args.max_w_bytes = 40 << 20

cnt = {"tp_read_n": 0, "tp_read_b": 0, "tp_write_n": 0, "tp_write_b": 0,
       "fs_load_n": 0, "fs_load_b": 0, "fs_store_n": 0, "fs_store_b": 0,
       "matched": 0, "guard_skips": 0}

import expfs  # noqa: E402
import scheduler as sch  # noqa: E402
if mode:
    sch.install(mode, max_w_bytes=args.max_w_bytes,
                write_quantum_chunks=args.write_quantum)
if os.environ.get("FIX"):  # 인과 폐쇄 재검증용 (sched/prof_fix.py)
    import prof_fix  # noqa: E402
    prof_fix.install()
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
def _cl(path, view, off, bs):
    cnt["fs_load_n"] += 1
    cnt["fs_load_b"] += bs
    return _ol(path, view, off, bs)
def _cs(path, buf, off, bs):
    cnt["fs_store_n"] += 1
    cnt["fs_store_b"] += bs
    return _os_(path, buf, off, bs)
fsm.load_block, fsm.store_block = _cl, _cs
import vllm.v1.kv_offload.tiering.manager as tmgr  # noqa: E402
_ps = tmgr.TieringOffloadingManager.prepare_store
def _ps_guard(self, keys, req_context):
    if req_context.req_id not in self._req_state:
        cnt["guard_skips"] += 1
        return None
    return _ps(self, keys, req_context)
tmgr.TieringOffloadingManager.prepare_store = _ps_guard
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
from vllm.sampling_params import RequestOutputKind  # noqa: E402

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
                               "spec_module_path": "expfs",
                               "expfs_root_dir": kvroot,
                               "expfs_transport": transport, "block_size": 64})

llm = LLM(model="facebook/opt-2.7b", kv_transfer_config=ktc,
          gpu_memory_utilization=0.7, max_model_len=2048, enforce_eager=True)
eng = llm.llm_engine


def sp_for(q):
    n = min(max(q.get("ref_output_tokens", 16), 16), 128)
    s = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
    s.output_kind = RequestOutputKind.CUMULATIVE  # step마다 출력 → TTFT 실측
    return s


# ---- 요청 시퀀스: R1(cold, 문서순, q1) → R2(shuffle, q2), 단일 도착 스트림 ----
rng = random.Random(W["seed"] + 1)
order2 = list(range(len(docs)))
rng.shuffle(order2)
reqs = []
for rnd, order, qidx in ((1, list(range(len(docs))), 0), (2, order2, 1)):
    for di in order:
        d = docs[di]
        q = d["questions"][min(qidx, len(d["questions"]) - 1)]
        reqs.append(dict(rid=f"r{len(reqs)}", round=rnd, doc=di,
                         subtask=d["subtask"], q=q["q_index"],
                         toks=d["prefix"] + delim + q["tokens"], sp=sp_for(q)))
arrivals = None
if args.arrival == "poisson":
    arng = random.Random(W["seed"] + 7)
    t = 0.0
    arrivals = []
    for _ in reqs:
        t += arng.expovariate(args.rate)
        arrivals.append(t)

# ---- 1Hz 샘플러: backlog/inflight/kvroot 크기/디스크 여유 시계열 ----
ts_rows = []
inflight_now = [0]
stop_sampler = threading.Event()
abort = [None]


def kvroot_bytes():
    total = 0
    for root, _dirs, files in os.walk(kvroot):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def sampler():
    t0 = time.perf_counter()
    while not stop_sampler.wait(1.0):
        backlog = sch.summary()["deferred_pending"] if mode else 0
        free = shutil.disk_usage("/").free
        ts_rows.append(dict(t_s=round(time.perf_counter() - t0, 1),
                            backlog=backlog, inflight=inflight_now[0],
                            kvroot_b=kvroot_bytes(), disk_free_b=free))
        if free < 25 << 30:
            abort[0] = "disk_free<25GB"


threading.Thread(target=sampler, daemon=True).start()

# ---- 구동 루프 ----
rows = []
active = {}
gap_worker: list = []
gaps = 0
next_idx = 0
need_gap = False
steps_total = 0
steps_empty = 0
t_all0, c_all0 = time.perf_counter(), time.process_time()


def submit(i):
    r = reqs[i]
    eng.add_request(r["rid"], {"prompt_token_ids": r["toks"]}, r["sp"])
    active[r["rid"]] = dict(meta=r, submit=time.perf_counter(), first=None)
    inflight_now[0] = len(active)


while (next_idx < len(reqs) or active) and not abort[0]:
    now_rel = time.perf_counter() - t_all0
    # gap 방출은 foreground가 완전히 비었고 즉시 넣을 도착도 없을 때만 (W2b 의미 유지)
    if not active and need_gap and args.arm in ("D1", "E1"):
        due = (next_idx < len(reqs)
               and (args.arrival == "closed" or arrivals[next_idx] <= now_rel))
        if not due or args.arrival == "closed":
            if sch.release_gap(gap_worker) > 0:
                gaps += 1
        need_gap = False
    # 도착 admission
    if args.arrival == "closed":
        while next_idx < len(reqs) and len(active) < args.conc:
            submit(next_idx)
            next_idx += 1
    else:
        while next_idx < len(reqs) and arrivals[next_idx] <= now_rel:
            submit(next_idx)
            next_idx += 1
    if active:
        outs = eng.step()
        steps_total += 1
        if not outs:
            steps_empty += 1  # 요청은 있으나 아무것도 진행 못한 step (블록 대기 등)
        for out in outs:
            st = active.get(out.request_id)
            if st is None:
                continue
            if st["first"] is None and out.outputs and out.outputs[0].token_ids:
                st["first"] = time.perf_counter()
            if out.finished:
                fin = time.perf_counter()
                m = st["meta"]
                rows.append(dict(
                    rid=m["rid"], round=m["round"], doc=m["doc"],
                    subtask=m["subtask"], q=m["q"],
                    prompt_tokens=len(m["toks"]),
                    submit_rel_s=round(st["submit"] - t_all0, 3),
                    ttft_s=round((st["first"] or fin) - st["submit"], 4),
                    e2e_s=round(fin - st["submit"], 4),
                    out_tokens=len(out.outputs[0].token_ids),
                    cached=getattr(out, "num_cached_tokens", None),
                    backlog=sch.summary()["deferred_pending"] if mode else 0))
                del active[st["meta"]["rid"]]
                inflight_now[0] = len(active)
                if not active:
                    need_gap = True
    elif next_idx < len(reqs):
        time.sleep(min(0.02, max(0.0, arrivals[next_idx] - (time.perf_counter() - t_all0))))

# 최종 drain (wall/CPU에 포함)
if mode:
    n_final = sch.release_gap(gap_worker)
    print(f"final drain released={n_final}", flush=True)
_w = getattr(expfs, "LAST_WORKER", None)
if _w is not None and hasattr(_w.transport, "flush"):
    _w.transport.flush()  # staged 잔여 비동기 쓰기를 wall/CPU에 포함
    meta["staged_write_errors"] = getattr(_w.transport, "write_errors", 0)
wall = time.perf_counter() - t_all0
cpu = time.process_time() - c_all0
stop_sampler.set()
meta["io_threads_schedstat"] = snapshot.thread_schedstat()
try:
    eng.engine_core.shutdown()
except Exception as e:
    print("shutdown:", e)

meta["end"] = snapshot.collect("end")
out_tok = sum(r["out_tokens"] for r in rows)
meta["totals"] = dict(wall_s=round(wall, 2), cpu_s=round(cpu, 2),
                      gaps=gaps, out_tokens=out_tok,
                      out_tok_per_s=round(out_tok / wall, 2),
                      completed=len(rows), aborted=abort[0],
                      steps_total=steps_total, steps_empty=steps_empty,
                      service_rate=round(len(rows) / wall, 3), **cnt)
if mode:
    meta["sched"] = sch.summary()
    assert abort[0] or meta["sched"]["deferred_pending"] == 0, "잔여 backlog"
if rows:
    with open(os.path.join(OUT, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, "timeseries.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["t_s", "backlog", "inflight",
                                      "kvroot_b", "disk_free_b"])
    w.writeheader(); w.writerows(ts_rows)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

r2 = [r for r in rows if r["round"] == 2]
tt = sorted(x["ttft_s"] for x in r2) or [0]
p = lambda qv: tt[min(len(tt) - 1, int(round(qv * (len(tt) - 1))))]
mb = max((t["backlog"] for t in ts_rows), default=0)
s = sch.summary() if mode else {}
print(f"RESULT {args.run_id} arm={args.arm} {args.arrival} "
      f"conc={args.conc} rate={args.rate if args.arrival=='poisson' else '-'} "
      f"R2 ttft p50={p(.5):.3f} p95={p(.95):.3f} "
      f"tok/s={out_tok/wall:.1f} wall={wall:.1f}s cpu={cpu:.1f}s "
      f"backlog_max={mb} gaps={gaps} "
      f"forced={s.get('forced_flushes', 0)} "
      f"age_max_ms={round(s.get('max_deferred_age_ms', 0))} "
      f"steps={steps_total}/{steps_empty}빈 "
      f"svc={len(rows)/wall:.3f}req/s "
      f"fence={s.get('fence_wait_n', 0)}회/{round(s.get('fence_wait_ns', 0)/1e9, 1)}s "
      f"guard_skips={cnt['guard_skips']} aborted={abort[0]}", flush=True)
shutil.rmtree(kvroot, ignore_errors=True)
