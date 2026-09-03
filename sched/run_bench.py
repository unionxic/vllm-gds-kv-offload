# scheduling 레짐 실험용 축소 윈도 벤치.
#   Bailian coder trace 앞 N요청(기본 150)을 W1과 동일하게 재생. 최소 계측.
#   run별 산출물: sched/runs/<run_id>/{raw.csv, meta.json}
# usage:
#   python run_bench.py --run-id <id> --design D|E [--n 150] [--switch-interval s]
#                       [--write-threads n] [--read-threads n] [--note "..."]
import argparse
import csv
import json
import os
import shutil
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--design", required=True, choices=["D", "E", "C", "S", "P"],
                help="D=cufile E=posix C=tiering S=cufile_staged P=posix_staged")
ap.add_argument("--staging-slots", type=int, default=6,
                help="staged ring 슬롯 수 (V1 b256은 chunk 80MiB라 2 권장)")
ap.add_argument("--staging-policy", default="block",
                choices=["block", "skip", "cpu_fallback", "value"])
ap.add_argument("--value-mode", default="value_density",
                choices=["random_skip", "seen_twice", "value_density", "oracle"])
ap.add_argument("--cpu-fallback-slots", type=int, default=0)
ap.add_argument("--max-ob-bytes", type=int, default=None,
                help="outstanding write bytes 상한")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--switch-interval", type=float, default=None, help="초 단위 (예: 0.0005)")
ap.add_argument("--write-threads", type=int, default=8)
ap.add_argument("--read-threads", type=int, default=8)
ap.add_argument("--policy", default="baseline",
                choices=["baseline", "read_priority", "deferred_store", "phase_sep"])
ap.add_argument("--note", default="")
args = ap.parse_args()

# switch interval은 어떤 스레드도 생기기 전에 적용
if args.switch_interval is not None:
    sys.setswitchinterval(args.switch_interval)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "phase2"))
sys.path.insert(0, os.path.join(HERE, "..", "phase1"))
import snapshot  # noqa: E402

OUT = os.path.join(HERE, "runs", args.run_id)
os.makedirs(OUT, exist_ok=True)
meta = {"args": vars(args), "start": snapshot.collect("start")}

import random  # noqa: E402

import expfs  # noqa: E402
import policies  # noqa: E402

if args.design == "C":
    assert args.policy == "baseline", "C는 기존 tiering 경로 그대로 — policy 불가"
else:
    policies.install(args.policy)
PROF = os.environ.get("PROF_INSTRUMENT") == "1"
if PROF:
    import prof_instrument
    prof_instrument.install()
if os.environ.get("ISOLATE"):
    assert args.design == "D", "원인 분해는 D 계열에서만"
    import prof_isolate
    prof_isolate.install()
    meta["isolate"] = {"mode": prof_isolate.MODE, "write_ms": prof_isolate.WRITE_MS}
if os.environ.get("FIX"):
    import prof_fix
    prof_fix.install()
    meta["fix"] = prof_fix.MODE
worker_holder: list = []

cnt = {"tp_read_n": 0, "tp_read_b": 0, "tp_write_n": 0, "tp_write_b": 0,
       "matched": 0, "guard_skips": 0}
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

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched  # noqa: E402

_gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
def _gmw(self, request, num_computed_tokens):
    r = _gm(self, request, num_computed_tokens)
    if r[0]:
        cnt["matched"] += r[0]
    return r
osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

# C(tiering) 대조군용: fs 티어 계수 + 종료 race 가드(v0.26.0 workaround)
import vllm.v1.kv_offload.tiering.fs.manager as fsm  # noqa: E402
_ol, _os_ = fsm.load_block, fsm.store_block
def _cl(path, view, off, bs):
    cnt["tp_read_n"] += 1
    cnt["tp_read_b"] += bs
    return _ol(path, view, off, bs)
def _cs(path, buf, off, bs):
    cnt["tp_write_n"] += 1
    cnt["tp_write_b"] += bs
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

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

kvroot = os.path.join(HERE, f"kvroot-{args.run_id}")
shutil.rmtree(kvroot, ignore_errors=True)
if args.design == "C":
    extra = {"spec_name": "TieringOffloadingSpec",
             "cpu_bytes_to_use": 8 << 30, "block_size": 16,
             "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]}
else:
    tname = {"D": "cufile", "E": "posix", "S": "cufile_staged",
             "P": "posix_staged"}[args.design]
    extra = {"spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
             "expfs_root_dir": kvroot,
             "expfs_transport": tname,
             "expfs_read_threads": args.read_threads,
             "expfs_write_threads": args.write_threads,
             "expfs_staging_slots": args.staging_slots,
             "expfs_staging_policy": args.staging_policy,
             "expfs_cpu_fallback_slots": args.cpu_fallback_slots,
             "expfs_max_outstanding_write_bytes": args.max_ob_bytes,
             "block_size": args.block}
llm = LLM(model="facebook/opt-2.7b",
          kv_transfer_config=KVTransferConfig(
              kv_connector="OffloadingConnector", kv_role="kv_both",
              kv_connector_extra_config=extra),
          gpu_memory_utilization=0.7, max_model_len=2048, enforce_eager=True)

if args.staging_policy == "value":
    sys.path.insert(0, os.path.join(HERE, "..", "admission"))
    import value_admission  # noqa: E402
    _vw = expfs.LAST_WORKER
    value_admission.install(_vw.transport, args.value_mode)

tok = llm.get_tokenizer()
vocab, bos = tok.vocab_size, tok.bos_token_id or 2
sp = SamplingParams(max_tokens=1, temperature=0.0)

reqs = []
with open(os.path.join(HERE, "..", "w1", "trace", "qwen_coder.jsonl")) as f:
    for line in f:
        reqs.append(json.loads(line))
        if len(reqs) >= args.n:
            break

bcache = {}
def block_tokens(h):
    rng = random.Random(0xB10C0000 + h)
    return [rng.randrange(1000, vocab - 1000) for _ in range(16)]

rows = []
t0_all, c0_all = time.perf_counter(), time.process_time()
for i, r in enumerate(reqs):
    toks = []
    for h in r["hash_ids"][:127]:
        if h not in bcache:
            bcache[h] = block_tokens(h)
        toks.extend(bcache[h])
    tail = min(r["input_length"] - 16 * len(r["hash_ids"]), 2040 - len(toks))
    if tail > 0:
        rng = random.Random(0x7A11 + r["chat_id"])
        toks.extend(rng.randrange(1000, vocab - 1000) for _ in range(tail))
    before = dict(cnt)
    t0 = time.perf_counter()
    llm.generate([{"prompt_token_ids": toks}], sp, use_tqdm=False)
    ttft = time.perf_counter() - t0
    d = {k: cnt[k] - before[k] for k in cnt}
    rows.append(dict(i=i, ttft_s=round(ttft, 4), matched=d["matched"],
                     read_ios=d["tp_read_n"], read_b=d["tp_read_b"],
                     write_ios=d["tp_write_n"], write_b=d["tp_write_b"]))
    policies.on_request_gap(worker_holder)

_w = getattr(expfs, "LAST_WORKER", None)
if _w is not None and hasattr(_w.transport, "flush"):
    _w.transport.flush()  # staged 잔여 비동기 쓰기를 wall/CPU에 포함 (생략 아님 증명)
    meta["staged_write_errors"] = getattr(_w.transport, "write_errors", 0)
    meta["staged_stats"] = dict(getattr(_w.transport, "stats", {}))
    if args.staging_policy == "value":
        meta["value_mode"] = args.value_mode
wall = time.perf_counter() - t0_all
cpu = time.process_time() - c0_all
meta["io_threads_schedstat"] = snapshot.thread_schedstat()  # 스레드 생존 중 캡처
try:
    llm.llm_engine.engine_core.shutdown()
except Exception as e:
    print("shutdown:", e)

meta["end"] = snapshot.collect("end")
meta["policy_state"] = policies.summary()
if PROF:
    meta["prof"] = prof_instrument.summary()
    print("PROF", json.dumps(meta["prof"], indent=1), flush=True)
meta["totals"] = dict(wall_s=round(wall, 2), cpu_s=round(cpu, 2), **cnt)
with open(os.path.join(OUT, "raw.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

tt = sorted(x["ttft_s"] for x in rows)
p = lambda q: tt[min(len(tt) - 1, int(round(q * (len(tt) - 1))))]
ps = policies.summary()
print(f"RESULT {args.run_id} {args.design} pol={args.policy} n={len(rows)} "
      f"p50={p(.5):.3f} p95={p(.95):.3f} p99={p(.99):.3f} "
      f"wall={wall:.1f}s cpu={cpu:.1f}s matched={cnt['matched']} "
      f"maxOutW={ps['max_outstanding_store']} maxOutR={ps['max_outstanding_load']} "
      f"forced={ps['forced_flushes']} gap={ps['gap_flushes']}", flush=True)
shutil.rmtree(kvroot, ignore_errors=True)
if PROF:
    # py-spy 자식 모드에서 인터프리터 teardown이 교착 → 산출물 기록 후 즉시 종료
    sys.stdout.flush()
    os._exit(0)
