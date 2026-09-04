# Phase 3: A~E 비교군 측정 (1 config × 1 runner = 1 프로세스).
#
#   A recompute      : config=none 의 최초 요청 (모든 config에서도 warm으로 기록)
#   B CPU hit        : config=tiering, reset_prefix_cache() 후
#   C Tiering FS     : config=tiering, reset_prefix_cache(reset_connector=True) 후
#   D expfs+cufile   : config=expfs-cufile, reset_prefix_cache() 후 (SSD→GPU 직행)
#   E expfs+posix    : config=expfs-posix, 동일 control plane, 전송만 bounce
#
# 러너 규율: V1 GDS vs V2 POSIX 직접 비교 금지 — 같은 러너 안에서만 군 비교.
# 측정: TTFT(max_tokens=1 wall), store wall(비동기 store 완료까지), 경로 계수 검증,
#       8-prefix 동시 load 총 시간(동시성 은폐 검사 — HiFC 교훈).
import argparse
import csv
import os
import random
import shutil
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib"))

# ---- 경로 계수 패치 (엔진 생성 전) ----
counters = {"fs_load": 0, "fs_store": [], "cpu_load": 0, "tp_read": 0, "tp_write": []}

import vllm.v1.kv_offload.tiering.fs.manager as fsm
import vllm.v1.kv_offload.cpu.gpu_worker as gw
import expfs

_ol, _os_ = fsm.load_block, fsm.store_block
def _cl(*a, **k):
    counters["fs_load"] += 1
    return _ol(*a, **k)
def _cs(*a, **k):
    t0 = time.perf_counter()
    r = _os_(*a, **k)
    counters["fs_store"].append((t0, time.perf_counter() - t0))
    return r
fsm.load_block, fsm.store_block = _cl, _cs

_sl = gw.CPUOffloadingWorker.submit_load
def _csl(self, *a, **k):
    counters["cpu_load"] += 1
    return _sl(self, *a, **k)
gw.CPUOffloadingWorker.submit_load = _csl

for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport):
    _r, _w = cls.read_chunk, cls.write_chunk
    def mk(fn, key, timed):
        def inner(self, *a, **kw):
            if timed:
                t0 = time.perf_counter()
                r = fn(self, *a, **kw)
                counters[key].append((t0, time.perf_counter() - t0))
                return r
            counters[key] += 1
            return fn(self, *a, **kw)
        return inner
    cls.read_chunk = mk(_r, "tp_read", False)
    cls.write_chunk = mk(_w, "tp_write", True)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def snap():
    return dict(fs_load=counters["fs_load"], cpu_load=counters["cpu_load"],
                tp_read=counters["tp_read"],
                fs_store=len(counters["fs_store"]), tp_write=len(counters["tp_write"]))


def store_wall(store_key, i0):
    ev = counters[store_key][i0:]
    if not ev:
        return 0.0, 0
    return max(t + d for t, d in ev) - min(t for t, _ in ev), len(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    choices=["none", "tiering", "expfs-cufile", "expfs-posix"])
    ap.add_argument("--runner", required=True, choices=["v1", "v2"])
    ap.add_argument("--model", default="facebook/opt-2.7b")
    ap.add_argument("--prefixes", default="1024,2032")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--conc-prefix", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    args = ap.parse_args()

    kvroot = os.path.join(HERE, f"kvroot-{args.config}-{args.runner}-b{args.block_size}")
    shutil.rmtree(kvroot, ignore_errors=True)

    ktc = None
    if args.config == "tiering":
        ktc = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": 8 << 30, "block_size": args.block_size,
                "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]})
    elif args.config.startswith("expfs"):
        ktc = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
                "expfs_root_dir": kvroot,
                "expfs_transport": args.config.split("-")[1],
                "block_size": args.block_size})

    llm = LLM(model=args.model, kv_transfer_config=ktc,
              gpu_memory_utilization=args.gpu_util, max_model_len=2048,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    vocab, bos = tok.vocab_size, tok.bos_token_id or 2
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(20260831)
    rand = lambda n: [rng.randrange(1000, vocab - 1000) for _ in range(n)]

    def gen(tokens_batch):
        t0 = time.perf_counter()
        llm.generate([{"prompt_token_ids": t} for t in tokens_batch], sp, use_tqdm=False)
        return time.perf_counter() - t0

    store_key = "fs_store" if args.config == "tiering" else "tp_write"

    def drain(expected_total, max_nudges=80):
        for _ in range(max_nudges):
            if len(counters[store_key]) >= expected_total:
                n = len(counters[store_key])
                time.sleep(0.5)
                if len(counters[store_key]) == n:
                    return True
            gen([[bos] + rand(8)])
            time.sleep(0.05)
        return len(counters[store_key]) >= expected_total

    rows, model = [], args.model.split("/")[-1]

    def row(**kw):
        rows.append(dict(runner=args.runner, config=args.config, model=model, block=args.block_size, **kw))

    for P in [int(x) for x in args.prefixes.split(",")]:
        for rep in range(args.repeats):
            prefix = [bos] + rand(P - 1)
            n_chunks = (P + 9) // args.block_size

            s0 = snap()
            si = len(counters[store_key])
            t_warm = gen([prefix + rand(8) + [0]])
            drained = True
            if ktc is not None:
                drained = drain(si + n_chunks)
            sw, sn = store_wall(store_key, si) if ktc is not None else (0.0, 0)
            row(prefix=P, rep=rep, arm="A_warm", ttft_s=round(t_warm, 4),
                store_wall_s=round(sw, 3), n_io=sn, path_ok=drained)

            if args.config == "tiering":
                assert llm.reset_prefix_cache()
                s1 = snap()
                t_b = gen([prefix + rand(8) + [1]])
                d1 = snap()
                row(prefix=P, rep=rep, arm="B_cpu_hit", ttft_s=round(t_b, 4),
                    store_wall_s=0, n_io=d1["cpu_load"] - s1["cpu_load"],
                    path_ok=(d1["fs_load"] == s1["fs_load"]
                             and d1["cpu_load"] > s1["cpu_load"]))
                assert llm.reset_prefix_cache(reset_connector=True)
                s2 = snap()
                t_c = gen([prefix + rand(8) + [2]])
                d2 = snap()
                row(prefix=P, rep=rep, arm="C_tiering_fs", ttft_s=round(t_c, 4),
                    store_wall_s=0, n_io=d2["fs_load"] - s2["fs_load"],
                    path_ok=d2["fs_load"] > s2["fs_load"])
            elif ktc is not None:
                assert llm.reset_prefix_cache()
                s1 = snap()
                t_d = gen([prefix + rand(8) + [1]])
                d1 = snap()
                arm = "D_gds" if args.config == "expfs-cufile" else "E_posix"
                row(prefix=P, rep=rep, arm=arm, ttft_s=round(t_d, 4),
                    store_wall_s=0, n_io=d1["tp_read"] - s1["tp_read"],
                    path_ok=d1["tp_read"] > s1["tp_read"])
            else:
                pass  # config=none: A_warm이 곧 순수 A
            print(f"P={P} rep={rep} done", flush=True)

    # ---- 동시성: conc개 prefix 동시 load ----
    Pc = args.conc_prefix
    prefixes = [[bos] + rand(Pc - 1) for _ in range(args.conc)]
    si = len(counters[store_key])
    t_conc_warm = gen([p + rand(8) + [0] for p in prefixes])
    if ktc is not None:
        drain(si + args.conc * ((Pc + 9) // args.block_size))
    if args.config == "tiering":
        assert llm.reset_prefix_cache(reset_connector=True)
    elif ktc is not None:
        assert llm.reset_prefix_cache()
    s1 = snap()
    t_conc = gen([p + rand(8) + [1] for p in prefixes])
    d1 = snap()
    hit_arm = {"none": "A", "tiering": "C", "expfs-cufile": "D", "expfs-posix": "E"}[args.config]
    row(prefix=Pc, rep=-1, arm=f"conc{args.conc}_{hit_arm}",
        ttft_s=round(t_conc, 4), store_wall_s=round(t_conc_warm, 4),
        n_io=(d1["fs_load"] - s1["fs_load"]) + (d1["tp_read"] - s1["tp_read"]),
        path_ok=True)
    print(f"conc done: warm={t_conc_warm:.3f}s hit={t_conc:.3f}s", flush=True)

    hdr = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if hdr:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
