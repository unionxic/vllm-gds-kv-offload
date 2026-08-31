# Phase 0.5: SSD-backed KV의 정당성 게이트 벤치.
#
# 한 프로세스 안에서 prefix별로 3개 arm을 순차 측정:
#   A (recompute): 최초 요청 = GPU/CPU/FS 전부 miss → prefill 재계산
#   B (CPU hit):   reset_prefix_cache()                → GPU만 비움, CPU primary hit
#   C (SSD hit):   reset_prefix_cache(reset_connector=True) → GPU+CPU 비움, FS 보존 → SSD hit
#     (fs 티어 load는 O_DIRECT라 페이지캐시 우회 — 같은 프로세스라도 실제 디스크 읽기)
#
# 경로 검증: 엔진 생성 전 몽키패치 계수
#   fs load_block/store_block (호출수·바이트·IO시간·wall) + CPUOffloadingWorker.submit_load
#   기대: A: fs_load=0, cpu_load=0 / B: fs_load=0, cpu_load>0 / C: fs_load>0
#
# TTFT 프록시: max_tokens=1 요청의 wall time (모든 arm 동일 조건).
# 출력: CSV (phase05/results.csv에 append)
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- 계측 패치 (vLLM 엔진 생성 전) ----
import vllm.v1.kv_offload.tiering.fs.manager as fsm
import vllm.v1.kv_offload.cpu.gpu_worker as gw

fs_events = {"load": [], "store": []}  # (t_start, dur, nbytes)
cpu_counters = {"load_jobs": 0}

_orig_load, _orig_store = fsm.load_block, fsm.store_block


def _timed_load(source_path, view, offset, block_size):
    t0 = time.perf_counter()
    r = _orig_load(source_path, view, offset, block_size)
    fs_events["load"].append((t0, time.perf_counter() - t0, block_size))
    return r


def _timed_store(dest_path, buffer, offset, block_size):
    t0 = time.perf_counter()
    r = _orig_store(dest_path, buffer, offset, block_size)
    fs_events["store"].append((t0, time.perf_counter() - t0, block_size))
    return r


fsm.load_block, fsm.store_block = _timed_load, _timed_store

_orig_submit_load = gw.CPUOffloadingWorker.submit_load


def _counting_submit_load(self, *a, **kw):
    cpu_counters["load_jobs"] += 1
    return _orig_submit_load(self, *a, **kw)


gw.CPUOffloadingWorker.submit_load = _counting_submit_load

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def snapshot():
    return {
        "fs_load_calls": len(fs_events["load"]),
        "fs_store_calls": len(fs_events["store"]),
        "cpu_load_jobs": cpu_counters["load_jobs"],
    }


def delta(after, before):
    return {k: after[k] - before[k] for k in after}


def window(events, i0, i1):
    """events[i0:i1]의 (총 IO시간, wall시간, 총 바이트)"""
    ev = events[i0:i1]
    if not ev:
        return 0.0, 0.0, 0
    io = sum(d for _, d, _ in ev)
    wall = max(t + d for t, d, _ in ev) - min(t for t, _, _ in ev)
    nbytes = sum(b for _, _, b in ev)
    return io, wall, nbytes


# FS store 캐스케이드(CPU→FS enqueue)는 스케줄러 스텝(update_connector_output)에서만
# 진행된다. max_tokens=1 요청은 종료 후 스텝이 없어 store가 다음 generate까지 정체됨.
# → 16토큰 미만(=chunk 미생성) 더미 요청으로 스텝을 공급해 flush를 유도한다.
def nudge_drain(llm, sp, make_tiny_prompt, expected_total, max_nudges=60):
    for _ in range(max_nudges):
        if len(fs_events["store"]) >= expected_total:
            n = len(fs_events["store"])
            time.sleep(0.5)  # 진행 중인 스레드풀 쓰기 안정화
            if len(fs_events["store"]) == n:
                return True
        llm.generate([make_tiny_prompt()], sp, use_tqdm=False)
        time.sleep(0.05)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--prefixes", default="1024", help="comma list of prefix token counts")
    ap.add_argument("--repeats", type=int, default=3, help="독립 prefix 반복 수(=arm당 표본)")
    ap.add_argument("--cpu-gib", type=float, default=4.0)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=2304)
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    kvroot = os.path.join(HERE, "kvroot-bench")
    shutil.rmtree(kvroot, ignore_errors=True)

    llm = LLM(
        model=args.model,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "TieringOffloadingSpec",
                "cpu_bytes_to_use": int(args.cpu_gib * (1 << 30)),
                "block_size": 16,
                "secondary_tiers": [{"type": "fs", "root_dir": kvroot}],
            },
        ),
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_len,
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    vocab = tok.vocab_size
    bos = tok.bos_token_id if tok.bos_token_id is not None else 2
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rng = random.Random(20260831)

    def rand_tokens(n):
        return [rng.randrange(1000, vocab - 1000) for _ in range(n)]

    def timed_gen(token_ids):
        before = snapshot()
        li, si = len(fs_events["load"]), len(fs_events["store"])
        t0 = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": token_ids}], sp, use_tqdm=False)
        ttft = time.perf_counter() - t0
        d = delta(snapshot(), before)
        load_io, load_wall, load_bytes = window(fs_events["load"], li, len(fs_events["load"]))
        return ttft, d, (load_io, load_wall, load_bytes), (si, out)

    rows = []
    for P in [int(x) for x in args.prefixes.split(",")]:
        for rep in range(args.repeats):
            prefix = [bos] + rand_tokens(P - 1)

            def prompt_for(arm_idx):
                return prefix + rand_tokens(8) + [arm_idx]

            # ---- arm A: 전층 miss (이 prefix의 최초 요청) ----
            ttft_a, d_a, _, (si_a, _) = timed_gen(prompt_for(0))
            n_chunks = (P + 9) // 16
            drained = nudge_drain(
                llm, sp,
                make_tiny_prompt=lambda: {"prompt_token_ids": [bos] + rand_tokens(8)},
                expected_total=si_a + n_chunks,
            )
            st_io, st_wall, st_bytes = window(fs_events["store"], si_a, len(fs_events["store"]))
            n_stored = len(fs_events["store"]) - si_a  # nudge로 flush된 것 포함

            # ---- arm B: GPU만 리셋 → CPU hit ----
            assert llm.reset_prefix_cache(), "GPU prefix cache reset failed"
            ttft_b, d_b, _, _ = timed_gen(prompt_for(1))

            # ---- arm C: GPU+CPU 리셋 (FS 보존) → SSD hit ----
            assert llm.reset_prefix_cache(reset_connector=True), "connector reset failed"
            ttft_c, d_c, (ld_io, ld_wall, ld_bytes), _ = timed_gen(prompt_for(2))

            checks = {
                "A": d_a["fs_load_calls"] == 0 and d_a["cpu_load_jobs"] == 0,
                "B": d_b["fs_load_calls"] == 0 and d_b["cpu_load_jobs"] > 0,
                "C": d_c["fs_load_calls"] > 0,
            }
            base = dict(model=args.model, prefix_tokens=P, rep=rep, tag=args.tag,
                        store_drained=drained, chunk_files=n_stored)
            rows.append({**base, "arm": "A_recompute", "ttft_s": round(ttft_a, 4),
                         "fs_load_calls": d_a["fs_load_calls"], "cpu_load_jobs": d_a["cpu_load_jobs"],
                         "io_s": round(st_io, 4), "io_wall_s": round(st_wall, 4),
                         "io_bytes": st_bytes, "path_check": checks["A"]})
            rows.append({**base, "arm": "B_cpu_hit", "ttft_s": round(ttft_b, 4),
                         "fs_load_calls": d_b["fs_load_calls"], "cpu_load_jobs": d_b["cpu_load_jobs"],
                         "io_s": 0, "io_wall_s": 0, "io_bytes": 0, "path_check": checks["B"]})
            rows.append({**base, "arm": "C_ssd_hit", "ttft_s": round(ttft_c, 4),
                         "fs_load_calls": d_c["fs_load_calls"], "cpu_load_jobs": d_c["cpu_load_jobs"],
                         "io_s": round(ld_io, 4), "io_wall_s": round(ld_wall, 4),
                         "io_bytes": ld_bytes, "path_check": checks["C"]})
            print(f"P={P} rep={rep}: A={ttft_a:.4f}s B={ttft_b:.4f}s C={ttft_c:.4f}s "
                  f"checks={checks} store_wall={st_wall:.3f}s load_wall={ld_wall:.3f}s")

    write_header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {args.out}")

    # 요약: H_min = ceil(T_store / (T_A - T_C))
    import math
    from collections import defaultdict

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["prefix_tokens"]][r["arm"]].append(r["ttft_s"])
        if r["arm"] == "A_recompute":
            agg[r["prefix_tokens"]]["store_wall"].append(r["io_wall_s"])
    print("\n== summary (mean) ==")
    for P, d in sorted(agg.items()):
        a = sum(d["A_recompute"]) / len(d["A_recompute"])
        b = sum(d["B_cpu_hit"]) / len(d["B_cpu_hit"])
        c = sum(d["C_ssd_hit"]) / len(d["C_ssd_hit"])
        stw = sum(d["store_wall"]) / len(d["store_wall"])
        gain = a - c
        hmin = math.ceil(stw / gain) if gain > 0 else None
        print(f"P={P}: A={a:.4f} B={b:.4f} C={c:.4f} | C<A: {c < a} | "
              f"store_wall={stw:.3f}s | H_min={hmin}")


if __name__ == "__main__":
    main()
