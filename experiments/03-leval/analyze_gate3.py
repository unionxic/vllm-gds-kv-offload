# Gate 3 판정: arm별 반복 통계(R1/R2 분리, CV), D1 vs E1 paired, 성공 기준 체크.
import csv
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "..", "results", "leval", "raw")
med = statistics.median


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


runs = []
for mp in sorted(glob.glob(os.path.join(RAW, "g3-*", "meta.json"))):
    m = json.load(open(mp))
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(mp), "raw.csv"))))
    out = dict(id=m["args"]["run_id"], arm=m["args"]["arm"],
               wall=m["totals"]["wall_s"], cpu=m["totals"]["cpu_s"],
               matched=m["totals"]["matched"],
               store_ios=m["totals"].get("tp_write_n", 0) + m["totals"].get("fs_store_n", 0))
    for rnd in ("1", "2"):
        tt = [float(r["ttft_s"]) for r in rows if r["round"] == rnd]
        out[f"r{rnd}_p50"] = med(tt)
        out[f"r{rnd}_p95"] = pct(tt, 95)
        if rnd == "2":
            out["fs_hits"] = sum(1 for r in rows if r["round"] == "2"
                                 and int(r["ssd_read_ios"]) > 0)
    # IO 스레드 runqueue 대기 % (shutdown 전 캡처)
    ss = m.get("io_threads_schedstat", {})
    io = [v for v in ss.values() if v["name"].startswith("expfs")]
    if io:
        c = sum(v["cpu_ns"] for v in io)
        w = sum(v["wait_ns"] for v in io)
        out["io_runq_pct"] = round(w / (c + w) * 100, 1) if c + w else None
    runs.append(out)

print(f"{'id':14s} {'R1p50':>6s} {'R1p95':>6s} {'R2p50':>6s} {'R2p95':>6s} "
      f"{'hits':>5s} {'cpu':>6s} {'runq%':>6s}")
for r in runs:
    print(f"{r['id']:14s} {r['r1_p50']:6.3f} {r['r1_p95']:6.3f} "
          f"{r['r2_p50']:6.3f} {r['r2_p95']:6.3f} {r.get('fs_hits','-'):>5} "
          f"{r['cpu']:6.1f} {str(r.get('io_runq_pct','-')):>6s}")

from collections import defaultdict
g = defaultdict(list)
for r in runs:
    g[r["arm"]].append(r)
print("\narm별 반복 통계:")
for arm in sorted(g):
    v = g[arm]
    for key in ("r2_p50", "r2_p95", "r1_p95"):
        xs = [x[key] for x in v]
        m_ = statistics.mean(xs)
        cv = statistics.pstdev(xs) / m_ if m_ else 0
        print(f"  {arm:3s} {key}: n={len(xs)} mean={m_:.3f} cv={cv:.03f} "
              f"min={min(xs):.3f} max={max(xs):.3f}")

# 판정: D1 vs E1 (성공 기준: p95 또는 처리량 10%+ 우세, 5회 중 4회 같은 방향)
if "D1" in g and "E1" in g:
    d = sorted(g["D1"], key=lambda x: x["id"])
    e = sorted(g["E1"], key=lambda x: x["id"])
    wins = sum(1 for i in range(min(len(d), len(e)))
               if d[i]["r2_p95"] < e[i]["r2_p95"] * 0.9)
    print(f"\nD1 vs E1 (R2 p95, rep 순서 무관 min-pair): "
          f"10%+ 우세 {wins}/{min(len(d), len(e))}회")
    print("hit·store 동일성:",
          all(x["matched"] == d[0]["matched"] for x in d + e),
          all(x["store_ios"] == d[0]["store_ios"] for x in d + e))
