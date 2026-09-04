# step1 정책 실험 요약: 정책×설계별 반복 통계(p95 평균·CV), 레짐 분류, D/E paired 비교
import csv
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

FAST_P95, SLOW_P95 = 2.5, 4.0  # 창 0~149 기준 (plan.md)


def regime(p95):
    return "FAST" if p95 <= FAST_P95 else ("SLOW" if p95 >= SLOW_P95 else "MID")


runs = []
for meta_path in sorted(glob.glob(os.path.join(HERE, "runs", "*", "meta.json"))):
    m = json.load(open(meta_path))
    rid = m["args"]["run_id"]
    raw = os.path.join(os.path.dirname(meta_path), "raw.csv")
    tt = sorted(float(r["ttft_s"]) for r in csv.DictReader(open(raw)))
    p = lambda q: tt[min(len(tt) - 1, int(round(q * (len(tt) - 1))))]
    runs.append(dict(
        id=rid, design=m["args"]["design"], policy=m["args"].get("policy", "baseline"),
        note=m["args"].get("note", ""), n=len(tt),
        p50=p(.5), p95=p(.95), p99=p(.99),
        wall=m["totals"]["wall_s"], cpu=m["totals"]["cpu_s"],
        matched=m["totals"]["matched"],
        maxW=m.get("policy_state", {}).get("max_outstanding_store"),
        maxR=m.get("policy_state", {}).get("max_outstanding_load"),
        forced=m.get("policy_state", {}).get("forced_flushes"),
        regime=regime(p(.95)),
    ))

print(f"{'run_id':26s} {'des':3s} {'policy':14s} {'p50':>6s} {'p95':>6s} {'p99':>6s} "
      f"{'wall':>6s} {'cpu':>7s} {'maxW':>4s} {'reg':>4s}")
for r in runs:
    print(f"{r['id']:26s} {r['design']:3s} {r['policy']:14s} {r['p50']:6.3f} "
          f"{r['p95']:6.3f} {r['p99']:6.3f} {r['wall']:6.1f} {r['cpu']:7.1f} "
          f"{str(r['maxW']):>4s} {r['regime']:>4s}")

print("\n정책×설계 반복 통계 (s1- 런만):")
from collections import defaultdict
g = defaultdict(list)
for r in runs:
    if r["id"].startswith("s1-"):
        g[(r["policy"], r["design"])].append(r)
for k in sorted(g):
    v = g[k]
    p95s = [x["p95"] for x in v]
    m = statistics.mean(p95s)
    cv = (statistics.pstdev(p95s) / m) if m else 0
    regs = [x["regime"] for x in v]
    print(f"  {k[0]:14s} {k[1]}: n={len(v)} p95 mean={m:.3f} cv={cv:.2f} "
          f"min={min(p95s):.3f} max={max(p95s):.3f} regimes={regs}")
