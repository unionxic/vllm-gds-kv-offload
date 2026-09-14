#!/usr/bin/env python3
"""런 폴더 요약. requests.jsonl(요청별 submit/first/finish)와 events.jsonl로 단계별 지연 분포와 KV IO 합계.
usage: summarize.py RUN_DIR"""
import json, statistics, sys, os
R = sys.argv[1]
reqs = [json.loads(l) for l in open(os.path.join(R, "requests.jsonl"))] if os.path.exists(os.path.join(R, "requests.jsonl")) else []
ev = [json.loads(l) for l in open(os.path.join(R, "events.jsonl"))] if os.path.exists(os.path.join(R, "events.jsonl")) else []
print("phase,count,ttft_p50,ttft_p95,e2e_p50,e2e_p95,e2e_max,phase_wall_s")
for ph in sorted({r["phase"] for r in reqs}, key=lambda p: min(r["submit_mono"] for r in reqs if r["phase"] == p)):
    rows = [r for r in reqs if r["phase"] == ph]
    ttft = sorted(r["first_mono"] - r["submit_mono"] for r in rows if r.get("first_mono")); e2e = sorted(r["finish_mono"] - r["submit_mono"] for r in rows if r.get("finish_mono"))
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
    wall = max(r["finish_mono"] for r in rows) - min(r["submit_mono"] for r in rows)
    print(f"{ph},{len(rows)},{q(ttft,.5):.2f},{q(ttft,.95):.2f},{q(e2e,.5):.2f},{q(e2e,.95):.2f},{max(e2e) if e2e else 0:.2f},{wall:.1f}")
for op in ("r", "w"):
    e = [x for x in ev if x["kind"] == f"kv_{op}_end"]
    if e: print(f"kv_{op}: {len(e)} ops, {sum(x['bytes'] for x in e)/2**30:.2f} GiB, busy {sum(x['dur_ms'] for x in e)/1e3:.1f} s")
