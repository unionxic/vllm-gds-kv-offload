# open-loop/동시성 스위프 집계: raw.csv + timeseries.csv + meta.json →
#   R2 TTFT 분위수, backlog 증가율(최소자승 기울기), SSD 증가 속도, starvation 지표.
import csv
import glob
import json
import os
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "results", "leval-openloop", "raw")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0.0


def slope(pairs):
    n = len(pairs)
    if n < 2:
        return 0.0
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    d = n * sxx - sx * sx
    return (n * sxy - sx * sy) / d if d else 0.0


pat = sys.argv[1] if len(sys.argv) > 1 else "ol-*"
rows = []
for d in sorted(glob.glob(os.path.join(BASE, pat))):
    mf = os.path.join(d, "meta.json")
    if not os.path.exists(mf):
        continue
    meta = json.load(open(mf))
    a = meta["args"]
    t = meta["totals"]
    s = meta.get("sched", {})
    raw = list(csv.DictReader(open(os.path.join(d, "raw.csv"))))
    ts = list(csv.DictReader(open(os.path.join(d, "timeseries.csv"))))
    r2 = [float(r["ttft_s"]) for r in raw if r["round"] == "2"]
    r2e = [float(r["e2e_s"]) for r in raw if r["round"] == "2"]
    # backlog 기울기: 정상 상태 구간(초반 엔진 웜업 15s 제외)
    bl = [(float(r["t_s"]), int(r["backlog"])) for r in ts if float(r["t_s"]) > 15]
    kv = [(float(r["t_s"]), int(r["kvroot_b"])) for r in ts if int(r["kvroot_b"]) > 0]
    rows.append(dict(
        run=os.path.basename(d), arm=a["arm"], arrival=a["arrival"],
        conc=a["conc"], rate=a["rate"] if a["arrival"] == "poisson" else "",
        r2_ttft_p50=round(pct(r2, .5), 3), r2_ttft_p95=round(pct(r2, .95), 3),
        r2_e2e_p95=round(pct(r2e, .95), 3),
        tok_s=t["out_tok_per_s"], wall=t["wall_s"], cpu=t["cpu_s"],
        backlog_max=max((b for _, b in bl), default=0),
        backlog_slope_per_min=round(slope(bl) * 60, 3),
        kv_gib_per_min=round(slope(kv) * 60 / 2**30, 2),
        gaps=t["gaps"], forced=s.get("forced_flushes", 0),
        deferred=s.get("deferred_writes", 0),
        age_max_ms=round(s.get("max_deferred_age_ms", 0)),
        guard_skips=t["guard_skips"], aborted=t["aborted"]))

cols = list(rows[0].keys())
wid = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
print(" | ".join(c.ljust(wid[c]) for c in cols))
for r in rows:
    print(" | ".join(str(r[c]).ljust(wid[c]) for c in cols))
