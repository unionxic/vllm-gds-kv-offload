#!/usr/bin/env python3
"""results/model-host-baseline의 none/cufile 쌍을 모델 × host 표로 요약."""
import glob, json, os, re, sys
GIB = 2**30
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "model-host-baseline")
def stats(r):
    """배치 = n_tok이 다시 작아지는 지점으로 구분. 배치의 prefill = 출력 0인 step이 있으면 그것, 없으면 배치 안 최장 step.
    decode = 나머지 step의 중앙값. 대기 = 앞 배치 끝에서 이 배치 첫 step까지."""
    real = [s for s in r["steps"] if s["t1"] - s["t0"] > 0.3]
    groups, cur, prev_tok = [], [], -1
    for s in real:
        if cur and (s["n_out"] == 0 or s["n_tok"] <= prev_tok):
            groups.append(cur); cur = []
        cur.append(s); prev_tok = s["n_tok"]
    if cur: groups.append(cur)
    pre, dec, waits, prev_end = [], [], [], 0.0
    for g in groups:
        z = [s for s in g if s["n_out"] == 0]
        p = z[0] if z else max(g, key=lambda s: s["t1"] - s["t0"])
        pre.append(p["t1"] - p["t0"]); dec += [s["t1"] - s["t0"] for s in g if s is not p]
        waits.append(g[0]["t0"] - prev_end); prev_end = g[-1]["t1"]
    dec.sort()
    return dict(wall=r["wall_s"], pre=sum(pre) / len(pre), dec=dec[len(dec) // 2], wait=sum(waits) / len(waits), waits=waits, n_batches=len(groups))
print("model host | CPU/SSD tier | fwd meas/model | prefill none/hit | wait/batch | wall none/hit | store extra | 2-round total none/store+hit")
for nf in sorted(glob.glob(os.path.join(root, "*-none.json"))):
    m = re.match(r"(.+)-(h|ram)([\d.]+)-none\.json", os.path.basename(nf)); cf = nf.replace("-none.json", "-cufile.json")
    if not m:
        continue
    a = json.load(open(nf)); t = a["tiers"]; an = stats(a["rounds"][1])
    model = t["host_tier_gib"] * GIB / 12.3e9 + t["ssd_tier_gib"] * GIB / 3.44e9
    label = m.group(3) if m.group(2) == "h" else "RAM" + m.group(3)
    line = f"{m.group(1)} {label:>6} | {t['n_modules']-t['n_ssd']:2}/{t['n_ssd']:2} {t['host_tier_gib']:6.1f}/{t['ssd_tier_gib']:6.1f} GiB | {an['dec']:5.2f}/{model:5.2f} | {an['pre']:5.2f}"
    if os.path.exists(cf):
        b = json.load(open(cf)); bs = stats(b["rounds"][0]); bh = stats(b["rounds"][1])
        tn = sum(r["wall_s"] for r in a["rounds"]); tc = sum(r["wall_s"] for r in b["rounds"])
        line += (f"/{bh['pre']:5.2f} | {bh['wait']:4.2f} | {an['wall']:6.1f}/{bh['wall']:6.1f} ({(bh['wall']/an['wall']-1)*100:+5.1f}%) | {bs['wall']-an['wall']:+5.1f}"
                 f" | {tn:6.1f}/{tc:6.1f} ({(tc/tn-1)*100:+5.1f}%)")
    print(line)
