#!/usr/bin/env python3
"""results/model-host-baseline의 none/cufile 쌍을 모델 × host 표로 요약."""
import glob, json, os, re, sys
GIB = 2**30
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "model-host-baseline")
def stats(r):
    real = [s for s in r["steps"] if s["t1"] - s["t0"] > 0.5]
    pre = [s["t1"] - s["t0"] for s in real if s["n_out"] == 0]
    dec = sorted(s["t1"] - s["t0"] for s in real if s["n_out"] > 0)
    gaps = sum(max(0, real[i + 1]["t0"] - real[i]["t1"]) for i in range(len(real) - 1)) + real[0]["t0"]
    return dict(wall=r["wall_s"], pre=sum(pre) / len(pre), dec=dec[len(dec) // 2], wait=gaps / max(1, len(pre)))
print("model host | CPU/SSD tier | fwd meas/model | prefill none/hit | wait/batch | wall none/hit | store extra")
for nf in sorted(glob.glob(os.path.join(root, "*-none.json"))):
    m = re.match(r"(.+)-h([\d.]+)-none\.json", os.path.basename(nf)); cf = nf.replace("-none.json", "-cufile.json")
    a = json.load(open(nf)); t = a["tiers"]; an = stats(a["rounds"][1])
    model = t["host_tier_gib"] * GIB / 12.3e9 + t["ssd_tier_gib"] * GIB / 3.44e9
    line = f"{m.group(1)} {m.group(2):>3} | {t['n_modules']-t['n_ssd']:2}/{t['n_ssd']:2} {t['host_tier_gib']:6.1f}/{t['ssd_tier_gib']:6.1f} GiB | {an['dec']:5.2f}/{model:5.2f} | {an['pre']:5.2f}"
    if os.path.exists(cf):
        b = json.load(open(cf)); bs = stats(b["rounds"][0]); bh = stats(b["rounds"][1])
        line += f"/{bh['pre']:5.2f} | {bh['wait']:4.2f} | {an['wall']:6.1f}/{bh['wall']:6.1f} ({(bh['wall']/an['wall']-1)*100:+5.1f}%) | {bs['wall']-an['wall']:+5.1f}"
    print(line)
