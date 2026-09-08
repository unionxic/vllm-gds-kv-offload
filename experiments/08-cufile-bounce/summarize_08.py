"""08 결과 표. usage: python summarize_08.py"""
import json, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "../../results/cufile-bounce/*.json"))):
    d = json.load(open(f)); a = d.get("args", {}); tag = os.path.basename(f)[:-5]
    pb = next((int(s[2:]) for s in tag.split("-") if s.startswith("pb")), 1024)
    rows.append((pb, a.get("prefetch_step"), a.get("io_threads"), tag,
                 d.get("prefill_batch_s"), d.get("decode_step_s"), d.get("cpu_s"),
                 (d.get("nvfs_delta") or {}).get("GPU 0000.Cache_MiB")))
# 06 기준(같은 구성, ring 없음)
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "../../results/weight-offload/opt66b/c-h0.3-r[123].json"))):
    d = json.load(open(f))
    rows.append((1024, 1, 4, "06 " + os.path.basename(f)[:-5], d.get("prefill_batch_s"), d.get("decode_step_s"), d.get("cpu_s"), None))
hdr = ("chunk", "step", "thr", "tag", "prefill_s", "decode_s", "cpu_s", "cacheMiB")
w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
print("-+-".join("-" * x for x in w))
for r in sorted(rows, key=lambda r: (r[0], r[1] or 0, r[3])):
    print(" | ".join(str(x if x is not None else "").ljust(w[i]) for i, x in enumerate(r)))
