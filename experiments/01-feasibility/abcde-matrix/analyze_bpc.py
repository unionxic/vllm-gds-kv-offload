# 재대결 결과 요약: (runner, block) 그룹별 C/D/E 비교 + 기존 b16 대비
import csv
import statistics
from collections import defaultdict

med = statistics.median


def load(path):
    t = defaultdict(list)
    conc = {}
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return t, conc
    for r in rows:
        blk = int(r.get("block", 16))
        if int(r["rep"]) >= 0:
            t[(r["runner"], blk, int(r["prefix"]), r["arm"])].append(float(r["ttft_s"]))
            if r["arm"] == "A_warm" and float(r["store_wall_s"]) > 0:
                t[(r["runner"], blk, int(r["prefix"]), f"storewall_{r['config']}")].append(
                    float(r["store_wall_s"]))
        else:
            conc[(r["runner"], blk, r["config"])] = float(r["ttft_s"])
    return t, conc


t_old, conc_old = load("results.csv")       # block=16 원판
t_new, conc_new = load("results_bpc.csv")   # block=64/256
for k, v in t_new.items():
    t_old[k].extend(v)
conc_old.update(conc_new)
t, conc = t_old, conc_old

print(f"{'runner':6s} {'blk':>4s} {'P':>5s} | {'C tier':>8s} {'D gds':>8s} {'E posix':>8s} | C/D    E/D")
for runner in ("v1", "v2"):
    for blk in (16, 64, 256):
        for P in (1024, 2032):
            def g(arm):
                v = t.get((runner, blk, P, arm))
                return med(v) if v else None
            c, d, e = g("C_tiering_fs"), g("D_gds"), g("E_posix")
            if not any((c, d, e)):
                continue
            f = lambda x: f"{x:8.4f}" if x else "       -"
            r1 = f"{c/d:5.2f}x" if (c and d) else "  -  "
            r2 = f"{e/d:5.2f}x" if (e and d) else "  -  "
            print(f"{runner:6s} {blk:4d} {P:5d} | {f(c)} {f(d)} {f(e)} | {r1} {r2}")
print("\nstore wall 중앙값 (P=2032):")
for runner in ("v1", "v2"):
    for blk in (16, 64, 256):
        parts = []
        for cfg in ("tiering", "expfs-cufile", "expfs-posix"):
            v = t.get((runner, blk, 2032, f"storewall_{cfg}"))
            parts.append(f"{cfg}={med(v):.3f}" if v else f"{cfg}=-")
        if any("=0." in p or "=1." in p or "=2." in p for p in parts):
            print(f"  {runner} b{blk}: " + "  ".join(parts))
print("\n동시 8-load hit 총시간:")
for (runner, blk, cfg), v in sorted(conc.items()):
    print(f"  {runner} b{blk:<3d} {cfg:14s}: {v:.3f}s")
