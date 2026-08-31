# Phase 3 결과 요약: 러너별 arm 비교 (러너 간 교차 비교 금지 규율 준수)
import csv
import statistics
from collections import defaultdict

rows = list(csv.DictReader(open("results.csv")))
bad = [r for r in rows if r["path_ok"] != "True"]
print(f"path_ok failures: {len(bad)}")
for r in bad[:5]:
    print("  ", r)

# TTFT: (runner, prefix, arm) -> [ttft]
ttft = defaultdict(list)
store = defaultdict(list)
for r in rows:
    if int(r["rep"]) >= 0:
        ttft[(r["runner"], int(r["prefix"]), r["arm"])].append(float(r["ttft_s"]))
        if r["arm"] == "A_warm" and float(r["store_wall_s"]) > 0:
            store[(r["runner"], int(r["prefix"]), r["config"])].append(
                float(r["store_wall_s"]))

med = lambda v: statistics.median(v)

for runner in ("v2", "v1"):
    print(f"\n===== runner {runner} =====")
    print(f"{'P':>5s} | {'A(none)':>8s} {'B cpu':>8s} {'C tier':>8s} {'D gds':>8s} {'E posix':>8s} | C/D 배율")
    for P in (1024, 2032):
        def g(arm):
            v = ttft.get((runner, P, arm))
            return med(v) if v else None
        a = g("A_warm")  # config=none의 A_warm... 주의: A_warm은 모든 config에 있음
        # config=none의 A만 뽑기
        a_none = [float(r["ttft_s"]) for r in rows
                  if r["runner"] == runner and int(r["prefix"]) == P
                  and r["arm"] == "A_warm" and r["config"] == "none" and int(r["rep"]) >= 0]
        a = med(a_none) if a_none else None
        b, c, d, e = g("B_cpu_hit"), g("C_tiering_fs"), g("D_gds"), g("E_posix")
        f = lambda x: f"{x:8.4f}" if x else "       -"
        ratio = f"{c/d:5.2f}x" if (c and d) else "-"
        print(f"{P:5d} | {f(a)} {f(b)} {f(c)} {f(d)} {f(e)} | {ratio}")
    print("  store wall (median, s):")
    for P in (1024, 2032):
        parts = []
        for cfg in ("tiering", "expfs-cufile", "expfs-posix"):
            v = store.get((runner, P, cfg))
            parts.append(f"{cfg}={med(v):.3f}" if v else f"{cfg}=-")
        print(f"    P={P}: " + "  ".join(parts))
    print("  A_warm(store 켠 상태) vs A(none) — store가 warm TTFT에 주는 영향:")
    for P in (1024, 2032):
        parts = []
        for cfg in ("none", "tiering", "expfs-cufile", "expfs-posix"):
            v = [float(r["ttft_s"]) for r in rows
                 if r["runner"] == runner and int(r["prefix"]) == P
                 and r["arm"] == "A_warm" and r["config"] == cfg and int(r["rep"]) >= 0]
            parts.append(f"{cfg}={med(v):.3f}" if v else f"{cfg}=-")
        print(f"    P={P}: " + "  ".join(parts))
    print("  동시 8-load 총시간:")
    for r in rows:
        if r["runner"] == runner and r["arm"].startswith("conc"):
            print(f"    {r['config']:14s} {r['arm']:10s}: hit={float(r['ttft_s']):.3f}s "
                  f"warm={float(r['store_wall_s']):.3f}s n_io={r['n_io']}")
