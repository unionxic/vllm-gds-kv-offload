# W1 요약: 러너 내부 C vs D. "각 설계의 최선 구성 간 end-to-end 비교" (transport 단독 비교 아님).
import csv
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "w1_results.csv"))))
med = statistics.median


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


tags = sorted({r["tag"] for r in rows})
by = {t: [r for r in rows if r["tag"] == t] for t in tags}

print("tag | n | TTFT p50/p95/p99 | tok/s | GPU-hit% storage-hit% computed% "
      "| ssd_read GiB (IOs) | store GiB (IOs) | cpu_s/req")
for t in tags:
    rs = by[t]
    tt = [float(r["ttft_s"]) for r in rs]
    ptok = sum(int(r["prompt_tokens"]) for r in rs)
    cached = sum(int(r["cached_tokens"] or 0) for r in rs)
    matched = sum(int(r["matched_tokens"]) for r in rs)
    gpu_hit = cached - matched  # cached = GPU prefix hit + storage matched
    wall = json.load(open(os.path.join(HERE, f"w1_total_{t}.json")))["wall_s"]
    cpu = json.load(open(os.path.join(HERE, f"w1_total_{t}.json")))["cpu_s"]
    rb = sum(int(r["ssd_read_bytes"]) for r in rs)
    ri = sum(int(r["ssd_read_ios"]) for r in rs)
    tot = json.load(open(os.path.join(HERE, f"w1_total_{t}.json")))
    sb = tot["fs_store_b"] + tot["tp_write_b"]
    si = tot["fs_store_n"] + tot["tp_write_n"]
    print(f"{t:12s} | {len(rs)} | {med(tt):.3f}/{pct(tt,95):.3f}/{pct(tt,99):.3f} "
          f"| {ptok/wall:7.0f} | {gpu_hit/ptok*100:5.1f}% {matched/ptok*100:5.1f}% "
          f"{(ptok-cached)/ptok*100:5.1f}% | {rb/2**30:6.2f} ({ri}) "
          f"| {sb/2**30:6.2f} ({si}) | {cpu/len(rs):6.2f}")

print("\nstorage-hit 요청만의 TTFT (storage-matched > 0):")
for t in tags:
    hit = [float(r["ttft_s"]) for r in by[t] if int(r["matched_tokens"]) > 0]
    miss = [float(r["ttft_s"]) for r in by[t] if int(r["matched_tokens"]) == 0]
    if hit:
        print(f"  {t:12s}: hit n={len(hit)} p50={med(hit):.3f} p95={pct(hit,95):.3f} "
              f"| miss n={len(miss)} p50={med(miss):.3f}")

print("\nhost-memory traffic (유도): C = 2×ssd_read+2×store(스테이징 왕복), D = 0")
for t in tags:
    tot = json.load(open(os.path.join(HERE, f"w1_total_{t}.json")))
    rb = tot["fs_load_b"]
    sb = tot["fs_store_b"]
    host = 2 * (rb + sb) if t.split("-")[1] == "C" else 0
    print(f"  {t:12s}: {host/2**30:.2f} GiB")
