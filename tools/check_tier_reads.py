#!/usr/bin/env python3
"""요청별 실제 적재 층(어느 티어에서 몇 바이트를 읽었나)으로 적중 요청을 나눠 TTFT를 본다.

requests.jsonl의 load_bytes(kvtrace, --kv-trace 필요)를 쓴다.
  hybrid/cufile: {"host": B, "<루트 디렉터리>": B, ...}. 루트가 SSD인지 DRAM인지는 result.json kv_roots[].fs 로 판단
                 (마운트 장치 이름에 ssd 가 들어가거나 fstype 이 ext4/xfs 인 로컬 NVMe 루트 → --ssd-roots 로 명시 가능)
  mooncake:      {"mc_memory": B, "mc_disk": B, "mc_unknown": B}
층 구분: dram(host pinned·원격 DRAM 램디스크·mooncake memory) / ssd(로컬 NVMe·원격 SSD·mooncake disk).
적중 요청을 SSD 몫(ssd/(dram+ssd))으로 0 / (0,0.5) / [0.5,1] 세 구간에 나눠 건수, 적재 GiB, TTFT(도착)·엔진지연 중앙/p95를 낸다.
용법: python tools/check_tier_reads.py --dir results/<캠페인> [--order "mooncake ours-ratio"] [--ssd-roots kv-p3-local,rain-ssd]
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--order", default="")
ap.add_argument("--phase", default="stream")
ap.add_argument("--ssd-roots", default="kv-p3-local,rain-ssd,local-ssd",
                help="쉼표 구분. 루트 디렉터리 경로에 이 문자열이 들어가면 SSD 층으로 본다")
a, _unknown = ap.parse_known_args()
D = a.dir
conds = [c for c in a.order.split() if os.path.exists(os.path.join(D, c, "result.json"))]
conds += sorted(c for c in os.listdir(D) if c not in conds and os.path.exists(os.path.join(D, c, "result.json")))
SSD_MARK = [s.strip() for s in a.ssd_roots.split(",") if s.strip()]


def q(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def tiers(lb):
    dram = ssd = unk = 0
    for k, v in (lb or {}).items():
        if k == "host" or k == "mc_memory":
            dram += v
        elif k == "mc_disk":
            ssd += v
        elif k == "mc_unknown":
            unk += v
        elif any(m in k for m in SSD_MARK):
            ssd += v
        else:
            dram += v  # 원격 DRAM 램디스크 등
    return dram, ssd, unk


print(f"== 요청별 적재 층 (phase {a.phase}, SSD 루트 표식 {SSD_MARK})")
for c in conds:
    r = json.load(open(os.path.join(D, c, "result.json")))
    rq = [json.loads(l) for l in open(os.path.join(D, c, "requests.jsonl"))]
    S = [x for x in rq if x.get("phase") == a.phase]
    if not any(x.get("load_bytes") for x in S):
        print(f"  {c:11} load_bytes 없음(--kv-trace 없이 돈 런)")
        continue
    H = [x for x in S if x.get("matched_of", 0) > 0 and x.get("first_mono") is not None]
    n_nofirst = sum(1 for x in S if x.get("first_mono") is None)
    if n_nofirst:
        print(f"  {c:11} 첫 토큰 없는 요청 {n_nofirst}건 제외")
    bins = {"ssd 0": [], "ssd (0,0.5)": [], "ssd [0.5,1]": []}
    tot = [0, 0, 0]
    for x in H:
        d, s_, u = tiers(x.get("load_bytes"))
        tot[0] += d; tot[1] += s_; tot[2] += u
        share = s_ / (d + s_) if (d + s_) else 0.0
        x["_share"] = share; x["_ssd"] = s_; x["_dram"] = d
        (bins["ssd 0"] if share == 0 else bins["ssd (0,0.5)"] if share < 0.5 else bins["ssd [0.5,1]"]).append(x)
    print(f"  {c:11} 적중 {len(H)}건, 적재 합 dram {tot[0] / 2**30:.2f} GiB / ssd {tot[1] / 2**30:.2f} GiB / unknown {tot[2] / 2**30:.2f} GiB")
    print(f"  {'':11} {'구간':12} | 건수 | 적중토큰 중앙 | 적재 GiB(dram/ssd) | TTFT(도착) 중앙/p95 | 엔진지연 중앙/p95 | 도착시각 중앙 s")
    for name, xs in bins.items():
        if not xs:
            print(f"  {'':11} {name:12} |    0 |")
            continue
        ta = [x["first_wall"] - x["arrival_wall"] for x in xs if x.get("first_wall")]
        te = [x["first_mono"] - x["submit_mono"] for x in xs if x.get("first_mono")]
        print(f"  {'':11} {name:12} | {len(xs):4} | {q([x['matched_of'] for x in xs], 0.5):13.0f} | "
              f"{sum(x['_dram'] for x in xs) / 2**30:6.2f}/{sum(x['_ssd'] for x in xs) / 2**30:6.2f}      | "
              f"{q(ta, 0.5):6.2f}/{q(ta, 0.95):6.2f}        | {q(te, 0.5):6.2f}/{q(te, 0.95):6.2f}     | {q([x['arrival_s'] for x in xs], 0.5):6.1f}")
    # 적중 토큰 규모를 맞춘 비교: ssd 0 구간과 ssd>0 구간에서 matched_of 가 비슷한(±25 %) 쌍의 엔진지연 차
    z = bins["ssd 0"]; nz = bins["ssd (0,0.5)"] + bins["ssd [0.5,1]"]
    pairs = []
    for x in nz:
        cand = [y for y in z if abs(y["matched_of"] - x["matched_of"]) <= 0.25 * max(1, x["matched_of"])]
        if cand:
            y = min(cand, key=lambda y: abs(y["matched_of"] - x["matched_of"]))
            pairs.append((x["first_mono"] - x["submit_mono"]) - (y["first_mono"] - y["submit_mono"]))
    if pairs:
        print(f"  {'':11} 적중량 맞춘 쌍 {len(pairs)}건: 엔진지연 차(ssd>0 − ssd 0) 중앙 {q(pairs, 0.5):+.3f} s, ssd 쪽이 느린 쌍 {sum(1 for p in pairs if p > 0)}/{len(pairs)}")


# ---- 공통 적중 요청 짝 비교(--pair A B): 같은 rid가 두 조건 모두 적중일 때 적중 토큰·SSD 바이트·TTFT(도착)·엔진지연 ----
import sys as _sys
if "--pair" in _sys.argv:
    i = _sys.argv.index("--pair"); A, B = _sys.argv[i + 1], _sys.argv[i + 2]

    def _load(c):
        rq = [json.loads(l) for l in open(os.path.join(D, c, "requests.jsonl"))]
        return {x["rid"]: x for x in rq if x.get("phase") == a.phase and x.get("first_mono") is not None}

    RA, RB = _load(A), _load(B)
    both = [k for k in RA if k in RB and RA[k].get("matched_of", 0) > 0 and RB[k].get("matched_of", 0) > 0]
    rows = []
    for k in both:
        xa, xb = RA[k], RB[k]
        da, sa, _ = tiers(xa.get("load_bytes")); db, sb, _ = tiers(xb.get("load_bytes"))
        rows.append(dict(rid=k, tok_a=xa["matched_of"], tok_b=xb["matched_of"], ssd_a=sa / 2**30, ssd_b=sb / 2**30,
                         ttft_a=xa["first_wall"] - xa["arrival_wall"], ttft_b=xb["first_wall"] - xb["arrival_wall"],
                         eng_a=xa["first_mono"] - xa["submit_mono"], eng_b=xb["first_mono"] - xb["submit_mono"], arr=xa["arrival_s"]))
    print()
    print(f"== 공통 적중 요청 {len(both)}건 ({A} vs {B}); A만 적중 {sum(1 for k in RA if RA[k].get('matched_of',0)>0 and (k not in RB or RB[k].get('matched_of',0)==0))}, B만 적중 {sum(1 for k in RB if RB[k].get('matched_of',0)>0 and (k not in RA or RA[k].get('matched_of',0)==0))}")
    if rows:
        def col(key): return [r[key] for r in rows]
        print(f"  적중토큰 중앙 A {q(col('tok_a'),0.5):.0f} / B {q(col('tok_b'),0.5):.0f}; SSD GiB 합 A {sum(col('ssd_a')):.2f} / B {sum(col('ssd_b')):.2f}; SSD>0 요청 A {sum(1 for r in rows if r['ssd_a']>0)} / B {sum(1 for r in rows if r['ssd_b']>0)}")
        print(f"  TTFT(도착) 중앙 A {q(col('ttft_a'),0.5):.2f} / B {q(col('ttft_b'),0.5):.2f} s; 엔진지연 중앙 A {q(col('eng_a'),0.5):.2f} / B {q(col('eng_b'),0.5):.2f} s")
        d_t = [r["ttft_b"] - r["ttft_a"] for r in rows]; d_e = [r["eng_b"] - r["eng_a"] for r in rows]
        print(f"  paired(B−A): TTFT(도착) 중앙 {q(d_t,0.5):+.2f} s (B 느린 {sum(1 for x in d_t if x>0)}/{len(rows)}), 엔진지연 중앙 {q(d_e,0.5):+.2f} s (B 느린 {sum(1 for x in d_e if x>0)}/{len(rows)})")
        # 둘 다 SSD를 읽은 요청만
        ss = [r for r in rows if r["ssd_a"] > 0 and r["ssd_b"] > 0]
        if ss:
            d_e2 = [r["eng_b"] - r["eng_a"] for r in ss]
            print(f"  둘 다 SSD 읽은 {len(ss)}건: 적중토큰 중앙 A {q([r['tok_a'] for r in ss],0.5):.0f}/B {q([r['tok_b'] for r in ss],0.5):.0f}, SSD GiB 중앙 A {q([r['ssd_a'] for r in ss],0.5):.2f}/B {q([r['ssd_b'] for r in ss],0.5):.2f}, 엔진지연 중앙 A {q([r['eng_a'] for r in ss],0.5):.2f}/B {q([r['eng_b'] for r in ss],0.5):.2f}, paired(B−A) 중앙 {q(d_e2,0.5):+.2f} s")
        print("  rid | 도착 s | 적중토큰 A/B | SSD GiB A/B | TTFT(도착) A/B | 엔진지연 A/B   (SSD 바이트 큰 순 12건)")
        for r in sorted(rows, key=lambda r: -(r["ssd_a"] + r["ssd_b"]))[:12]:
            print(f"  {r['rid']:11} | {r['arr']:6.1f} | {r['tok_a']:6d}/{r['tok_b']:6d} | {r['ssd_a']:5.2f}/{r['ssd_b']:5.2f} | {r['ttft_a']:6.2f}/{r['ttft_b']:6.2f} | {r['eng_a']:5.2f}/{r['eng_b']:5.2f}")
