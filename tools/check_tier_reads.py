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
a = ap.parse_args()
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
    H = [x for x in S if x.get("matched_of", 0) > 0]
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
