#!/usr/bin/env python3
"""저장 간섭 실험(run_obs --mode stream_replay, --stream-store on|off|shadow) 결과 대조.

(1) 조건별 stream 표: queue, TTFT(도착), 엔진지연, decode(첫 토큰→완료), e2e(제출) 중앙/p95를 적중·미적중으로 나눠
(2) 전제 확인: 조건 사이에 요청별 적중 토큰(matched_of)·GPU 캐시 토큰·출력 길이·층별 적재 바이트(load_bytes)가 같은지,
    출력 토큰열이 같은지, 저장 켠 조건에서 쓰기가 실제로 났는지(kv_io write, shadow gib, 저장 큐 통계)
(3) 요청별 paired 차이: 같은 요청의 decode·엔진지연 차(조건 - 기준 조건) 중앙값과 부호 비율
용법: python tools/check_replay.py --dir results/<캠페인> [--order "store-off store-shadow store-on"] [--ref store-off]
"""
import argparse
import json
import os
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--order", default="")
ap.add_argument("--ref", default="")
ap.add_argument("--phase", default="stream")
a = ap.parse_args()
D = a.dir
conds = [c for c in a.order.split() if os.path.exists(os.path.join(D, c, "result.json"))]
conds += sorted(c for c in os.listdir(D) if c not in conds and os.path.exists(os.path.join(D, c, "result.json")))
if not conds:
    raise SystemExit(f"result.json이 있는 조건이 없음: {D}")
ref = a.ref or conds[0]


def q(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def med(v):
    return st.median(v) if v else 0.0


def load(c):
    r = json.load(open(os.path.join(D, c, "result.json")))
    rq = [json.loads(l) for l in open(os.path.join(D, c, "requests.jsonl"))]
    S = {}
    for x in rq:
        if x.get("phase") != a.phase:
            continue
        x["ttft_arr"] = (x["first_wall"] - x["arrival_wall"]) if x.get("first_wall") and x.get("arrival_wall") else None
        x["eng"] = (x["first_mono"] - x["submit_mono"]) if x.get("first_mono") else None
        x["dec"] = (x["finish_mono"] - x["first_mono"]) if x.get("finish_mono") and x.get("first_mono") else None
        x["e2e"] = (x["finish_mono"] - x["submit_mono"]) if x.get("finish_mono") else None
        x["hit"] = x.get("matched_of", 0) > 0
        S[x["rid"]] = x
    return r, S


R = {c: load(c) for c in conds}


def fmt(v):
    return f"{q(v, 0.5):6.2f}/{q(v, 0.95):6.2f}"


print(f"== 저장 간섭: {D} (phase {a.phase}, 기준 {ref})")
print("조건 | store | 요청 | wall s | 적중 | 적중토큰 | queue | TTFT(도착) | 엔진지연 | decode | e2e(제출) | [적중만] 엔진지연 | decode | [미적중] 엔진지연 | decode")
for c in conds:
    r, S = R[c]
    ph = (r.get("phases") or {}).get(a.phase) or {}
    L = list(S.values()); H = [x for x in L if x["hit"]]; M = [x for x in L if not x["hit"]]
    def col(xs, k):
        return [x[k] for x in xs if x.get(k) is not None]
    print(" | ".join([f"{c:12}", f"{r['args'].get('stream_store', '-'):6}", f"{len(L):4}", f"{ph.get('wall_s', 0):7.1f}",
                      f"{len(H):4}", f"{sum(x['matched_of'] for x in H):8}",
                      fmt(col(L, 'queue_s')), fmt(col(L, 'ttft_arr')), fmt(col(L, 'eng')), fmt(col(L, 'dec')), fmt(col(L, 'e2e')),
                      fmt(col(H, 'eng')), fmt(col(H, 'dec')), fmt(col(M, 'eng')), fmt(col(M, 'dec'))]))

print()
print("== 전제 확인(기준 조건 대비)")
_, S0 = R[ref]
for c in conds:
    r, S = R[c]
    both = [k for k in S0 if k in S]
    dm = sum(1 for k in both if S0[k].get("matched_of") != S[k].get("matched_of"))
    dg = sum(1 for k in both if S0[k].get("gpu_cached") != S[k].get("gpu_cached"))
    dn = sum(1 for k in both if S0[k].get("ntok") != S[k].get("ntok"))
    di = sum(1 for k in both if S0[k].get("ids") != S[k].get("ids"))
    dl = sum(1 for k in both if (S0[k].get("load_bytes") or {}) != (S[k].get("load_bytes") or {}))
    lb = {}
    for x in S.values():
        for kk, v in (x.get("load_bytes") or {}).items():
            lb[kk] = lb.get(kk, 0) + v
    io = r.get("kv_io") or {}
    m = r.get("kv_manager") or {}
    sh = (m.get("ssd_manager") or {}).get("shadow") or {}
    pre = (r.get("phases") or {}).get("prefill") or {}
    cc = r.get("commit_check_prefill") or {}
    print(f"  {c:12} 요청 {len(both)}건 중 다른 것: matched {dm}, gpu_cached {dg}, 출력길이 {dn}, 토큰열 {di}, load_bytes {dl}")
    print(f"  {'':12} prefill {pre.get('wall_s', 0):.1f} s, commit ok={cc.get('ok')} files={cc.get('files_on_disk')}; "
          f"cuFile write {io.get('write_n', 0)}건 {io.get('write_gib', 0):.2f} GiB (stages ev/io {io.get('write_stages_s', {}).get('ev', 0)}/{io.get('write_stages_s', {}).get('io', 0)} s), "
          f"read {io.get('read_n', 0)}건 {io.get('read_gib', 0):.2f} GiB; shadow {sh}; stores host/ssd {m.get('stores_host')}/{m.get('stores_ssd')}")
    print(f"  {'':12} 적재 바이트 합(GiB): " + ", ".join(f"{('/'.join(k.rstrip('/').split('/')[-2:]) if '/' in k else k)}={v / 2**30:.2f}" for k, v in sorted(lb.items())))

print()
print(f"== paired 차이(조건 − {ref}), 같은 요청끼리")
for c in conds:
    if c == ref:
        continue
    _, S = R[c]
    both = [k for k in S0 if k in S]
    for name, key, sel in (("decode(전체)", "dec", both), ("decode(적중)", "dec", [k for k in both if S0[k]["hit"]]),
                           ("엔진지연(전체)", "eng", both), ("엔진지연(적중)", "eng", [k for k in both if S0[k]["hit"]]),
                           ("queue", "queue_s", both)):
        d = [S[k][key] - S0[k][key] for k in sel if S[k].get(key) is not None and S0[k].get(key) is not None]
        if not d:
            continue
        print(f"  {c:12} {name:12} n={len(d):3} 중앙 {med(d):+.3f} s, p95 {q(d, 0.95):+.3f}, 조건이 느린 요청 {sum(1 for x in d if x > 0)}/{len(d)}")
