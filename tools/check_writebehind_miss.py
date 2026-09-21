"""write-behind 때문에 생긴 miss를 keytrace.jsonl에서 가려낸다.

cufile_fs_trace_keys=true로 돌린 런 폴더가 대상이다. 트레이스 한 줄이 한 사건이고
시각은 러너의 submit_mono와 같은 time.monotonic() 기준이다.
  enq  파일 티어가 저장 대상으로 잡은 시각(쓰기 큐 투입)
  cmt  쓰기가 끝나 읽을 수 있게 된 시각
  lk   파일 티어 lookup 결과, lkh 상위 매니저(host+파일) 최종 판정

판정: 어떤 요청의 miss 키가 "그 요청보다 먼저 투입됐고 그 요청 뒤에 커밋된" 것이면
그 miss는 쓰기가 늦어서 생긴 것이다. 기준 두 가지를 따로 센다.
  lookup 기준  enq <= lookup 시각 < cmt (그 조회 자체가 쓰기 중인 키를 만남)
  도착 기준    enq <= 요청 도착 시각 < cmt (요청이 올 때 이미 쓰는 중이었음)
출력은 그런 요청·키 수, 기다렸어야 할 시간 분포, 그리고 기준 런과의 적중 요청 수 차이 대조.

usage: python tools/check_writebehind_miss.py --dir results/stream-bailian1240-8b-trace/ours-ratio \
         [--ref results/stream-bailian1240-8b/mooncake] [--phase stream]
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="keytrace.jsonl이 있는 런 폴더")
ap.add_argument("--ref", default=None, help="적중 요청 수를 맞대 볼 기준 런 폴더(예: mooncake)")
ap.add_argument("--phase", default="stream")
ap.add_argument("--max-list", type=int, default=15, help="목록으로 뽑을 요청 수")
a = ap.parse_args()

D = os.path.abspath(a.dir)
TR = os.path.join(D, "keytrace.jsonl")
if not os.path.exists(TR):
    res = os.path.join(D, "result.json")
    if os.path.exists(res):
        p = ((json.load(open(res)).get("kv_manager") or {}).get("ssd_manager") or {}).get("trace_path")
        if p and os.path.exists(p):
            TR = p
if not os.path.exists(TR):
    sys.exit(f"keytrace.jsonl이 없음: {D} (cufile_fs_trace_keys=true로 돌린 런이어야 함)")

rows = [json.loads(l) for l in open(TR) if l.strip()]
reqs = [json.loads(l) for l in open(os.path.join(D, "requests.jsonl"))
        if l.strip() and json.loads(l).get("phase") == a.phase]
RQ = {r["rid"]: r for r in reqs}


def base_rid(rid):
    """엔진이 붙인 접미사를 떼어 requests.jsonl의 rid로 되돌린다(stream-42-9fec -> stream-42)."""
    if rid is None:
        return None
    if rid in RQ:
        return rid
    head = rid.rsplit("-", 1)[0]
    return head if head in RQ else rid


# ---- 키별 투입·커밋 시각 ----
enq, cmt, enq_q = {}, {}, {}
lk_all, lk_hy = [], []
for r in rows:
    ev = r["ev"]
    if ev == "enq":
        enq.setdefault(r["k"], r["t"]); enq_q.setdefault(r["k"], r.get("q"))
    elif ev == "cmt":
        if r.get("ok", True):
            cmt.setdefault(r["k"], r["t"])
    elif ev == "lk":
        lk_all.append(r)
    elif ev == "lkh":
        lk_hy.append(r)

print(f"== 트레이스 ({TR})")
print(f"  줄 {len(rows)}: enq {len(enq)}키, cmt {len(cmt)}키, 파일티어 lookup {len(lk_all)}, 상위 lookup {len(lk_hy)}")
n_commit_only = sum(1 for k in enq if k not in cmt)
print(f"  투입됐지만 커밋 기록이 없는 키 {n_commit_only}(런 끝까지 쓰기가 안 끝난 몫)")
if cmt:
    lat = sorted(cmt[k] - enq[k] for k in cmt if k in enq)
    if lat:
        def qq(v, p):
            return v[min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))]
        print(f"  투입→커밋 지연 s: 중앙 {qq(lat,0.5):.2f}, p95 {qq(lat,0.95):.2f}, 최대 {lat[-1]:.2f}")

# ---- 요청별 miss 분석 ----
lk = lk_hy if lk_hy else lk_all  # 상위 판정이 있으면 그쪽이 최종 판정
by_req = {}
for r in lk:
    by_req.setdefault(base_rid(r.get("r")), []).append(r)

wb_lookup_req, wb_arrival_req = {}, {}
wb_keys_lookup, wb_keys_arrival = set(), set()
decisive = {}  # 요청 -> 마지막 miss 행(그 요청의 적중 경계를 정한 조회)
for rid, rs in by_req.items():
    rq = RQ.get(rid)
    arrival = rq.get("submit_mono") if rq else None
    miss = [x for x in rs if x.get("res") == "miss"]
    if not miss:
        continue
    decisive[rid] = max(miss, key=lambda x: x["t"])
    for x in miss:
        k = x["k"]
        e, c = enq.get(k), cmt.get(k)
        if e is None:
            continue
        if e <= x["t"] and (c is None or c > x["t"]):
            wb_lookup_req.setdefault(rid, []).append((x, c))
            wb_keys_lookup.add(k)
        if arrival is not None and e <= arrival and (c is None or c > arrival):
            wb_arrival_req.setdefault(rid, []).append((x, c))
            wb_keys_arrival.add(k)


def dist(v, label):
    if not v:
        print(f"  {label}: 표본 없음")
        return
    s = sorted(v)

    def qq(p):
        return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    print(f"  {label}: n={len(s)} 최소 {s[0]:.2f} 중앙 {qq(0.5):.2f} p95 {qq(0.95):.2f} 최대 {s[-1]:.2f} 평균 {sum(s)/len(s):.2f}")


print()
print(f"== write-behind가 원인인 miss (phase {a.phase}, 요청 {len(RQ)}건)")
print(f"  lookup 기준: 요청 {len(wb_lookup_req)}건, 키 {len(wb_keys_lookup)}개")
print(f"  도착 기준  : 요청 {len(wb_arrival_req)}건, 키 {len(wb_keys_arrival)}개")

wait_lookup = [c - x["t"] for rs in wb_lookup_req.values() for x, c in rs if c is not None]
wait_arrival = []
for rid, rs in wb_arrival_req.items():
    arr = RQ[rid]["submit_mono"]
    wait_arrival += [c - arr for x, c in rs if c is not None]
print()
print("== 기다렸다면 들었을 시간(초)")
dist(wait_lookup, "커밋 - 조회 시각")
dist(wait_arrival, "커밋 - 요청 도착 시각")
never = sum(1 for rs in wb_lookup_req.values() for x, c in rs if c is None)
if never:
    print(f"  (런이 끝날 때까지 커밋이 안 온 조회 {never}건은 위 분포에서 뺌)")

# ---- 적중 요청 수 차이 대조 ----
hit_req = [r for r in reqs if r.get("matched_of", 0) > 0]
no_hit = [r for r in reqs if r.get("matched_of", 0) == 0]
rescuable = [r for r in no_hit if r["rid"] in wb_lookup_req]
rescuable_arr = [r for r in no_hit if r["rid"] in wb_arrival_req]
print()
print("== 적중 요청 수 대조")
print(f"  이 런: 적중 {len(hit_req)} / 무적중 {len(no_hit)} (요청 {len(reqs)})")
print(f"  무적중 가운데 write-behind miss가 있던 요청: lookup 기준 {len(rescuable)}건, 도착 기준 {len(rescuable_arr)}건")
if a.ref:
    RD = os.path.abspath(a.ref)
    rr = [json.loads(l) for l in open(os.path.join(RD, "requests.jsonl"))
          if l.strip() and json.loads(l).get("phase") == a.phase]
    ref_hit = sum(1 for r in rr if r.get("matched_of", 0) > 0)
    gap = ref_hit - len(hit_req)
    print(f"  기준 런({os.path.basename(RD)}): 적중 {ref_hit} → 차이 {gap}건")
    print(f"  차이 대비 설명 몫: lookup 기준 {len(rescuable)}/{gap}" if gap > 0 else "  기준 런보다 적중이 많거나 같음")

print()
print(f"== write-behind miss가 있던 무적중 요청 (앞 {a.max_list}건)")
print("  요청 | 프롬프트토큰 | 도착 s | miss키 | 최장 대기 s")
for r in sorted(rescuable, key=lambda r: r.get("arrival_s", 0))[:a.max_list]:
    rs = wb_lookup_req[r["rid"]]
    w = [c - x["t"] for x, c in rs if c is not None]
    print(f"  {r['rid']:14} {r.get('tokens', 0):7} {r.get('arrival_s', 0):9.1f} {len(rs):6} "
          f"{max(w) if w else float('nan'):11.2f}")
if len(rescuable) > a.max_list:
    print(f"  ... 외 {len(rescuable) - a.max_list}건")
