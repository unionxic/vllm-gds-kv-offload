"""write-behind miss 대책 세 조건(current, pending-wait, cpu-landing) 점검.

기준선 행(mooncake)은 이 캠페인에서 다시 돌리지 않고 --mooncake가 가리키는 기존 런에서 가져온다.
(1) 조건별 표: 적중 요청·적중 토큰, TTFT·e2e 중앙/p95, wall clock, 티어별 적중·저장,
    pending-wait는 기다린 횟수·평균 대기·상한 초과 수,
(2) 조건 사이 출력 토큰열 일치. 어긋난 요청은 첫 불일치 위치와 함께 목록으로 뽑는다.

usage: python tools/check_stream_fix.py [--dir results/stream-bailian1240-8b-fix]
         [--mooncake results/stream-bailian1240-8b/mooncake] [--order current,pending-wait,cpu-landing]
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/stream-bailian1240-8b-fix")
ap.add_argument("--mooncake", default="results/stream-bailian1240-8b/mooncake",
                help="기준선 행을 읽어 올 기존 런 폴더(없으면 생략)")
ap.add_argument("--order", default="current,pending-wait,cpu-landing")
ap.add_argument("--phase", default="stream")
ap.add_argument("--max-mismatch", type=int, default=10)
a = ap.parse_args()

D = os.path.abspath(a.dir)
if not os.path.isdir(D):
    sys.exit(f"결과 폴더가 없음: {D}")
conds = [c for c in a.order.split(",") if os.path.exists(os.path.join(D, c, "result.json"))]
conds += sorted(c for c in os.listdir(D)
                if c not in conds and os.path.exists(os.path.join(D, c, "result.json")))
if not conds:
    sys.exit(f"result.json이 있는 조건이 없음: {D}")


def load(path):
    r = json.load(open(os.path.join(path, "result.json")))
    f = os.path.join(path, "requests.jsonl")
    rq = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    return r, rq


R = {c: load(os.path.join(D, c)) for c in conds}
MC = os.path.abspath(a.mooncake) if a.mooncake else None
if MC and os.path.exists(os.path.join(MC, "result.json")):
    R["mooncake"] = load(MC)
    rows = ["mooncake"] + conds
else:
    MC = None
    rows = list(conds)


def q(v, p):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def phase_of(r):
    return (r.get("phases") or {}).get(a.phase) or {}


print(f"== 조건별 ({D}, phase {a.phase}" + (f", 기준선 {MC}" if MC else "") + ")")
hdr = ("조건", "transport", "요청", "wall s", "적중요청", "적중토큰",
       "TTFT 중앙/p95 s", "e2e 중앙/p95 s", "queue 중앙/p95 s")
print(" | ".join(hdr))
for c in rows:
    r, rq = R[c]
    ph = phase_of(r)
    rqp = [x for x in rq if x.get("phase") == a.phase]
    n = ph.get("n_requests") or len(rqp)
    print(" | ".join([
        f"{c:12}", f"{r['args']['kv_transport']:9}", f"{n:4}", f"{ph.get('wall_s', 0):8.1f}",
        f"{sum(1 for x in rqp if x.get('matched_of', 0) > 0):5}",
        f"{sum(x.get('matched_of', 0) for x in rqp):9}",
        f"{q(ph.get('ttft') or [], 0.5):7.2f}/{q(ph.get('ttft') or [], 0.95):7.2f}",
        f"{q(ph.get('e2e') or [], 0.5):7.2f}/{q(ph.get('e2e') or [], 0.95):7.2f}",
        f"{q(ph.get('queue_s') or [x.get('queue_s', 0.0) for x in rqp], 0.5):7.2f}"
        f"/{q(ph.get('queue_s') or [x.get('queue_s', 0.0) for x in rqp], 0.95):7.2f}"]))

print()
print("== 티어 적중·저장과 대기 정책")
for c in rows:
    r = R[c][0]
    m = r.get("kv_manager") or {}
    if not m:
        mc = r.get("kv_mooncake") or {}
        alloc = mc.get("master_allocated_bytes", 0) if "error" not in mc else 0
        print(f"  {c:12} Mooncake 풀 {alloc / 2**30:.2f} GiB (티어별 계측 없음)")
        continue
    ssd = m.get("ssd_manager") or {}
    print(f"  {c:12} 배치 {str(m.get('placement')):10} host몫 {str(m.get('host_share')):5} "
          f"적중 host/ssd {m.get('hits_host')}/{m.get('hits_ssd')} miss {m.get('misses')} "
          f"보류 host/ssd {m.get('pending_host')}/{m.get('pending_ssd')} "
          f"저장 host/ssd {m.get('stores_host')}/{m.get('stores_ssd')}")
    pw = ssd.get("pending_wait")
    if pw:
        print(f"  {c:12}   pending-wait: 기다린 (요청,키) {pw['waits']}쌍 / 조회 {pw['wait_calls']}회, "
              f"적중으로 풀림 {pw['resolved']}, 상한 초과 {pw['timeouts']}, "
              f"평균 대기 {pw['wait_s_mean']:.3f} s (합 {pw['wait_s_total']:.1f} s), "
              f"안 기다림: 재계산이 더 쌈 {pw['declined']} / 비용 미상 {pw['unknown_cost']}")
        print(f"  {c:12}   추정값: 청크 한 칸 {pw['chunk_service_s']:.4f} s, "
              f"토큰당 재계산 {pw['recompute_s_per_token'] * 1e3:.4f} ms "
              f"(청크 {pw['tokens_per_chunk']}토큰 → {pw['recompute_s_per_token'] * pw['tokens_per_chunk']:.3f} s), "
              f"상한 {pw['max_s']} s")

print()
print("== 출력 토큰열 대조")
base = rows[0]
ref = {x["rid"]: x.get("ids") for x in R[base][1] if x.get("phase") == a.phase}
for c in rows[1:]:
    cur = {x["rid"]: x.get("ids") for x in R[c][1] if x.get("phase") == a.phase}
    both = sorted(set(ref) & set(cur))
    miss = sorted(set(ref) - set(cur)); extra = sorted(set(cur) - set(ref))
    bad = [k for k in both if ref[k] != cur[k]]
    ok = not (miss or extra or bad)
    print(f"  {base} vs {c}: {'일치' if ok else '불일치'} (요청 {len(both)}건 비교"
          + (f", 누락 {len(miss)}, 초과 {len(extra)}, 다름 {len(bad)}" if not ok else "") + ")")
    for k in bad[:a.max_mismatch]:
        i = next((j for j, (x, y) in enumerate(zip(ref[k], cur[k])) if x != y), None)
        print(f"    {k}: 첫 불일치 위치 {i}, {base}={ref[k]} {c}={cur[k]}")
    if len(bad) > a.max_mismatch:
        print(f"    ... 외 {len(bad) - a.max_mismatch}건")
