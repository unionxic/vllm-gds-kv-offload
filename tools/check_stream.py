"""stream 모드 캠페인 점검. 조건은 성능 비교 두 개(mooncake, ours-ratio)만 본다.
(1) 조건별 표: wall clock, 도착 span, queue_s/TTFT/e2e 중앙값과 p95, 적중 요청 수와 적중 토큰,
    처리율(요청/시간), 티어별 바이트(있는 조건만),
(2) 두 조건 사이 출력 토큰열 일치. 어긋난 요청은 목록으로 뽑는다.
usage: python tools/check_stream.py [--dir results/phase3-stream-8b] [--order mooncake,ours-ratio]"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/phase3-stream-8b")
ap.add_argument("--order", default="mooncake,ours-ratio")
ap.add_argument("--phase", default="stream")
ap.add_argument("--max-mismatch", type=int, default=10, help="목록으로 뽑을 토큰열 불일치 요청 수")
a = ap.parse_args()

D = os.path.abspath(a.dir)
if not os.path.isdir(D):
    sys.exit(f"결과 폴더가 없음: {D}")
conds = [c for c in a.order.split(",") if os.path.exists(os.path.join(D, c, "result.json"))]
conds += sorted(c for c in os.listdir(D)
                if c not in conds and os.path.exists(os.path.join(D, c, "result.json")))
if not conds:
    sys.exit(f"result.json이 있는 조건이 없음: {D}")


def load(c):
    r = json.load(open(os.path.join(D, c, "result.json")))
    f = os.path.join(D, c, "requests.jsonl")
    rq = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
    return r, rq


R = {c: load(c) for c in conds}


def q(v, p):
    """정렬된 분위수(선형 보간 없이 가까운 순위)."""
    if not v:
        return 0.0
    s = sorted(v)
    i = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[i]


def short(path):
    return "/".join(p for p in str(path).rstrip("/").split("/")[-2:])


# ---- 조건별 표 ----
print(f"== stream 조건별 ({D}, phase {a.phase})")
# TTFT 두 정의를 요청별 시각으로 따로 계산한다(중앙값끼리 더하지 않음).
#   TTFT(도착)  = first_wall - arrival_wall  사용자가 겪는 첫 토큰까지의 시간(외부 대기 포함)
#   엔진지연     = first_mono - submit_mono   엔진 제출 → 첫 토큰(외부 대기 제외)
#   queue       = 도착 → 엔진 제출(max_concurrency 대기)
hdr = ("조건", "transport", "요청", "wall s", "도착span s", "queue 중앙/p95 s", "TTFT(도착) 중앙/p95 s",
       "엔진지연 중앙/p95 s", "e2e(제출) 중앙/p95 s", "적중요청", "적중토큰", "요청/시간")
print(" | ".join(hdr))
for c in conds:
    r, rq = R[c]
    ph = (r.get("phases") or {}).get(a.phase) or {}
    rqp = [x for x in rq if x.get("phase") == a.phase]
    wall = ph.get("wall_s") or 0.0
    n = ph.get("n_requests") or len(rqp)
    qs = [x.get("queue_s", 0.0) for x in rqp] or ph.get("queue_s") or []
    ta = [x["first_wall"] - x["arrival_wall"] for x in rqp if x.get("first_wall") and x.get("arrival_wall")]
    tt = [x["first_mono"] - x["submit_mono"] for x in rqp if x.get("first_mono")] or ph.get("ttft") or []
    e2 = [x["finish_mono"] - x["submit_mono"] for x in rqp if x.get("finish_mono")] or ph.get("e2e") or []
    hit_req = sum(1 for x in rqp if x.get("matched_of", 0) > 0)
    hit_tok = sum(x.get("matched_of", 0) for x in rqp)
    thr = (n / wall * 3600) if wall else 0.0
    print(" | ".join([
        f"{c:11}", f"{r['args']['kv_transport']:9}", f"{n:4}", f"{wall:8.1f}",
        f"{ph.get('arrival_span_s', 0):9.1f}",
        f"{q(qs, 0.5):7.2f}/{q(qs, 0.95):7.2f}",
        f"{q(ta, 0.5):7.2f}/{q(ta, 0.95):7.2f}",
        f"{q(tt, 0.5):7.2f}/{q(tt, 0.95):7.2f}",
        f"{q(e2, 0.5):7.2f}/{q(e2, 0.95):7.2f}",
        f"{hit_req:5}", f"{hit_tok:9}", f"{thr:8.1f}"]))

# ---- 티어별 바이트 ----
print()
print("== 티어별 바이트(있는 조건만)")
for c in conds:
    r = R[c][0]
    k = r.get("kv_io") or {}
    by = r.get("kv_files_by_root") or []
    mc = r.get("kv_mooncake") or {}
    by = [x for x in by if x.get("files")]  # 빈 루트는 그 조건이 쓰지 않은 것
    if mc and "error" not in mc:
        cap = mc.get("master_total_capacity_bytes", 0)
        alloc = mc.get("master_allocated_bytes", 0)
        print(f"  {c:11} Mooncake 풀(런 끝 시점) {alloc/2**30:8.3f} GiB / 용량 {cap/2**30:.1f} GiB")
        for kk, vv in sorted(mc.items()):
            if kk.startswith("segment_allocated_bytes"):
                seg = kk.split("segment=")[-1].strip('}"')
                print(f"  {c:11}   세그먼트 {seg:22} {vv/2**30:8.3f} GiB")
    elif by:
        # 파일 티어 조건: 런 끝 시점 루트별 실측 + 포크 계측의 cuFile 전송량
        for x in by:
            fs = next((p.get("fs") or {} for p in (r.get("kv_roots") or []) if p["dir"] == x["dir"]), {})
            print(f"  {c:11} 파일티어 {short(x['dir']):22} w={x['weight']:<3} "
                  f"fstype={fs.get('fstype','-'):5} files={x['files']:6} {x['bytes_gib']:8.3f} GiB")
        m = r.get("kv_manager") or {}
        print(f"  {c:11} cuFile write {k.get('write_gib',0):.2f} GiB / read {k.get('read_gib',0):.2f} GiB, "
              f"hits host/ssd={m.get('hits_host')}/{m.get('hits_ssd')} miss={m.get('misses')} "
              f"stores host/ssd={m.get('stores_host')}/{m.get('stores_ssd')}")
    else:
        print(f"  {c:11} 티어 계측 없음" + (f" ({mc.get('error')})" if mc else ""))

# ---- 출력 토큰열 대조 ----
print()
print("== 출력 토큰열 대조(조건 사이)")
if len(conds) < 2:
    print("  비교할 조건이 하나뿐")
else:
    base = conds[0]
    ref = {x["rid"]: x.get("ids") for x in R[base][1] if x.get("phase") == a.phase}
    for c in conds[1:]:
        cur = {x["rid"]: x.get("ids") for x in R[c][1] if x.get("phase") == a.phase}
        both = sorted(set(ref) & set(cur))
        miss = sorted(set(ref) - set(cur)); extra = sorted(set(cur) - set(ref))
        bad = [k for k in both if ref[k] != cur[k]]
        ok = not (miss or extra or bad)
        print(f"  {base} vs {c}: {'일치' if ok else '불일치'} (요청 {len(both)}건 비교"
              + (f", 누락 {len(miss)}, 초과 {len(extra)}, 다름 {len(bad)}" if not ok else "") + ")")
        for k in bad[:a.max_mismatch]:
            # 앞에서 몇 번째 토큰부터 갈라지는지
            i = next((j for j, (x, y) in enumerate(zip(ref[k], cur[k])) if x != y), None)
            print(f"    {k}: 첫 불일치 위치 {i}, {base}={ref[k]} {c}={cur[k]}")
        if len(bad) > a.max_mismatch:
            print(f"    ... 외 {len(bad) - a.max_mismatch}건")
