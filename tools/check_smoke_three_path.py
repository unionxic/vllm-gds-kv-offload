"""results/smoke-three-path 점검. check_smoke_remote_tier.py와 같은 형식에 세 경로 배치용 열을 더한다.
(1) 조건 none(재계산) 대비 나머지 조건의 출력 토큰열 일치,
(2) 조건별 표: 저장 라운드(cold_fill)·적중 라운드(reverse_retrieve) wall clock과 prefill 시간,
    cuFile 쓰기/읽기 전송량과 구간 시간,
(3) 루트별 파일 수·용량(kv_files_by_root)과 매니저가 센 루트별 매핑 횟수(kv_manager.roots),
(4) hybrid 워커의 티어별 하위 job 수(sub_jobs_host / sub_jobs_ssd)와 배치 정책.
usage: python tools/check_smoke_three_path.py [--dir results/smoke-three-path] [--ref none]"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/smoke-three-path")
ap.add_argument("--ref", default="none", help="출력 토큰열 기준 조건")
ap.add_argument("--order", default="none,host-first,ratio,multi-root")
a = ap.parse_args()

D = os.path.abspath(a.dir)
conds = [c for c in a.order.split(",") if os.path.exists(os.path.join(D, c, "result.json"))]
conds += sorted(c for c in os.listdir(D) if c not in conds and os.path.exists(os.path.join(D, c, "result.json")))
if not conds:
    sys.exit(f"result.json이 있는 조건이 없음: {D}")


def load(c):
    r = json.load(open(os.path.join(D, c, "result.json")))
    st = [json.loads(l) for l in open(os.path.join(D, c, "steps.jsonl"))] if os.path.exists(os.path.join(D, c, "steps.jsonl")) else []
    rq = [json.loads(l) for l in open(os.path.join(D, c, "requests.jsonl"))] if os.path.exists(os.path.join(D, c, "requests.jsonl")) else []
    return r, st, rq


R = {c: load(c) for c in conds}


def phase_prefill(st, ph):
    """해당 phase에서 forward로 잡힌 step(0.3 s 초과) 중 prefill 몫의 합과 개수."""
    v = [x["dur"] for x in st if x["phase"] == ph and x["dur"] > 0.3 and x["kind"] == "prefill"]
    return round(sum(v), 2), len(v)


def short(path):
    """루트 경로를 마운트 지점 + 마지막 폴더로 줄인다."""
    return "/".join(p for p in path.rstrip("/").split("/")[-2:])


# ---- 출력 토큰열 대조 ----
print(f"== 출력 토큰열 대조 (기준 조건: {a.ref})")
if a.ref not in R:
    print(f"  기준 조건 {a.ref} 결과 없음 — 건너뜀")
else:
    ref = {(x["phase"], x["rid"]): x.get("ids") for x in R[a.ref][2]}
    for c in conds:
        if c == a.ref: continue
        cur = {(x["phase"], x["rid"]): x.get("ids") for x in R[c][2]}
        miss = sorted(set(ref) - set(cur)); extra = sorted(set(cur) - set(ref))
        bad = [k for k in sorted(set(ref) & set(cur)) if ref[k] != cur[k]]
        ok = not (miss or extra or bad)
        print(f"  {c:12} {'일치' if ok else '불일치'}  요청 {len(set(ref) & set(cur))}건 비교"
              + (f", 누락 {len(miss)}, 초과 {len(extra)}, 다름 {len(bad)}" if not ok else ""))
        for k in bad[:5]:
            print(f"    {k[0]}/{k[1]}: ref={ref[k]} cur={cur[k]}")

# ---- 조건별 표 ----
print()
print("== 조건별 (cold_fill=저장 라운드, reverse_retrieve=적중 라운드)")
hdr = ("조건", "transport", "배치", "host몫", "cold_fill s", "cf prefill s", "reverse s", "rv prefill s",
       "rv ttft중앙 s", "적중토큰/요청", "write GiB", "read GiB", "w ev/open/io/fin s", "r open/io/fin s",
       "w/r calls", "kv files", "err")
print(" | ".join(hdr))
for c in conds:
    r, st, rq = R[c]
    k = r.get("kv_io") or {}
    ws = k.get("write_stages_s") or {}; rs = k.get("read_stages_s") or {}
    m = r.get("kv_manager") or {}
    cf, cfn = phase_prefill(st, "cold_fill"); rv, rvn = phase_prefill(st, "reverse_retrieve")
    ph = r["phases"]; tt = sorted(ph["reverse_retrieve"]["ttft"])
    rv_req = [x for x in rq if x["phase"] == "reverse_retrieve"]
    if rv_req and "matched_of" in rv_req[0]:
        hit_tok = sum(x.get("matched_of", 0) for x in rv_req); hit_req = sum(1 for x in rv_req if x.get("matched_of", 0) > 0)
        hit_s = f"{hit_tok:7d}/{hit_req}건"
    else:
        hit_s = f"{ph['reverse_retrieve']['matched']:7d}(누적)"
    share = m.get("host_share")
    print(" | ".join([
        f"{c:11}", f"{r['args']['kv_transport']:9}", f"{str(m.get('placement', '-')):10}",
        f"{('-' if share is None else f'{share:.2f}'):6}",
        f"{ph['cold_fill']['wall_s']:8.1f}", f"{cf:8.1f}({cfn})",
        f"{ph['reverse_retrieve']['wall_s']:8.1f}", f"{rv:8.1f}({rvn})",
        f"{(tt[len(tt)//2] if tt else 0):8.2f}", hit_s,
        f"{k.get('write_gib', 0):6.2f}", f"{k.get('read_gib', 0):6.2f}",
        "/".join(f"{ws.get(x, 0):.1f}" for x in ("ev", "open", "io", "fin")),
        "/".join(f"{rs.get(x, 0):.1f}" for x in ("open", "io", "fin")),
        f"{k.get('write_calls', 0)}/{k.get('read_calls', 0)}",
        f"{r.get('kv_files', 0)}", f"{k.get('errors', 0)}"]))

# ---- 루트별 배치 ----
print()
print("== 루트별 파일(kv_files_by_root: 런 끝 시점 디스크 실측, mapped: 매퍼 호출 수)")
for c in conds:
    r = R[c][0]
    by = r.get("kv_files_by_root") or []
    if not by:
        print(f"  {c:12} 없음"); continue
    mgr = r.get("kv_manager") or {}
    roots = (mgr.get("ssd_manager") or mgr).get("roots") or []
    mapped = {x["root"]: x["mapped"] for x in roots}
    tot = sum(x["files"] for x in by) or 1
    for x in by:
        fs = next((q.get("fs") or {} for q in (r.get("kv_roots") or []) if q["dir"] == x["dir"]), {})
        mp = mapped.get(x["dir"])
        print(f"  {c:12} {short(x['dir']):22} w={x['weight']:<3} fstype={fs.get('fstype','-'):5} "
              f"files={x['files']:5} ({100.0 * x['files'] / tot:5.1f} %) {x['bytes_gib']:6.3f} GiB"
              + (f"  mapped={mp}" if mp is not None else ""))

# ---- host / 파일 티어 분담 ----
print()
print("== 티어 분담(hybrid 워커 하위 job 수, 매니저 적중·저장 수)")
for c in conds:
    r = R[c][0]
    w = r.get("kv_worker_hybrid") or {}
    m = r.get("kv_manager") or {}
    if not w and "placement" not in m:
        print(f"  {c:12} hybrid 아님"); continue
    print(f"  {c:12} sub_jobs_host={w.get('sub_jobs_host')} sub_jobs_ssd={w.get('sub_jobs_ssd')} "
          f"inflight={w.get('inflight')} | hits host/ssd={m.get('hits_host')}/{m.get('hits_ssd')} "
          f"miss={m.get('misses')} stores host/ssd={m.get('stores_host')}/{m.get('stores_ssd')} "
          f"host_blocks={m.get('host_blocks_used')}/{m.get('host_blocks_total')} "
          f"host_alloc_fail={m.get('host_alloc_fail')} write_through={m.get('write_through')}")

# ---- 전송률 ----
print()
print("== 전송률(구간 io 시간 기준, 스레드 시간 합이므로 참고값)")
for c in conds:
    k = R[c][0].get("kv_io") or {}
    ws = (k.get("write_stages_s") or {}).get("io", 0); rs = (k.get("read_stages_s") or {}).get("io", 0)
    w = f"{k.get('write_gib', 0) / ws:.2f}" if ws else "-"
    rd = f"{k.get('read_gib', 0) / rs:.2f}" if rs else "-"
    print(f"  {c:12} write {w} GiB/s, read {rd} GiB/s")
