"""results/phase3-* 점검. 성능 비교는 합의한 기준선(b1-tiering)과 우리 조건(ours-ratio) 두 행만 쓴다.
조건 ref-none(재계산)은 출력 토큰열 정합성 기준일 뿐이어서 성능 표에 넣지 않는다.
(1) ref-none 대비 나머지 조건의 출력 토큰열 일치,
(2) 비교 표: 저장 라운드(cold_fill)·적중 라운드(reverse_retrieve) wall clock과 한 사이클 합,
    적중 라운드 TTFT 중앙·p95, 적중 요청 수와 적중 토큰(요청별 matched_of), GPU 최대 사용량,
(3) 티어별 전송·잔존: ours는 cuFile 계측(kv_io)과 루트별 파일(kv_files_by_root),
    b1은 fs 티어 루트의 파일 실측(전송량 계측 없음),
(4) 한 사이클 합과 적중 TTFT의 비(ours/b1).
usage: python tools/check_phase3.py [--dir results/phase3-8k-host8] [--ref ref-none]"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/phase3-8k-host8")
ap.add_argument("--ref", default="ref-none", help="출력 토큰열 정합성 기준 조건(성능 비교에는 쓰지 않음)")
ap.add_argument("--base", default="b1-tiering", help="성능 비교 기준선 조건")
ap.add_argument("--ours", default="ours-ratio", help="우리 조건")
a = ap.parse_args()

D = os.path.abspath(a.dir)
if not os.path.isdir(D):
    sys.exit(f"결과 폴더가 없음: {D}")
conds = [c for c in (a.ref, a.base, a.ours) if os.path.exists(os.path.join(D, c, "result.json"))]
conds += sorted(c for c in os.listdir(D) if c not in conds and os.path.exists(os.path.join(D, c, "result.json")))
if not conds:
    sys.exit(f"result.json이 있는 조건이 없음: {D}")


def load(c):
    """check_smoke_three_path.py와 같은 적재 함수(result.json·steps.jsonl·requests.jsonl)."""
    r = json.load(open(os.path.join(D, c, "result.json")))
    st = [json.loads(l) for l in open(os.path.join(D, c, "steps.jsonl"))] if os.path.exists(os.path.join(D, c, "steps.jsonl")) else []
    rq = [json.loads(l) for l in open(os.path.join(D, c, "requests.jsonl"))] if os.path.exists(os.path.join(D, c, "requests.jsonl")) else []
    return r, st, rq


R = {c: load(c) for c in conds}


def phase_prefill(st, ph):
    """해당 phase에서 forward로 잡힌 step(0.3 s 초과) 중 prefill 몫의 합과 개수. check_smoke_three_path.py와 같음."""
    v = [x["dur"] for x in st if x["phase"] == ph and x["dur"] > 0.3 and x["kind"] == "prefill"]
    return round(sum(v), 2), len(v)


def short(path):
    """루트 경로를 마운트 지점 + 마지막 폴더로 줄인다. check_smoke_three_path.py와 같음."""
    return "/".join(p for p in path.rstrip("/").split("/")[-2:])


def pct(v, q):
    """정렬된 목록의 백분위(선형 보간 없이 가장 가까운 인덱스)."""
    if not v: return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(round((len(s) - 1) * q)))]


def hits(rq, ph):
    """해당 phase의 적중 토큰 합과 적중 요청 수(요청별 matched_of 기준)."""
    v = [x for x in rq if x["phase"] == ph]
    if not v or "matched_of" not in v[0]:
        return None, None
    return sum(x.get("matched_of", 0) for x in v), sum(1 for x in v if x.get("matched_of", 0) > 0)


def cycle(c):
    """한 사이클(저장 라운드 + 적중 라운드) 지표 묶음."""
    r, st, rq = R[c]
    ph = r["phases"]
    cf_w = ph["cold_fill"]["wall_s"]; rv_w = ph["reverse_retrieve"]["wall_s"]
    tt = ph["reverse_retrieve"]["ttft"]
    hit_tok, hit_req = hits(rq, "reverse_retrieve")
    k = r.get("kv_io") or {}
    return dict(cond=c, transport=r["args"]["kv_transport"], placement=(r.get("kv_manager") or {}).get("placement", "-"),
                host_gb=r["args"].get("kv_host_gb"), cf=cf_w, rv=rv_w, cycle=round(cf_w + rv_w, 1),
                cf_pf=phase_prefill(st, "cold_fill")[0], rv_pf=phase_prefill(st, "reverse_retrieve")[0],
                ttft_med=pct(tt, 0.5), ttft_p95=pct(tt, 0.95),
                hit_tok=hit_tok, hit_req=hit_req, n_req=len([x for x in rq if x["phase"] == "reverse_retrieve"]),
                w_gib=k.get("write_gib"), r_gib=k.get("read_gib"),
                files=r.get("kv_files"), files_gib=r.get("kv_bytes_gib"), gpu=r.get("gpu_max_gib"))


# ---- 출력 토큰열 대조 ----
print(f"== 출력 토큰열 대조 (정합성 기준 조건: {a.ref}, 성능 비교에는 쓰지 않음)")
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

# ---- 성능 비교 표(기준선과 우리 조건만) ----
rows = [cycle(c) for c in (a.base, a.ours) if c in R]
print()
print("== 성능 비교 (cold_fill=저장 라운드, reverse_retrieve=적중 라운드). 재계산 조건은 행에 넣지 않음")
if not rows:
    print(f"  비교할 조건이 없음(기준선 {a.base}, 우리 조건 {a.ours})")
else:
    hdr = ("조건", "transport", "배치", "host GB", "저장 라운드 s", "저장 prefill s", "적중 라운드 s",
           "적중 prefill s", "한 사이클 s", "적중 ttft중앙 s", "적중 ttft p95 s", "적중 요청", "적중 토큰",
           "cuFile write GiB", "cuFile read GiB", "파일 잔존 GiB", "파일 수", "GPU max GiB")
    print(" | ".join(hdr))
    for x in rows:
        hit_req = "-" if x["hit_req"] is None else "{}/{}".format(x["hit_req"], x["n_req"])
        hit_tok = "-" if x["hit_tok"] is None else str(x["hit_tok"])
        w_gib = "-" if not x["w_gib"] else "{:.2f}".format(x["w_gib"])
        r_gib = "-" if not x["r_gib"] else "{:.2f}".format(x["r_gib"])
        print(" | ".join([
            f"{x['cond']:11}", f"{x['transport']:9}", f"{str(x['placement']):10}", f"{x['host_gb']}",
            f"{x['cf']:9.1f}", f"{x['cf_pf']:9.1f}", f"{x['rv']:9.1f}", f"{x['rv_pf']:9.1f}",
            f"{x['cycle']:9.1f}", f"{x['ttft_med']:9.2f}", f"{x['ttft_p95']:9.2f}",
            f"{hit_req:>7}", f"{hit_tok:>9}", f"{w_gib:>7}", f"{r_gib:>7}",
            f"{x['files_gib']}", f"{x['files']}", f"{x['gpu']}"]))

# ---- 비(ours/b1) ----
if len(rows) == 2:
    b, o = rows[0], rows[1]
    print()
    print("== 비(ours/b1)")
    print(f"  한 사이클 합      {o['cycle']:.1f} / {b['cycle']:.1f} = {o['cycle'] / b['cycle']:.3f}")
    print(f"  저장 라운드       {o['cf']:.1f} / {b['cf']:.1f} = {o['cf'] / b['cf']:.3f}")
    print(f"  적중 라운드       {o['rv']:.1f} / {b['rv']:.1f} = {o['rv'] / b['rv']:.3f}")
    if b["ttft_med"]:
        print(f"  적중 ttft 중앙    {o['ttft_med']:.2f} / {b['ttft_med']:.2f} = {o['ttft_med'] / b['ttft_med']:.3f}")
    if b["ttft_p95"]:
        print(f"  적중 ttft p95     {o['ttft_p95']:.2f} / {b['ttft_p95']:.2f} = {o['ttft_p95'] / b['ttft_p95']:.3f}")

# ---- 루트별 파일 ----
print()
print("== 루트별 파일(kv_files_by_root: 런 끝 시점 디스크 실측, mapped: 매퍼 호출 수)")
for c in conds:
    if c == a.ref: continue
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
              f"files={x['files']:6} ({100.0 * x['files'] / tot:5.1f} %) {x['bytes_gib']:7.3f} GiB"
              + (f"  mapped={mp}" if mp is not None else ""))

# ---- 티어 분담 ----
print()
print("== 티어 분담(ours: hybrid 워커·매니저 계측. b1: in-tree tiering은 같은 계측이 없어 파일 실측만)")
for c in conds:
    if c == a.ref: continue
    r = R[c][0]
    w = r.get("kv_worker_hybrid") or {}
    m = r.get("kv_manager") or {}
    if not w and "placement" not in m:
        ex = r.get("kv_extra_config") or {}
        tiers = ", ".join(f"{t.get('type')}:{short(str(t.get('root_dir')))}" for t in (ex.get("secondary_tiers") or []))
        print(f"  {c:12} spec={ex.get('spec_name')} cpu_bytes={ex.get('cpu_bytes_to_use')} 2차티어=[{tiers}] "
              f"(전송량 계측 없음, 파일 실측 {r.get('kv_bytes_gib')} GiB / {r.get('kv_files')}개)")
        continue
    print(f"  {c:12} sub_jobs_host={w.get('sub_jobs_host')} sub_jobs_ssd={w.get('sub_jobs_ssd')} "
          f"inflight={w.get('inflight')} | hits host/ssd={m.get('hits_host')}/{m.get('hits_ssd')} "
          f"miss={m.get('misses')} stores host/ssd={m.get('stores_host')}/{m.get('stores_ssd')} "
          f"host_blocks={m.get('host_blocks_used')}/{m.get('host_blocks_total')} "
          f"host_alloc_fail={m.get('host_alloc_fail')} write_through={m.get('write_through')}")

# ---- 전송률 ----
print()
print("== 전송률(ours 전용, 구간 io 시간 기준. 스레드 시간 합이므로 참고값)")
for c in conds:
    if c == a.ref: continue
    k = R[c][0].get("kv_io") or {}
    ws = (k.get("write_stages_s") or {}).get("io", 0); rs = (k.get("read_stages_s") or {}).get("io", 0)
    w = f"{k.get('write_gib', 0) / ws:.2f} GiB/s" if ws else "-"
    rd = f"{k.get('read_gib', 0) / rs:.2f} GiB/s" if rs else "-"
    print(f"  {c:12} write {w}, read {rd}")
