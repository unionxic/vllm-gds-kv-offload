"""results/smoke-remote-tier 점검. (1) 조건 none(재계산) 대비 cufile 조건들의 출력 토큰열 일치,
(2) 조건별 표: 저장 라운드(cold_fill)·적중 라운드(reverse_retrieve) wall clock과 prefill 시간,
    kv_io의 쓰기/읽기 구간 분해와 전송량, 커넥터 통계(kv_manager).
usage: python tools/check_smoke_remote_tier.py [--dir results/smoke-remote-tier] [--ref none]"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/smoke-remote-tier")
ap.add_argument("--ref", default="none", help="출력 토큰열 기준 조건")
ap.add_argument("--order", default="none,local-ssd,remote-dram,remote-ssd")
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
def phase_prefill(st, ph):
    """해당 phase에서 forward로 잡힌 step(0.3 s 초과) 중 prefill 몫의 합과 개수."""
    v = [x["dur"] for x in st if x["phase"] == ph and x["dur"] > 0.3 and x["kind"] == "prefill"]
    return round(sum(v), 2), len(v)


print()
print("== 조건별 (cold_fill=저장 라운드, reverse_retrieve=적중 라운드)")
hdr = ("조건", "KV 위치", "fstype", "cold_fill s", "cf prefill s", "reverse s", "rv prefill s",
       "rv ttft중앙 s", "적중토큰/요청", "write GiB", "read GiB", "w ev/open/io/fin s", "r open/io/fin s",
       "w/r calls", "kv files", "err")
print(" | ".join(hdr))
for c in conds:
    r, st, rq = R[c]
    k = r.get("kv_io") or {}
    ws = k.get("write_stages_s") or {}; rs = k.get("read_stages_s") or {}
    fs = r.get("kv_root_fs") or {}
    mnt = fs.get("mount") or os.path.dirname(r.get("kv_root") or r["args"]["kv_root"])
    cf, cfn = phase_prefill(st, "cold_fill"); rv, rvn = phase_prefill(st, "reverse_retrieve")
    ph = r["phases"]; tt = sorted(ph["reverse_retrieve"]["ttft"])
    # 적중 토큰은 요청별 lookup 최댓값의 합(requests.jsonl의 matched_of). 없으면 phase 누적값(lookup 호출 합, 부풀음)으로 대체.
    rv_req = [x for x in rq if x["phase"] == "reverse_retrieve"]
    if rv_req and "matched_of" in rv_req[0]:
        hit_tok = sum(x.get("matched_of", 0) for x in rv_req); hit_req = sum(1 for x in rv_req if x.get("matched_of", 0) > 0)
        hit_s = f"{hit_tok:7d}/{hit_req}건"
    else:
        hit_s = f"{ph['reverse_retrieve']['matched']:7d}(누적)"
    print(" | ".join([
        f"{c:11}", f"{mnt:18}", f"{fs.get('fstype','-'):5}",
        f"{ph['cold_fill']['wall_s']:8.1f}", f"{cf:8.1f}({cfn})",
        f"{ph['reverse_retrieve']['wall_s']:8.1f}", f"{rv:8.1f}({rvn})",
        f"{(tt[len(tt)//2] if tt else 0):8.2f}", hit_s,
        f"{k.get('write_gib', 0):6.2f}", f"{k.get('read_gib', 0):6.2f}",
        "/".join(f"{ws.get(x, 0):.1f}" for x in ("ev", "open", "io", "fin")),
        "/".join(f"{rs.get(x, 0):.1f}" for x in ("open", "io", "fin")),
        f"{k.get('write_calls', 0)}/{k.get('read_calls', 0)}",
        f"{r.get('kv_files', 0)}", f"{k.get('errors', 0)}"]))

print()
print("== 커넥터 통계(kv_manager)")
for c in conds:
    m = R[c][0].get("kv_manager") or {}
    if not m: print(f"  {c:12} 없음"); continue
    print(f"  {c:12} " + ", ".join(f"{x}={m[x]}" for x in
          ("files", "pending", "total_gib", "lookup_hit", "lookup_miss", "evicted", "refused") if x in m)
          + (f"  (디스크 파일 {R[c][0].get('kv_files')}개: files+pending과 대조)" if R[c][0].get("kv_files") else ""))

print()
print("== 전송률(구간 io 시간 기준, 스레드 시간 합이므로 참고값)")
for c in conds:
    k = R[c][0].get("kv_io") or {}
    ws = (k.get("write_stages_s") or {}).get("io", 0); rs = (k.get("read_stages_s") or {}).get("io", 0)
    w = f"{k.get('write_gib', 0) / ws:.2f}" if ws else "-"
    rd = f"{k.get('read_gib', 0) / rs:.2f}" if rs else "-"
    print(f"  {c:12} write {w} GiB/s, read {rd} GiB/s")
