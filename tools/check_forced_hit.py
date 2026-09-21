"""강제 적중(forced_hit) 캠페인 점검. 저장 라운드(cold_fill)와 재생 라운드(replay)를 나눠 본다.
재생 라운드는 저장을 막고 GPU 프리픽스 캐시를 비운 구간이라, 남는 차이는 적재 경로뿐이다.
(1) 조건별 표: 적중 요청 수, TTFT, 적재 완료 시간, 적재 바이트, decode 시간의 중앙값·p95,
    재생 구간의 CPU 사용률(vmstat us+sy 평균),
(2) 짝지은 비교: 두 조건 모두 적중한 같은 문서의 요청만 골라 TTFT 차(ours - mooncake)의 중앙값과 부호 수,
(3) 출력 토큰열 대조와 저장 커밋 확인 결과.
usage: python tools/check_forced_hit.py [--dir results/forced-hit-8b] [--order mooncake,ours-ratio]"""
import argparse, json, os, sys
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/forced-hit-8b")
ap.add_argument("--order", default="mooncake,ours-ratio")
ap.add_argument("--phase", default="replay", help="적재 경로를 보는 구간")
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


def jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass  # 런이 끊겨 마지막 줄이 잘린 경우
    return out


def q(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def cpu_window(run_dir, t0_ns, t1_ns):
    """vmstat의 us+sy 평균(%)과 표본 수. 창은 wall_ns 두 개(엔진 phase 경계)."""
    f = os.path.join(run_dir, "vmstat.log")
    if not (os.path.exists(f) and t0_ns and t1_ns):
        return None, 0
    lo = datetime.fromtimestamp(t0_ns / 1e9)
    hi = datetime.fromtimestamp(t1_ns / 1e9)
    vals = []
    for line in open(f):
        p = line.split()
        if len(p) < 19 or not p[0].isdigit():
            continue
        try:
            ts = datetime.strptime(p[-2] + " " + p[-1], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if lo <= ts <= hi:
            vals.append(int(p[12]) + int(p[13]))  # us, sy
    return (sum(vals) / len(vals) if vals else None), len(vals)


def phase_window(run_dir, phase):
    """events.jsonl의 phase 마커에서 (시작 wall_ns, 끝 wall_ns)."""
    t0 = t1 = None
    for e in jsonl(os.path.join(run_dir, "events.jsonl")):
        if e.get("kind") != "phase":
            continue
        if e.get("name") == phase: t0 = e["wall_ns"]
        elif e.get("name") == phase + "_end": t1 = e["wall_ns"]
    return t0, t1


def attach(trace_reqs, rids):
    """커넥터 쪽 request_id(접미사가 붙을 수 있음)를 러너 rid로 되돌린다."""
    m = {}
    for tr in trace_reqs:
        if tr in rids:
            m[tr] = tr; continue
        for r in rids:
            if tr.startswith(r + "-"):
                m[tr] = r; break
    return m


def load_cond(c):
    run = os.path.join(D, c)
    res = json.load(open(os.path.join(run, "result.json")))
    reqs = [x for x in jsonl(os.path.join(run, "requests.jsonl"))]
    tr = jsonl(os.path.join(run, "kvtrace.jsonl"))
    return run, res, reqs, tr


R = {c: load_cond(c) for c in conds}


def per_request(c):
    """요청별 지표. rid -> dict(doc, ttft, decode, matched, load_s, load_bytes, io_s)"""
    run, res, reqs, tr = R[c]
    rows = {}
    for x in reqs:
        if x.get("phase") != a.phase:
            continue
        rid = x["rid"]
        rows[rid] = dict(
            doc=x.get("doc"), tokens=x.get("tokens"), matched=x.get("matched_of", 0),
            gpu_cached=x.get("gpu_cached", 0), ids=x.get("ids"),
            ttft=(x["first_mono"] - x["submit_mono"]) if x.get("first_mono") else None,
            decode=(x["finish_mono"] - x["first_mono"]) if (x.get("first_mono") and x.get("finish_mono")) else None,
            load_s=None, load_bytes=0, io_s=0.0, n_host=0, n_ssd=0)
    rids = set(rows)
    treqs = {e.get("req") for e in tr if e.get("req")}
    back = attach(treqs, rids)
    # 적재 제출 -> 완료. 파일 티어는 load_prep/load_done, mooncake는 mc_load_enq/mc_load_done.
    start, done = {}, {}
    for e in tr:
        if e.get("ph") != a.phase:
            continue
        rid = back.get(e.get("req"))
        if rid is None or rid not in rows:
            continue
        ev = e["ev"]
        if ev in ("load_prep", "mc_load_enq"):
            start.setdefault(rid, e["t"])
            if ev == "load_prep":
                rows[rid]["load_bytes"] += e.get("bytes", 0) or 0
                rows[rid]["n_host"] += e.get("n_host", 0) or 0
                rows[rid]["n_ssd"] += e.get("n_ssd", 0) or 0
        elif ev in ("load_done", "mc_load_done"):
            done[rid] = e["t"]
            if ev == "mc_load_done":
                rows[rid]["load_bytes"] += e.get("bytes", 0) or 0
                rows[rid]["io_s"] += e.get("io_s", 0.0) or 0.0
        elif ev in ("r_end",):
            pass  # job 단위 시간은 조건별 요약에서 따로 본다
    for rid in rows:
        if rid in start and rid in done:
            rows[rid]["load_s"] = done[rid] - start[rid]
    # 파일 티어의 순수 IO 시간은 job 단위(r_submit/r_end)로만 나온다. 요청별로 나누지 않고 합만 본다.
    return rows


P = {c: per_request(c) for c in conds}

print(f"== 강제 적중 조건별 ({D}, 구간 {a.phase})")
hdr = ("조건", "transport", "요청", "적중요청", "적중토큰중앙",
       "TTFT 중앙/p95 s", "적재완료 중앙/p95 s", "적재 MiB 중앙", "decode 중앙/p95 s", "CPU us+sy %")
print(" | ".join(hdr))
for c in conds:
    run, res, reqs, tr = R[c]
    rows = P[c]
    hit = [v for v in rows.values() if v["matched"] > 0]
    tt = [v["ttft"] for v in hit if v["ttft"] is not None]
    ls = [v["load_s"] for v in hit if v["load_s"] is not None]
    lb = [v["load_bytes"] / 2**20 for v in hit if v["load_bytes"]]
    dc = [v["decode"] for v in hit if v["decode"] is not None]
    t0, t1 = phase_window(run, a.phase)
    cpu, n_s = cpu_window(run, t0, t1)
    print(" | ".join([
        f"{c:11}", f"{res['args']['kv_transport']:9}", f"{len(rows):4}", f"{len(hit):5}",
        f"{q([v['matched'] for v in rows.values()], 0.5):9.0f}",
        f"{q(tt,0.5):7.2f}/{q(tt,0.95):7.2f}",
        f"{q(ls,0.5):8.2f}/{q(ls,0.95):8.2f}",
        f"{q(lb,0.5):10.0f}",
        f"{q(dc,0.5):7.2f}/{q(dc,0.95):7.2f}",
        f"{'-' if cpu is None else format(cpu, '6.1f')}({n_s})"]))

print()
print(f"== 적재 티어 나눔과 전송 job({a.phase})")
for c in conds:
    run, res, reqs, tr = R[c]
    rows = P[c]
    nh = sum(v["n_host"] for v in rows.values()); ns = sum(v["n_ssd"] for v in rows.values())
    jobs = [e for e in tr if e.get("ph") == a.phase and e.get("ev") in ("r_end", "mc_get")]
    dur = [e.get("dur_s", 0.0) or 0.0 for e in jobs]
    byt = sum(e.get("bytes", 0) or 0 for e in jobs)
    tier = f"host 청크 {nh} / ssd 청크 {ns}" if (nh or ns) else "티어 나눔 없음(단일 풀)"
    print(f"  {c:11} {tier} | 전송 job {len(jobs)}건, 합 {byt/2**30:.2f} GiB, "
          f"job 시간 중앙 {q(dur,0.5):.2f} s / p95 {q(dur,0.95):.2f} s / 최대 {max(dur) if dur else 0:.2f} s")

print()
print("== 구간 합계(저장 라운드 포함, result.json)")
for c in conds:
    run, res, reqs, tr = R[c]
    ph = res.get("phases") or {}
    k = res.get("kv_io") or {}
    line = f"  {c:11} "
    line += " ".join(f"{n} wall {v.get('wall_s')} s(적중요청 {v.get('hit_requests')})" for n, v in ph.items())
    if k:
        line += f" | cuFile read {k.get('read_gib',0):.2f} GiB / write {k.get('write_gib',0):.2f} GiB"
        line += f", read_stages {k.get('read_stages_s')}"
    mc = res.get("kv_mooncake") or {}
    if mc and "error" not in mc:
        line += f" | Mooncake 풀 {mc.get('master_allocated_bytes',0)/2**30:.2f} GiB, key {mc.get('master_key_count',0):.0f}"
    print(line)
    cc = res.get("commit_check_cold_fill")
    if cc:
        print(f"  {c:11} 커밋 확인: {json.dumps(cc, ensure_ascii=False)}")
    tk = res.get("kv_trace")
    if tk:
        print(f"  {c:11} kvtrace: 행 {tk.get('rows')}, 저장 차단 호출 {tk.get('store_blocked_calls')}")

# ---- 짝지은 비교 ----
print()
print(f"== 짝지은 TTFT 차({a.phase}, 두 조건 모두 적중한 같은 문서만)")
if len(conds) < 2:
    print("  비교할 조건이 하나뿐")
else:
    base = conds[0]
    for c in conds[1:]:
        A, B = P[base], P[c]
        docA = {v["doc"]: v for v in A.values() if v["matched"] > 0 and v["ttft"] is not None}
        docB = {v["doc"]: v for v in B.values() if v["matched"] > 0 and v["ttft"] is not None}
        common = sorted(set(docA) & set(docB))
        d = [docB[x]["ttft"] - docA[x]["ttft"] for x in common]
        neg = sum(1 for x in d if x < 0); pos = sum(1 for x in d if x > 0)
        print(f"  {c} - {base}: 공통 적중 문서 {len(common)}건, 차 중앙 {q(d,0.5):+.3f} s, "
              f"p95 {q(d,0.95):+.3f} s, {c}가 빠른 건 {neg} / 느린 건 {pos}")
        dl = [docB[x]["load_s"] - docA[x]["load_s"] for x in common
              if docA[x]["load_s"] is not None and docB[x]["load_s"] is not None]
        if dl:
            print(f"  {c} - {base}: 적재 완료 시간 차 중앙 {q(dl,0.5):+.3f} s (표본 {len(dl)})")
        print(f"  문서별 TTFT({base} / {c}, s): " + ", ".join(
            f"{x}:{docA[x]['ttft']:.2f}/{docB[x]['ttft']:.2f}" for x in common[:12])
            + (" ..." if len(common) > 12 else ""))

# ---- 출력 토큰열 ----
print()
print("== 출력 토큰열 대조(조건 사이, 구간마다)")
phases = sorted({x.get("phase") for c in conds for x in R[c][2] if x.get("phase")})
if len(conds) < 2:
    print("  비교할 조건이 하나뿐")
else:
    base = conds[0]
    for ph in phases:
        ref = {x["rid"]: x.get("ids") for x in R[base][2] if x.get("phase") == ph}
        for c in conds[1:]:
            cur = {x["rid"]: x.get("ids") for x in R[c][2] if x.get("phase") == ph}
            both = sorted(set(ref) & set(cur))
            miss = sorted(set(ref) - set(cur)); extra = sorted(set(cur) - set(ref))
            bad = [k for k in both if ref[k] != cur[k]]
            ok = not (miss or extra or bad)
            print(f"  [{ph}] {base} vs {c}: {'일치' if ok else '불일치'} (요청 {len(both)}건 비교"
                  + (f", 누락 {len(miss)}, 초과 {len(extra)}, 다름 {len(bad)}" if not ok else "") + ")")
            for k in bad[:a.max_mismatch]:
                i = next((j for j, (x, y) in enumerate(zip(ref[k], cur[k])) if x != y), None)
                print(f"    {k}: 첫 불일치 위치 {i}, {base}={ref[k]} {c}={cur[k]}")
            if len(bad) > a.max_mismatch:
                print(f"    ... 외 {len(bad) - a.max_mismatch}건")
