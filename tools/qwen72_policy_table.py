"""results/qwen72b/bailian-ram0.5-* 조건별 요약 표. usage: qwen72_policy_table.py [tag ...]"""
import json, os, sys
D="results/qwen72b"; base="bailian-ram0.5-none"
def load(tag):
    R=f"{D}/{tag}"; r=json.load(open(f"{R}/result.json")); st=[json.loads(l) for l in open(f"{R}/steps.jsonl")]; rq=[json.loads(l) for l in open(f"{R}/requests.jsonl")]
    return r,st,rq
def ids(rq): return {(x["phase"], x["doc"]): tuple(x.get("ids") or ()) for x in rq}
tags=sys.argv[1:] or sorted(t for t in os.listdir(D) if t.startswith("bailian-ram0.5-") and os.path.exists(f"{D}/{t}/result.json"))
b=load(base); a=ids(b[2]); btot=sum(b[0]["phases"][p]["wall_s"] for p in ("cold_fill","reverse_retrieve"))
print("조건 | cold_fill s / fwd / prefill avg | reverse s / fwd / prefill avg | 합계 (재계산 대비) | 읽기 GiB | 쓰기 GiB | manager | 토큰열 불일치")
for t in tags:
    r,st,rq=load(t); k=r["kv_io"]; m=r.get("kv_manager") or {}; row=[]; tot=0
    for ph in ("cold_fill","reverse_retrieve"):
        p=r["phases"][ph]; s=[x for x in st if x["phase"]==ph and x["dur"]>0.3]; pf=[x["dur"] for x in s if x["kind"]=="prefill"]
        tot+=p["wall_s"]; row.append(f"{p['wall_s']:.0f} / {len(s)} / {sum(pf)/max(1,len(pf)):.0f}")
    bb=ids(rq); mism=sum(1 for kk in a if kk in bb and a[kk]!=bb[kk])
    mg=f"files {m.get('files')} {m.get('total_gib')} GiB, evicted {m.get('evicted')}, refused {m.get('refused')}, hit/miss {m.get('lookup_hit')}/{m.get('lookup_miss')}" + (f", adm {m['admission']['admit']}/{m['admission']['reject']}" if m.get('admission') else "")
    print(f"{t.replace('bailian-ram0.5-','')} | {row[0]} | {row[1]} | {tot:.0f} ({(tot/btot-1)*100:+.1f}%) | {k['read_gib']} | {k['write_gib']} | {mg} | {mism}/{len(bb)} err {k['errors']}")
