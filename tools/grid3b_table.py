"""results/grid3b 격자 표: 재계산 비율 K × host 비율 G × 모드. reverse_retrieve(적중 단계) wall clock과 prefill forward 평균, 적재·분할 통계."""
import json, os, sys
D="results/grid3b"
rows=[]
for t in sorted(os.listdir(D)):
    p=f"{D}/{t}/result.json"
    if not os.path.exists(p) or not t.startswith("k"): continue
    r=json.load(open(p)); st=[json.loads(l) for l in open(f"{D}/{t}/steps.jsonl")]
    rv=[x for x in st if x["phase"]=="reverse_retrieve" and x["dur"]>0.3]; pf=[x["dur"] for x in rv if x["kind"]=="prefill"]
    sp=r.get("kv_split") or {}; k=r["kv_io"]; m=r.get("kv_manager") or {}
    parts=t.split("-"); K=parts[0][1:]; G=parts[1][1:]; mode=parts[2]
    rows.append(dict(K=float(K), G=float(G), mode=mode, tag=t, reverse=r["phases"]["reverse_retrieve"]["wall_s"], cold=r["phases"]["cold_fill"]["wall_s"], fwd=len(rv), pf=sum(pf)/max(1,len(pf)),
        matched=r["phases"]["reverse_retrieve"]["matched"], split=sp.get("requests_split",0), wait=sp.get("tail_wait_s_total",0), read=k["read_gib"], err=k["errors"], hh=m.get("hits_host"), hs=m.get("hits_ssd")))
rows.sort(key=lambda x:(x["G"],x["K"],x["mode"]))
print("G(host) | K(재계산) | 모드 | reverse s | forward | prefill avg s | 적중 토큰 | 분할 요청 | 뒤 대기 s | SSD 읽기 GiB | host/ssd hit | err")
for x in rows:
    print(f"{x['G']:4} | {x['K']:4} | {x['mode']:6} | {x['reverse']:6.1f} | {x['fwd']:3d} | {x['pf']:5.2f} | {x['matched']:6d} | {x['split']:2d} | {x['wait']:5.1f} | {x['read']:5.2f} | {x['hh']}/{x['hs']} | {x['err']}")
