"""07 결과 표: results/combined/opt66b/*.json → 런별 round1(cold, KV 저장)/round2(SSD KV 적중) prefill·decode, KV I/O, 메모리."""
import glob, json, os
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "combined", "opt66b")
rows = []
for f in sorted(glob.glob(os.path.join(O, "*.json"))):
    d = json.load(open(f)); a = d["args"]; t = d["tiers"]
    r1, r2 = d["r1"], d["r2"]
    rows.append(dict(tag=d["args"]["tag"], hf=a["host_fraction"], kv=a["kv_transport"], load_s=d["load_s"],
        host=f'{t["host_tier_gib"]:.0f}G/{t["n_modules"]-t["n_ssd"]}L', ssd=f'{t["ssd_tier_gib"]:.0f}G/{t["n_ssd"]}L',
        r1_pre=r1["prefill_s"], r1_dec=r1["decode_step_s"], r2_pre=r2["prefill_s"], r2_dec=r2["decode_step_s"],
        r2_match=r2.get("matched", 0), r2_kvrd=f'{r2["kv_read_n"]}x/{r2["kv_read_b"]/2**30:.2f}G', r1_kvwr=f'{r1["kv_write_n"]}x/{r1["kv_write_b"]/2**30:.2f}G',
        w_ssd=f'{d.get("weight_ssd_stats",{}).get("read_gib",0):.0f}G', ids_eq=d["ids_equal_rounds"],
        avail_end=f'{d["sysmem_end"]["MemAvailable"]/2**30:.1f}G', rss=f'{d["mem_end"]["VmRSS"]/2**30:.1f}G'))
cols = ["tag","load_s","host","ssd","r1_pre","r1_dec","r2_pre","r2_dec","r2_match","r2_kvrd","r1_kvwr","w_ssd","ids_eq","avail_end","rss"]
if not rows: raise SystemExit("no results yet")
w = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
print(" | ".join(c.ljust(w[c]) for c in cols)); print("-+-".join("-"*w[c] for c in cols))
for r in rows: print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))
