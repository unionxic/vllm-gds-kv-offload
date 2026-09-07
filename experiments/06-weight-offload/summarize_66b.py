"""results/weight-offload/opt66b/*.json → arm별 표(중앙값) + 토큰 일치 확인."""
import glob, json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, "..", "..", "results", "weight-offload", "opt66b")
rows = {}
ref_ids = None; mism = []
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    d = json.load(open(f)); tag = os.path.basename(f)[:-5]
    a = d.get("args", {})
    arm = f"{a.get('transport')} h{a.get('host_fraction')} step{a.get('prefetch_step')} thr{a.get('io_threads')} ring{a.get('ring_mb') or 0}" + (" nsys" if a.get("nsys") else "")
    rows.setdefault(arm, []).append(d)
    ids = d.get("ids")
    if ref_ids is None: ref_ids = ids; ref_tag = tag
    elif ids != ref_ids: mism.append(tag)
def med(rs, k): 
    v = [r[k] for r in rs if k in r and r[k] is not None]; return f"{st.median(v):.2f}" if v else "-"
print("| arm | n | load s | prefill s | decode step s | tok/s | cpu s | GPU max GiB | RSS GiB | nvfs reads | nvfs GiB |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for arm, rs in rows.items():
    nv = [r.get("nvfs_delta") or {} for r in rs]
    nreads = f"{st.median([x['Reads.n'] for x in nv if 'Reads.n' in x]):.0f}" if any('Reads.n' in x for x in nv) else "-"
    ngib = f"{st.median([x['Reads.readMiB']/1024 for x in nv if 'Reads.readMiB' in x]):.0f}" if any('Reads.readMiB' in x for x in nv) else "-"
    rss = (f"{st.median([r['mem_end']['VmRSS']/2**30 for r in rs if 'mem_end' in r]):.1f}" if any('mem_end' in r for r in rs) else '-')
    print(f"| {arm} | {len(rs)} | {med(rs,'load_s')} | {med(rs,'prefill_batch_s')} | {med(rs,'decode_step_s')} | {med(rs,'out_tok_per_s')} | {med(rs,'cpu_s')} | {med(rs,'gpu_max_gib')} | {rss} | {nreads} | {ngib} |")
print("token match vs", ref_tag, ":", "ALL OK" if not mism else f"MISMATCH {mism}")
