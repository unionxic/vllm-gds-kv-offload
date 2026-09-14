"""nsys sqlite(NVTX_EVENTS)에서 forward(step)마다 가중치 cuFileRead, KV cuFileWrite/Read의 합집합 시간과 겹침, ssd_window 표시 길이를 표로. usage: nsys_overlap.py timeline.sqlite PHASE"""
import sqlite3, sys
def load(db):
    c=sqlite3.connect(db)
    q="select e.start, e.end, coalesce(s.value, e.text) as name, e.globalTid from NVTX_EVENTS e left join StringIds s on e.textId=s.id"
    return [(a,b,n,t) for a,b,n,t in c.execute(q)]
def union(iv):
    iv=sorted(iv); out=[]
    for a,b in iv:
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
def inter(u1,u2):
    i=j=0; tot=0
    while i<len(u1) and j<len(u2):
        a=max(u1[i][0],u2[j][0]); b=min(u1[i][1],u2[j][1])
        if b>a: tot+=b-a
        if u1[i][1]<u2[j][1]: i+=1
        else: j+=1
    return tot
ev=load(sys.argv[1]); phase=sys.argv[2]
steps=sorted([(a,b) for a,b,n,t in ev if n==f"step:{phase}"])
kvfiles=[(a,b) for a,b,n,t in ev if n in ("kv_store_file","kv_load_file")]
def inside(x, ivs): return any(a<=x[0] and x[1]<=b for a,b in ivs)
reads=[(a,b) for a,b,n,t in ev if n=="cuFileRead" and b]; writes=[(a,b) for a,b,n,t in ev if n=="cuFileWrite" and b]
kvthreads={t for a,b,n,t in ev if n in ("kv_store_file","kv_load_file")}
wreads=[(a,b) for a,b,n,t in ev if n=="cuFileRead" and b and t not in kvthreads]; kreads=[(a,b) for a,b,n,t in ev if n=="cuFileRead" and b and t in kvthreads]
marks=sorted([(a,n) for a,b,n,t in ev if n in ("ssd_window:on","ssd_window:off")])
print(f"{phase}: steps {len(steps)}, weight cuFileRead {len(wreads)}, kv cuFileRead {len(kreads)}, cuFileWrite {len(writes)}, ssd_window marks {len(marks)}")
print("step  dur_s  wRead_union_s  kvWrite_union_s  overlap(write∩wRead)_s  kvRead_union_s  overlap(kvRead∩wRead)_s  ssd_window_s")
for i,(s0,s1) in enumerate(steps):
    clip=lambda ivs: union([[max(a,s0),min(b,s1)] for a,b in ivs if b>s0 and a<s1])
    ur=clip(wreads); uw=clip(writes); uk=clip(kreads)
    # ssd window from marks
    win=[]; cur=None
    for a,n in marks:
        if n=="ssd_window:on": cur=a
        elif cur is not None: win.append((cur,a)); cur=None
    uwin=clip(win)
    f=lambda u: sum(b-a for a,b in u)/1e9
    print(f"{i:4d} {(s1-s0)/1e9:6.1f} {f(ur):13.1f} {f(uw):16.1f} {inter(uw,ur)/1e9:23.1f} {f(uk):15.1f} {inter(uk,ur)/1e9:25.1f} {f(uwin):12.1f}")
