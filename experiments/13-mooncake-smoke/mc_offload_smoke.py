"""sunny에서 rain Mooncake(DRAM 2 GB + SSD 오프로드)에 DRAM보다 큰 데이터를 넣고 다시 읽어
SSD 층 넘침이 동작하는지, 읽기가 맞는지, 어느 층에서 왔는지(master 메트릭) 확인한다."""
import os, sys, time, hashlib, urllib.request
from mooncake.store import MooncakeDistributedStore
IP="30.0.0.4"; MASTER="30.0.0.3:50051"; META="http://30.0.0.3:8080/metadata"; DEV="mlx5_0"
os.environ["MOONCAKE_REQUESTER_LOCAL_HOSTNAME"]=IP
N=int(sys.argv[1]) if len(sys.argv)>1 else 640   # 객체 수
SZ=8<<20                                          # 객체 8 MiB → 640개 = 5 GiB (> 2 GB DRAM)
def metrics():
    t=urllib.request.urlopen("http://30.0.0.3:9003/metrics",timeout=5).read().decode()
    return {l.split()[0]:l.split()[1] for l in t.splitlines() if l and not l.startswith("#") and len(l.split())==2}
st=MooncakeDistributedStore()
assert st.setup(IP, META, 0, 1<<28, "rdma", DEV, MASTER)==0
print("remove_all", st.remove_all())
vals=[]; t0=time.time()
for i in range(N):
    v=hashlib.sha256(str(i).encode()).digest()*(SZ//32)
    r=st.put(f"k{i}", v); time.sleep(float(os.environ.get("PUT_SLEEP","0")))
    if r!=0: print("put fail", i, r); break
    vals.append(hashlib.sha256(v).hexdigest())
dt=time.time()-t0; print(f"put {i+1} x {SZ>>20} MiB in {dt:.1f}s = {(i+1)*SZ/dt/1e9:.2f} GB/s")
time.sleep(15)  # 오프로드 하트비트(10 s) 지나도록
m=metrics(); print({k:v for k,v in m.items() if any(x in k for x in ("allocated","capacity","key_count","offload","disk","evict"))})
ok=0; bad=0; miss=0; t0=time.time(); nbytes=0
for i in range(N):
    b=st.get(f"k{i}")
    if not b: miss+=1; continue
    nbytes+=len(b)
    if hashlib.sha256(bytes(b)).hexdigest()==vals[i]: ok+=1
    else: bad+=1
dt=time.time()-t0; print(f"get ok={ok} bad={bad} miss={miss} in {dt:.1f}s = {nbytes/dt/1e9:.2f} GB/s")
m2=metrics(); print({k:v for k,v in m2.items() if any(x in k for x in ("allocated","key_count","offload","disk","evict","promot"))})
st.close()

# ---- GPU 버퍼로 직접 받기(vLLM 커넥터가 쓰는 batch_get_into 경로). SSD 사본만 남은 키도 되는지 본다.
import torch
st=MooncakeDistributedStore()
assert st.setup(IP, META, 0, 1<<28, "rdma", DEV, MASTER)==0
buf=torch.empty(SZ*16, dtype=torch.uint8, device="cuda")
assert st.register_buffer(buf.data_ptr(), buf.numel())==0
ok=bad=miss=0; t0=time.time(); nb=0
for i0 in range(0, N, 16):
    ks=[f"k{i}" for i in range(i0, min(N, i0+16))]
    ptrs=[buf.data_ptr()+j*SZ for j in range(len(ks))]; szs=[SZ]*len(ks)
    res=st.batch_get_into(ks, ptrs, szs)
    for j,(k,r) in enumerate(zip(ks,res)):
        if r<0: miss+=1; continue
        nb+=r
        if hashlib.sha256(buf[j*SZ:j*SZ+SZ].cpu().numpy().tobytes()).hexdigest()==vals[int(k[1:])]: ok+=1
        else: bad+=1
dt=time.time()-t0; print(f"batch_get_into(GPU) ok={ok} bad={bad} miss={miss} {nb/dt/1e9:.2f} GB/s (검증 복사 포함)")
st.unregister_buffer(buf.data_ptr()); st.close()
