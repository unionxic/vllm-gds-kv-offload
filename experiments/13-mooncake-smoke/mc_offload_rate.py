"""put을 계속 흘리면서 2 s마다 master 메트릭(메모리 할당, SSD 파일 바이트, 키 수, 축출 키 수)을 찍어
오프로드가 어떤 속도로 진행되는지 본다."""
import os, sys, time, hashlib, urllib.request, threading
from mooncake.store import MooncakeDistributedStore
IP="30.0.0.4"; MASTER="30.0.0.3:50051"; META="http://30.0.0.3:8080/metadata"; DEV="mlx5_0"
os.environ["MOONCAKE_REQUESTER_LOCAL_HOSTNAME"]=IP
N=int(sys.argv[1]); SLEEP=float(sys.argv[2]); SZ=8<<20
def metrics():
    t=urllib.request.urlopen("http://30.0.0.3:9003/metrics",timeout=5).read().decode()
    return {l.split()[0]:l.split()[1] for l in t.splitlines() if l and not l.startswith("#") and len(l.split())==2}
stop=False; nput=[0]; nfail=[0]
def mon():
    t0=time.time()
    while not stop:
        m=metrics()
        print(f"t={time.time()-t0:5.1f}s put={nput[0]} fail={nfail[0]} mem={int(m['master_allocated_bytes'])/2**30:.2f}G disk={int(m['master_allocated_file_size_bytes'])/2**30:.2f}G keys={m['master_key_count']} evicted={m['master_evicted_key_count_mem']}", flush=True)
        time.sleep(2)
st=MooncakeDistributedStore()
assert st.setup(IP, META, 0, 1<<28, "rdma", DEV, MASTER)==0
st.remove_all()
th=threading.Thread(target=mon, daemon=True); th.start()
v=b"\x5a"*SZ
for i in range(N):
    if st.put(f"k{i}", v)!=0: nfail[0]+=1
    else: nput[0]+=1
    time.sleep(SLEEP)
time.sleep(12); stop=True; time.sleep(2.5)
st.close()
