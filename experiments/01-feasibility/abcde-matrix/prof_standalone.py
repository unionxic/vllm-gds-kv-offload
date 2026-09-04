# in-engine V1-D와 동일 기하의 standalone 재현 (nsys 캡처: cudaProfilerApi 구간)
#  - 127 파일 × 5MiB, expfs와 같은 DualQueueThreadPool 8스레드
#  - 목적지 = 큰 GPU 풀의 매번 다른 오프셋 (KV 풀 모사)
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib"))
from gdslib import Gds

from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

N, CH = 127, 5 << 20
DATA = os.path.join(HERE, "prof_data")
os.makedirs(DATA, exist_ok=True)

g = Gds()
pool_t = torch.zeros(N * CH, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()

src = torch.zeros(CH, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
for i in range(N):
    p = f"{DATA}/c{i}.bin"
    if not os.path.exists(p):
        fd = os.open(p, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
        fh = g.handle_register(fd)
        g.write(fh, src.data_ptr(), CH, 0)
        g.handle_deregister(fh)
        os.close(fd)

tp = DualQueueThreadPool(8, 8, thread_name_prefix="prof")


def read_task(i):
    with torch.cuda.nvtx.range("standalone.load[cufile]"):
        fd = os.open(f"{DATA}/c{i}.bin", os.O_RDONLY | os.O_DIRECT)
        fh = g.handle_register(fd)
        g.read(fh, pool_t.data_ptr() + i * CH, CH, 0)
        g.handle_deregister(fh)
        os.close(fd)


# 워밍업 1패스 (캡처 밖)
tp.enqueue_load("warm", N, [lambda i=i: read_task(i) for i in range(N)])
while not tp.get_finished():
    time.sleep(0.005)

torch.cuda.profiler.start()
t0 = time.perf_counter()
tp.enqueue_load("run", N, [lambda i=i: read_task(i) for i in range(N)])
while not tp.get_finished():
    time.sleep(0.001)
dt = time.perf_counter() - t0
torch.cuda.profiler.stop()
print(f"standalone: {dt*1e3:.1f} ms total, {dt/N*1e3:.2f} ms/chunk, {N*CH/dt/2**30:.2f} GiB/s")
tp.shutdown()
g.close()
