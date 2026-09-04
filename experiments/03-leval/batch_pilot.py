# cuFile Batch API 1-chunk standalone 파일럿 (프로토콜의 사전 검증 — 엔진 밖).
# 판별 축: 미등록 vs 등록 버퍼 × 단일 vs 다중 entry × write/read.
# 결과: 각 케이스의 이벤트 도착 여부·status·ret. 전부 무도착이면 unsupported로 기록.
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
from cufile_batch import CUFILE_COMPLETE, CUFILE_READ, CUFILE_WRITE, CuFileBatch
from gdslib import Gds

SPAN = 640 * 1024  # b64 chunk의 텐서당 span 크기 (20KiB/tok*32tok=640KiB)
g = Gds()
batch = CuFileBatch(g.lib, 64)

buf = torch.arange(SPAN * 4, dtype=torch.uint8, device="cuda") % 251
dst = torch.zeros(SPAN * 4, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()

path = os.path.join(HERE, "batch_pilot.bin")
fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
fh = g.handle_register(fd)


def drain(expect, budget_s=10.0):
    got, t0 = [], time.time()
    while len(got) < expect and time.time() - t0 < budget_s:
        evs, timed_out = batch.get_status(1, 64, 500)
        got.extend(evs)
    return got


def case(name, entries, expect):
    try:
        arr = batch.submit(entries)  # noqa: F841
    except Exception as e:
        print(f"{name}: SUBMIT FAIL {e}")
        return False
    evs = drain(expect)
    ok = (len(evs) == expect
          and all(s == CUFILE_COMPLETE for _, s, _ in evs)
          and sum(r for _, _, r in evs) == sum(e[2] for e in entries))
    print(f"{name}: events={len(evs)}/{expect} "
          f"statuses={[hex(s) for _, s, _ in evs]} rets={[r for _, _, r in evs]} "
          f"-> {'OK' if ok else 'FAIL'}")
    return ok


print("--- 미등록 버퍼 ---")
w1 = case("unreg write x1", [(fh.value, buf.data_ptr(), SPAN, 0, CUFILE_WRITE, 101)], 1)
w4 = case("unreg write x4", [(fh.value, buf.data_ptr() + i * SPAN, SPAN, i * SPAN,
                              CUFILE_WRITE, 200 + i) for i in range(4)], 4)
r4 = case("unreg read  x4", [(fh.value, dst.data_ptr() + i * SPAN, SPAN, i * SPAN,
                              CUFILE_READ, 300 + i) for i in range(4)], 4)

print("--- 등록 버퍼 ---")
g.buf_register(buf.data_ptr(), SPAN * 4)
g.buf_register(dst.data_ptr(), SPAN * 4)
rw1 = case("reg write x4", [(fh.value, buf.data_ptr() + i * SPAN, SPAN, i * SPAN,
                             CUFILE_WRITE, 400 + i) for i in range(4)], 4)
rr1 = case("reg read  x4", [(fh.value, dst.data_ptr() + i * SPAN, SPAN, i * SPAN,
                             CUFILE_READ, 500 + i) for i in range(4)], 4)
if rr1:
    print("checksum:", bool((buf == dst).all()))
g.buf_deregister(buf.data_ptr())
g.buf_deregister(dst.data_ptr())

batch.destroy()
g.handle_deregister(fh)
os.close(fd)
os.remove(path)
g.close()
print("VERDICT:", "usable" if (w4 and r4) or (rw1 and rr1) else "unsupported-or-abi-issue")
