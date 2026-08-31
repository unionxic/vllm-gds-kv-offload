# 경로 순수성 분류: transport별 1회 IO를 TRACE 로그로 잡아 실제 커널 경로를 판정.
#   marker "bounce_buffer"  → nvidia-fs 내부 GPU bounce buffer 경로 (미등록 IO의 정상 경로)
#   marker "cufio-px" IO 라인 → userspace POSIX pool = compat (native 아님 — 경고 대상)
#   둘 다 없음 + Bar1 등록 존재 → registered direct DMA
# 사용: CUFILE_ENV_PATH_JSON=$PWD/cufile_trace.json python path_classify.py
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = ["gds_reg", "gds_unreg", "gds_staging"]

WORKER = r'''
import os, sys, torch
sys.path.insert(0, %(here)r)
from gdslib import Gds
mode = sys.argv[1]
g = Gds()
N = 1 << 20
src = torch.zeros(N, dtype=torch.uint8, device="cuda")
dst = torch.zeros(N, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
fd = os.open(os.path.join(%(here)r, "data", "classify.bin"),
             os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o644)
fh = g.handle_register(fd)
if mode == "gds_reg":
    g.buf_register(src.data_ptr(), N); g.buf_register(dst.data_ptr(), N)
    g.write(fh, src.data_ptr(), N, 0); g.read(fh, dst.data_ptr(), N, 0)
    g.buf_deregister(src.data_ptr()); g.buf_deregister(dst.data_ptr())
elif mode == "gds_unreg":
    g.write(fh, src.data_ptr(), N, 0); g.read(fh, dst.data_ptr(), N, 0)
else:  # gds_staging: staging은 등록 버퍼 — 등록 경로와 동일해야 함
    st = torch.zeros(N, dtype=torch.uint8, device="cuda"); torch.cuda.synchronize()
    g.buf_register(st.data_ptr(), N)
    st.copy_(src); torch.cuda.synchronize()
    g.write(fh, st.data_ptr(), N, 0); g.read(fh, st.data_ptr(), N, 0)
    g.buf_deregister(st.data_ptr())
g.handle_deregister(fh); os.close(fd); g.close()
print("worker-ok")
'''


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    env = dict(os.environ, CUFILE_ENV_PATH_JSON=os.path.join(HERE, "cufile_trace.json"))
    results = {}
    for mode in CASES:
        log = os.path.join(HERE, "cufile.log")
        if os.path.exists(log):
            os.remove(log)
        r = subprocess.run([sys.executable, "-c", WORKER % {"here": HERE}, mode],
                           cwd=HERE, env=env, capture_output=True, text=True)
        assert "worker-ok" in r.stdout, (mode, r.stdout[-500:], r.stderr[-500:])
        text = open(log).read() if os.path.exists(log) else ""
        bounce_w = len(re.findall(r"write_through_bounce_buffer completed", text))
        bounce_r = len(re.findall(r"read_through_bounce_buffer completed", text))
        px_io = len(re.findall(r"cufio-px.*(?:read|write)", text, re.I)) - text.count("px-pool")
        results[mode] = (bounce_w, bounce_r, max(0, px_io))
        cls_w = "COMPAT-POSIX" if px_io > 0 else ("INTERNAL-BOUNCE" if bounce_w else "DIRECT")
        cls_r = "COMPAT-POSIX" if px_io > 0 else ("INTERNAL-BOUNCE" if bounce_r else "DIRECT")
        print(f"{mode:12s} write={cls_w:16s} read={cls_r:16s} "
              f"(bounce_w={bounce_w} bounce_r={bounce_r} px_io={max(0, px_io)})")


if __name__ == "__main__":
    main()
