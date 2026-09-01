# shm_repro 오케스트레이터: 모드별로 child 엔진 기동 → 실행 중 /dev/shm 관찰 →
# SIGKILL → 잔재 판정. 다른 프로세스의 shm 파일은 건드리지 않는다(관찰만).
import glob
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable


def shm():
    return sorted(os.path.basename(f)
                  for f in glob.glob("/dev/shm/vllm_offload_*.mmap"))


for mode in ("cpu", "tiering"):
    base = set(shm())
    log = os.path.join(HERE, f"shm_{mode}.log")
    child = subprocess.Popen([PY, os.path.join(HERE, "shm_repro.py"), mode],
                             stdout=open(log, "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 420
    while time.time() < deadline:
        if "READY" in open(log).read():
            break
        if child.poll() is not None:
            print(f"{mode}: child died during init — see {log}")
            break
        time.sleep(2)
    else:
        child.kill()
        print(f"{mode}: init timeout")
        continue
    if child.poll() is not None:
        continue
    during = [f for f in shm() if f not in base]
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    time.sleep(2)
    leftover = [f for f in shm() if f not in base]
    print(f"{mode}: during_run={during} leftover_after_SIGKILL={leftover} "
          f"-> {'LEAK' if leftover else 'CLEAN'}")
