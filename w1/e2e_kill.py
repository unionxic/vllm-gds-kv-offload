# E2E: 실제 엔진 SIGKILL → stale 존재 → 다음 엔진 시작이 자동 회수 (관찰 원칙)
import glob, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ)

def shm():
    return sorted(glob.glob("/dev/shm/vllm_offload_*.mmap"))

print("shm at start:", shm())
victim = subprocess.Popen(
    [sys.executable, "replay.py", "--design", "C", "--runner", "v2",
     "--n-requests", "100"],
    cwd=HERE, env=env, stdout=open("e2e_victim.log", "w"), stderr=subprocess.STDOUT)
# 엔진이 region을 만들고 요청 처리에 들어갈 때까지 대기
deadline = time.time() + 300
while time.time() < deadline:
    if os.path.exists("e2e_victim.log") and "0/100" in open("e2e_victim.log").read():
        break
    time.sleep(2)
else:
    victim.kill(); sys.exit("victim never reached request loop")
time.sleep(5)
os.kill(victim.pid, signal.SIGKILL)
victim.wait()
time.sleep(2)
stale = shm()
print("after SIGKILL, stale:", stale)
assert stale, "FAIL: SIGKILL left no stale file (unexpected)"

nxt = subprocess.run(
    [sys.executable, "replay.py", "--design", "C", "--runner", "v2",
     "--n-requests", "3"],
    cwd=HERE, env=env, capture_output=True, text=True, timeout=600)
log = nxt.stdout
reclaimed = [l for l in log.splitlines() if "Reclaimed orphaned" in l]
obs = [l for l in log.splitlines() if l.startswith("[shm:")]
print("next-engine exit:", nxt.returncode)
print("reclaim log:", reclaimed[:2])
print("observations:", obs)
print("shm at end:", shm())
assert reclaimed and stale[0] in reclaimed[0], "FAIL: next engine did not reclaim"
assert not shm(), "FAIL: files remain after clean next-engine shutdown"
print("E2E PASS")
