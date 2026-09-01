# 실행 환경 스냅샷 수집기. 각 run의 meta.json에 시작/종료 스냅샷을 기록한다.
import glob
import json
import os
import subprocess
import sys
import time


def _read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def _proc_stat_ctxt():
    for line in open("/proc/stat"):
        if line.startswith("ctxt "):
            return int(line.split()[1])
    return None


def _self_ctx_switches():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")):
            k, v = line.split(":")
            out[k] = int(v)
    return out


def thread_schedstat() -> dict:
    """TID별 (cputime_ns, runqueue_wait_ns) — CPU 경합 대기 % 산출용."""
    out = {}
    base = f"/proc/{os.getpid()}/task"
    try:
        for tid in os.listdir(base):
            try:
                cpu_ns, wait_ns, _ = _read(f"{base}/{tid}/schedstat").split()
                name = _read(f"{base}/{tid}/comm")
                out[tid] = dict(name=name, cpu_ns=int(cpu_ns), wait_ns=int(wait_ns))
            except (OSError, AttributeError, ValueError):
                continue
    except OSError:
        pass
    return out


def diskstats(dev="nvme0n1") -> dict:
    """NVMe 누적 통계 — delta로 busy time(ms)·in-flight 산출."""
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] == dev:
            return dict(reads=int(f[3]), read_sectors=int(f[5]),
                        writes=int(f[7]), write_sectors=int(f[9]),
                        inflight=int(f[11]), io_ms=int(f[12]),
                        weighted_io_ms=int(f[13]))
    return {}


def collect(phase: str) -> dict:
    shm_files = sorted(glob.glob("/dev/shm/vllm_offload_*.mmap"))
    shm_stat = os.statvfs("/dev/shm")
    root_stat = os.statvfs("/")
    mem = {}
    for line in open("/proc/meminfo"):
        k = line.split(":")[0]
        if k in ("MemTotal", "MemFree", "MemAvailable", "Cached", "Shmem"):
            mem[k] = int(line.split()[1])  # kB
    freqs = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"))[:4]
    temps = {}
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        name = _read(f"{hw}/name")
        t = _read(f"{hw}/temp1_input")
        if name and t:
            temps[name] = int(t) / 1000
    snap = {
        "phase": phase,
        "ts": time.time(),
        "switch_interval": sys.getswitchinterval(),
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "loadavg": os.getloadavg(),
        "ctxt_total": _proc_stat_ctxt(),
        "self_ctx": _self_ctx_switches(),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_freq_khz": [int(_read(f) or 0) for f in freqs],
        "temps_c": temps,
        "shm_files": [os.path.basename(f) for f in shm_files],
        "shm_used_gib": round((shm_stat.f_blocks - shm_stat.f_bavail) * shm_stat.f_frsize / 2**30, 2),
        "root_free_gib": round(root_stat.f_bavail * root_stat.f_frsize / 2**30, 1),
        "meminfo_kb": mem,
        "gpu": _cmd(["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu,temperature.gpu,memory.used",
                     "--format=csv,noheader"]),
        "thread_schedstat": thread_schedstat(),
        "diskstats_nvme0n1": diskstats(),
    }
    if phase == "start":
        import torch
        snap.update({
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_vllm": _cmd(["git", "-C", os.path.expanduser("~/vllm"), "describe", "--always", "--dirty"]),
            "git_exp": _cmd(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "describe", "--always", "--dirty"]),
            "libcufile": _cmd(["bash", "-c",
                               "ls -la /usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0 | awk '{print $NF}'"]),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "vllm_env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
            "cpu_model": _cmd(["bash", "-c", "lscpu | grep 'Model name' | head -1"]),
            "cores": _cmd(["bash", "-c", "lscpu | grep -E '^(CPU\\(s\\)|Core|Socket|NUMA)'"]),
        })
    return snap
