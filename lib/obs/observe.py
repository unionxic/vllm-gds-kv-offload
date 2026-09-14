#!/usr/bin/env python3
"""외부 관측기. 1초마다 nvidia-fs 통계, 대상 프로세스 트리의 메모리, 캐시 디렉터리 파일 크기를 tier_samples.jsonl로.
usage: observe.py --run-dir DIR --pid PID [--cache-dir DIR] [--interval 1]
SIGTERM으로 종료. 각 줄에 wall_ns와 mono_ns를 둬 다른 기록과 정렬."""
import argparse, json, os, signal, time
from pathlib import Path

def nvfs_stats():
    try:
        return Path("/proc/driver/nvidia-fs/stats").read_text()
    except OSError as e:
        return f"error: {e}"

def proc_tree(root_pid):
    out, todo, seen = {}, [str(root_pid)], set()
    while todo:
        pid = todo.pop()
        if pid in seen: continue
        seen.add(pid); p = Path("/proc") / pid
        try:
            st = {k.strip(): v.strip() for line in p.joinpath("status").read_text().splitlines() if ":" in line
                  for k, v in [line.split(":", 1)] if k in ("Name", "VmRSS", "VmSize", "VmLck", "VmPin", "RssAnon", "RssFile", "RssShmem", "Threads")}
            out[pid] = st
            for t in p.joinpath("task").iterdir():
                try: todo.extend(t.joinpath("children").read_text().split())
                except OSError: pass
        except OSError:
            pass
    return out

def files_in(d):
    if not d: return None
    tot = n = alloc = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                st = os.stat(os.path.join(dp, f)); n += 1; tot += st.st_size; alloc += st.st_blocks * 512
            except OSError: pass
    return dict(files=n, bytes=tot, allocated_bytes=alloc)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--cache-dir", default=None); ap.add_argument("--interval", type=float, default=1.0); a = ap.parse_args()
    running = True
    def stop(*_):
        nonlocal running; running = False
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with open(os.path.join(a.run_dir, "tier_samples.jsonl"), "w", buffering=1) as out:
        while running:
            row = dict(wall_ns=time.time_ns(), mono_ns=time.monotonic_ns(), nvidia_fs=nvfs_stats(),
                       processes=proc_tree(a.pid), cache_files=files_in(a.cache_dir))
            out.write(json.dumps(row) + "\n")
            time.sleep(a.interval)

if __name__ == "__main__":
    main()
