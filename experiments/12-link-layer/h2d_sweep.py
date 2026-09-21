#!/usr/bin/env python3
"""pinned host↔GPU 복사 대역폭. 전송 크기(--sizes)·스트림 수(--streams)별로 --sec 초씩 돌려 GB/s를 jsonl 한 줄씩 남긴다.
usage: h2d_sweep.py --out FILE --tag TAG [--sizes 64K,1M,16M,256M] [--streams 1,2] [--sec 3] [--dir h2d|d2h|both]"""
import argparse, json, time, torch
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="")
ap.add_argument("--sizes", default="64K,256K,1M,4M,16M,64M,256M"); ap.add_argument("--streams", default="1,2")
ap.add_argument("--sec", type=float, default=3.0); ap.add_argument("--dir", default="both")
ap.add_argument("--bytes", type=float, default=0, help="0이 아니면 --sec 대신 이만큼(바이트) 옮기고 완료 시간을 잰다")
a = ap.parse_args()
def sz(s): u = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}; return int(float(s[:-1]) * u[s[-1]]) if s[-1] in u else int(s)
sizes = [sz(s) for s in a.sizes.split(",")]; streams = [int(x) for x in a.streams.split(",")]
dirs = ["h2d", "d2h"] if a.dir == "both" else [a.dir]
TOTAL = 1 << 30  # 스트림당 버퍼 1 GiB
out = open(a.out, "a")
for d in dirs:
    for ns in streams:
        st = [torch.cuda.Stream() for _ in range(ns)]
        host = [torch.empty(TOTAL, dtype=torch.uint8).pin_memory() for _ in range(ns)]
        dev = [torch.empty(TOTAL, dtype=torch.uint8, device="cuda") for _ in range(ns)]
        for n in sizes:
            nchunk = TOTAL // n
            # 워밍업
            for i in range(ns):
                with torch.cuda.stream(st[i]):
                    (dev[i][:n].copy_(host[i][:n], non_blocking=True) if d == "h2d" else host[i][:n].copy_(dev[i][:n], non_blocking=True))
            torch.cuda.synchronize()
            t0 = time.perf_counter(); moved = 0; k = 0
            while (moved < a.bytes) if a.bytes else (time.perf_counter() - t0 < a.sec):
                for _ in range(16):
                    off = (k % nchunk) * n; k += 1
                    for i in range(ns):
                        with torch.cuda.stream(st[i]):
                            if d == "h2d": dev[i][off:off + n].copy_(host[i][off:off + n], non_blocking=True)
                            else: host[i][off:off + n].copy_(dev[i][off:off + n], non_blocking=True)
                    moved += n * ns
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            row = dict(tag=a.tag, dir=d, streams=ns, size=n, gbps=round(moved / dt / 1e9, 3), sec=round(dt, 2))
            out.write(json.dumps(row) + "\n"); out.flush(); print(row, flush=True)
        del host, dev; torch.cuda.empty_cache()
