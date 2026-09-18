"""전송 대역폭 측정: host→GPU(pinned/pageable), GPU→host, SSD→GPU(cuFile bounce), GPU→SSD, GPU 계산(행렬곱),
단독과 동시 실행 쌍의 저하율. 결과 results/channels/<tag>.json. 모델 없음.
usage: python bench_channels.py --out results/channels/rain.json [--gib 4] [--reps 3]"""
import argparse, ctypes, json, os, shutil, subprocess, threading, time
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--gib", type=float, default=4.0); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ssd-dir", default=os.path.expanduser("~/gds-kv/vllm-gds-kv/results/channels/tmp"))
a = ap.parse_args()
N = int(a.gib * 2**30); dev = "cuda"
os.makedirs(a.ssd_dir, exist_ok=True)
def gbps(nbytes, s): return round(nbytes / s / 1e9, 2)

# ---- GPU 버퍼, host 버퍼 ----
g1 = torch.empty(N, dtype=torch.uint8, device=dev); g2 = torch.empty(N, dtype=torch.uint8, device=dev)
h_pin = torch.empty(N, dtype=torch.uint8, pin_memory=True); h_pin2 = torch.empty(N, dtype=torch.uint8, pin_memory=True); h_page = torch.empty(N, dtype=torch.uint8)
torch.cuda.synchronize()

def h2d(src, dst, stream):
    with torch.cuda.stream(stream): dst.copy_(src, non_blocking=True)
    stream.synchronize()
def d2h(src, dst, stream):
    with torch.cuda.stream(stream): dst.copy_(src, non_blocking=True)
    stream.synchronize()
def timed(fn):
    t = time.perf_counter(); fn(); return time.perf_counter() - t

# ---- cuFile (bounce) ----
# 포크 ssd_tier.py와 같은 ctypes 선언(검증됨)
from vllm.model_executor.offloader.ssd_tier import CuFile
CF = CuFile.get(); L = CF.lib
for _f in (L.cuFileRead, L.cuFileWrite):
    _f.restype = ctypes.c_ssize_t; _f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong, ctypes.c_longlong]
CHUNK = 1 << 20  # backend·gdsio와 같은 1 MiB 호출(64 MiB 호출은 bounce 경로에서 1.7 GB/s로 느림)
def cufile_io(path, gbuf, write, threads=4):
    """파일 하나를 threads개 스레드가 64 MiB 조각으로 나눠 cuFileRead/Write. 반환 초"""
    fd = os.open(path, (os.O_WRONLY | os.O_CREAT | os.O_TRUNC if write else os.O_RDONLY) | os.O_DIRECT, 0o644)
    fh = CF.handle_register(fd)
    base = gbuf.data_ptr(); n = gbuf.numel(); err = []
    def worker(t):
        off = t * CHUNK
        while off < n:
            sz = min(CHUNK, n - off)
            r = (L.cuFileWrite if write else L.cuFileRead)(fh, ctypes.c_void_p(base + off), sz, off, 0)
            if r != sz: err.append((off, r)); break
            off += threads * CHUNK
    t0 = time.perf_counter(); ths = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    [x.start() for x in ths]; [x.join() for x in ths]; s = time.perf_counter() - t0
    L.cuFileHandleDeregister(fh); os.close(fd); assert not err, err; return s
def drop_caches():
    subprocess.run("sync", shell=True)
F = os.path.join(a.ssd_dir, "chan.bin")
g1.random_(0, 255); torch.cuda.synchronize()
res = {"gib": a.gib, "reps": a.reps, "single": {}, "pair": {}}
s0, s1 = torch.cuda.Stream(), torch.cuda.Stream()

def rep(fn):
    ts = [timed(fn) for _ in range(a.reps)]; return min(ts)
# ---- 단독 ----
res["single"]["h2d_pinned"] = gbps(N, rep(lambda: h2d(h_pin, g1, s0)))
res["single"]["h2d_pageable"] = gbps(N, rep(lambda: h2d(h_page, g1, s0)))
res["single"]["d2h_pinned"] = gbps(N, rep(lambda: d2h(g1, h_pin, s0)))
res["single"]["ssd_write_cufile"] = gbps(N, rep(lambda: cufile_io(F, g1, True)))
drop_caches(); res["single"]["ssd_read_cufile"] = gbps(N, rep(lambda: cufile_io(F, g2, False)))
# GPU 계산: fp16 행렬곱 TFLOPS
A = torch.randn(8192, 8192, device=dev, dtype=torch.float16); B = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
def mm(n=10):
    for _ in range(n): C = A @ B
    torch.cuda.synchronize()
mm(2); t = rep(lambda: mm(10)); res["single"]["gpu_fp16_tflops"] = round(10 * 2 * 8192**3 / t / 1e12, 1)
# ---- 동시 쌍: 각 전송 경로을 스레드로 동시에 돌리고 각각의 소요를 잼 ----
def pair(name, fa, fb):
    out = {}
    def run(k, f):
        t = time.perf_counter(); f(); out[k] = time.perf_counter() - t
    ta = threading.Thread(target=run, args=("a", fa)); tb = threading.Thread(target=run, args=("b", fb))
    t0 = time.perf_counter(); ta.start(); tb.start(); ta.join(); tb.join(); wall = time.perf_counter() - t0
    return out, wall
def mmloop(n):
    """행렬곱 n회(8192³ fp16, 회당 1.1 TFLOP). 단독 시간 대비 쌍 실행 시간으로 계산 전송 경로 저하율을 본다."""
    for _ in range(n): C = A @ B
    torch.cuda.synchronize()
mm(2); MM_N = 60; res["single"]["gpu_mm60_s"] = round(rep(lambda: mmloop(MM_N)), 3)
def h2d_rep(k):
    for _ in range(k): h2d(h_pin, g1, s0)
pairs = {
    "h2d_pinned+ssd_read": (lambda: h2d(h_pin, g1, s0), lambda: cufile_io(F, g2, False), "h2d", "ssd_read"),
    "h2d_pinned+d2h_pinned": (lambda: h2d(h_pin, g1, s0), lambda: d2h(g2, h_pin2, s1), "h2d", "d2h"),
    "ssd_read+ssd_write": (lambda: cufile_io(F, g2, False), lambda: cufile_io(F + ".w", g1, True, 2), "ssd_read", "ssd_write"),
    "h2d_pinned+gpu_mm": (lambda: h2d_rep(3), lambda: mmloop(MM_N), "h2d3", "gpu_mm"),
    "ssd_read+gpu_mm": (lambda: cufile_io(F, g2, False), lambda: mmloop(MM_N), "ssd_read", "gpu_mm"),
    "h2d_pinned+ssd_read+gpu_mm": None,
    "d2h_pinned+ssd_read": (lambda: d2h(g1, h_pin2, s1), lambda: cufile_io(F, g2, False), "d2h", "ssd_read"),
}
for name, spec in pairs.items():
    drop_caches()
    if spec is None:
        out = {}
        def run(k, f):
            t = time.perf_counter(); f(); out[k] = time.perf_counter() - t
        ths = [threading.Thread(target=run, args=("h2d", lambda: h2d(h_pin, g1, s0))), threading.Thread(target=run, args=("ssd_read", lambda: cufile_io(F, g2, False))), threading.Thread(target=run, args=("gpu_mm", lambda: mmloop(MM_N)))]
        t0 = time.perf_counter(); [x.start() for x in ths]; [x.join() for x in ths]; wall = time.perf_counter() - t0
        res["pair"][name] = dict(h2d_gbps=gbps(N, out["h2d"]), ssd_read_gbps=gbps(N, out["ssd_read"]), gpu_mm_s=round(out["gpu_mm"], 3), wall_s=round(wall, 2))
        continue
    fa, fb, ka, kb = spec; out, wall = pair(name, fa, fb)
    d = {"wall_s": round(wall, 2)}
    for k, key in (("a", ka), ("b", kb)):
        if key == "gpu_mm": d["gpu_mm_s"] = round(out[k], 3)
        elif key == "h2d3": d["h2d_gbps"] = gbps(3 * N, out[k])
        else: d[key + "_gbps"] = gbps(N, out[k])
    res["pair"][name] = d
# 단독 대비 저하율
single = {"h2d": res["single"]["h2d_pinned"], "d2h": res["single"]["d2h_pinned"], "ssd_read": res["single"]["ssd_read_cufile"], "ssd_write": res["single"]["ssd_write_cufile"]}
for name, d in res["pair"].items():
    for key, v in list(d.items()):
        if key.endswith("_gbps"):
            base = single.get(key[:-5]);
            if base: d[key[:-5] + "_vs_single"] = round(v / base, 2)
    if "gpu_mm_s" in d: d["gpu_mm_vs_single"] = round(res["single"]["gpu_mm60_s"] / d["gpu_mm_s"], 2)
res["env"] = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, host=os.uname().nodename)
shutil.rmtree(a.ssd_dir, ignore_errors=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
