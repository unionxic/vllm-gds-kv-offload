"""vLLM SsdTier 읽기 패턴 재현: N 스레드가 각자 큰 cuFileRead(미등록 GPU 버퍼)를 반복. 조각 크기 json은 CUFILE_ENV_PATH_JSON으로.
usage: python bench_bounce.py <file> [threads=4] [chunk_mib=64] [passes=2] [register=0]"""
import ctypes, os, sys, time, threading, torch
L = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0")
class Err(ctypes.Structure): _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]
class Handle(ctypes.Union): _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]
class Descr(ctypes.Structure): _fields_ = [("type", ctypes.c_int), ("handle", Handle), ("fs_ops", ctypes.c_void_p)]
L.cuFileDriverOpen.restype = Err
L.cuFileHandleRegister.restype = Err; L.cuFileHandleRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(Descr)]
L.cuFileBufRegister.restype = Err; L.cuFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
L.cuFileRead.restype = ctypes.c_ssize_t; L.cuFileRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong, ctypes.c_longlong]
path = sys.argv[1]; nthr = int(sys.argv[2]) if len(sys.argv) > 2 else 4
chunk = (int(sys.argv[3]) if len(sys.argv) > 3 else 64) << 20; passes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
register = int(sys.argv[5]) if len(sys.argv) > 5 else 0
assert L.cuFileDriverOpen().err == 0
size = os.path.getsize(path); nchunks = size // chunk
fd = os.open(path, os.O_RDONLY | os.O_DIRECT); d = Descr(type=1, fs_ops=None); d.handle.fd = fd; h = ctypes.c_void_p()
assert L.cuFileHandleRegister(ctypes.byref(h), ctypes.byref(d)).err == 0
bufs = [torch.empty(chunk, dtype=torch.uint8, device="cuda") for _ in range(nthr)]
if register:
    for b in bufs: assert L.cuFileBufRegister(ctypes.c_void_p(b.data_ptr()), chunk, 0).err == 0
errs = []
def worker(t):
    for p in range(passes):
        for i in range(t, nchunks, nthr):
            n = L.cuFileRead(h, ctypes.c_void_p(bufs[t].data_ptr()), chunk, i * chunk, 0)
            if n != chunk: errs.append((t, i, n)); return
torch.cuda.synchronize(); t0 = time.perf_counter()
ths = [threading.Thread(target=worker, args=(t,)) for t in range(nthr)]
[x.start() for x in ths]; [x.join() for x in ths]
dt = time.perf_counter() - t0; total = nchunks * chunk * passes
print(f"BENCH json={os.path.basename(os.environ.get('CUFILE_ENV_PATH_JSON', 'default'))} threads={nthr} chunk={chunk>>20}MiB register={register} "
      f"total={total/2**30:.1f}GiB time={dt:.2f}s -> {total/dt/2**30:.2f} GiB/s errs={errs[:2]}")
