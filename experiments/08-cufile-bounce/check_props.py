"""적용된 cufile.json 값 확인: cuFileDriverGetProperties + 미등록 GPU 버퍼로 cuFileRead 1회(bounce 경로 유발).
usage: [CUFILE_ENV_PATH_JSON=...] python check_props.py [file_mib]"""
import ctypes, os, sys, torch
L = ctypes.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0")
class NVFS(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint), ("minor", ctypes.c_uint), ("poll_thresh_size", ctypes.c_size_t),
                ("max_direct_io_size", ctypes.c_size_t), ("dstatusflags", ctypes.c_uint), ("dcontrolflags", ctypes.c_uint)]
class Props(ctypes.Structure):
    _fields_ = [("nvfs", NVFS), ("fflags", ctypes.c_uint), ("max_device_cache_size", ctypes.c_uint),
                ("per_buffer_cache_size", ctypes.c_uint), ("max_device_pinned_mem_size", ctypes.c_uint),
                ("max_batch_io_size", ctypes.c_uint), ("max_batch_io_timeout_msecs", ctypes.c_uint)]
class Err(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]
class Handle(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]
class Descr(ctypes.Structure):   # CUfileDescr_t: type, union handle, fs_ops
    _fields_ = [("type", ctypes.c_int), ("handle", Handle), ("fs_ops", ctypes.c_void_p)]
L.cuFileDriverOpen.restype = Err
L.cuFileDriverGetProperties.restype = Err; L.cuFileDriverGetProperties.argtypes = [ctypes.POINTER(Props)]
L.cuFileHandleRegister.restype = Err; L.cuFileHandleRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(Descr)]
L.cuFileRead.restype = ctypes.c_ssize_t; L.cuFileRead.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_longlong, ctypes.c_longlong]
def props(when):
    p = Props(); e = L.cuFileDriverGetProperties(ctypes.byref(p)); assert e.err == 0, e.err
    print(f"[{when}] max_direct_io_size_kb={p.nvfs.max_direct_io_size} max_device_cache_size_kb={p.max_device_cache_size} per_buffer_cache_size_kb={p.per_buffer_cache_size}")
e = L.cuFileDriverOpen(); assert e.err == 0, f"cuFileDriverOpen err={e.err}"
print("json =", os.environ.get("CUFILE_ENV_PATH_JSON", "(default /etc/cufile.json)"))
props("after open")
mib = int(sys.argv[1]) if len(sys.argv) > 1 else 64
path = os.path.expanduser("~/experiments/vllm-gds-kv/results/weight-offload/check_props.bin")
if not os.path.exists(path) or os.path.getsize(path) != mib << 20:
    with open(path, "wb") as f: f.write(os.urandom(mib << 20))
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
d = Descr(type=1, fs_ops=None); d.handle.fd = fd; h = ctypes.c_void_p()
e = L.cuFileHandleRegister(ctypes.byref(h), ctypes.byref(d)); assert e.err == 0, f"HandleRegister err={e.err}"
gpu = torch.empty(mib << 20, dtype=torch.uint8, device="cuda")   # 미등록 → bounce 경로
n = L.cuFileRead(h, ctypes.c_void_p(gpu.data_ptr()), mib << 20, 0, 0)
print(f"cuFileRead unregistered {mib} MiB -> {n} bytes")
props("after read")
L.cuFileHandleDeregister(h); os.close(fd); L.cuFileDriverClose()
