# 시스템 libcufile(절대경로) ctypes 래퍼.
# cuda.bindings.cufile은 pip 휠(cu12/cu13) libcufile을 dlopen해 시스템 1.13과
# 이중 로드 → 비결정 segfault. 시스템 라이브러리 단일 로드로 회피 (README §3 함정).
import ctypes
import os

LIBCUFILE = "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0"

CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1


class CUfileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


class _DescrHandle(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class CUfileDescr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _DescrHandle),
        ("fs_ops", ctypes.c_void_p),
    ]


class GdsError(RuntimeError):
    pass


class Gds:
    def __init__(self):
        self.lib = ctypes.CDLL(LIBCUFILE, mode=ctypes.RTLD_GLOBAL)
        L = self.lib
        L.cuFileDriverOpen.restype = CUfileError
        L.cuFileDriverClose.restype = CUfileError
        L.cuFileHandleRegister.restype = CUfileError
        L.cuFileHandleRegister.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(CUfileDescr)]
        L.cuFileHandleDeregister.restype = None
        L.cuFileHandleDeregister.argtypes = [ctypes.c_void_p]
        L.cuFileBufRegister.restype = CUfileError
        L.cuFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        L.cuFileBufDeregister.restype = CUfileError
        L.cuFileBufDeregister.argtypes = [ctypes.c_void_p]
        for fn in (L.cuFileRead, L.cuFileWrite):
            fn.restype = ctypes.c_ssize_t
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                           ctypes.c_longlong, ctypes.c_longlong]
        self._check(L.cuFileDriverOpen(), "cuFileDriverOpen")

    @staticmethod
    def _check(e: CUfileError, what: str):
        if e.err != 0:
            raise GdsError(f"{what}: err={e.err} cu_err={e.cu_err}")

    def handle_register(self, fd: int):
        d = CUfileDescr()
        d.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD
        d.handle.fd = fd
        fh = ctypes.c_void_p()
        self._check(self.lib.cuFileHandleRegister(ctypes.byref(fh), ctypes.byref(d)),
                    "cuFileHandleRegister")
        return fh

    def handle_deregister(self, fh):
        self.lib.cuFileHandleDeregister(fh)

    def buf_register(self, ptr: int, size: int, flags: int = 0):
        self._check(self.lib.cuFileBufRegister(ctypes.c_void_p(ptr), size, flags),
                    "cuFileBufRegister")

    def buf_deregister(self, ptr: int):
        self._check(self.lib.cuFileBufDeregister(ctypes.c_void_p(ptr)), "cuFileBufDeregister")

    def read(self, fh, ptr: int, size: int, file_off: int, buf_off: int = 0) -> int:
        n = self.lib.cuFileRead(fh, ctypes.c_void_p(ptr), size, file_off, buf_off)
        if n < 0:
            raise GdsError(f"cuFileRead ret={n}")
        return n

    def write(self, fh, ptr: int, size: int, file_off: int, buf_off: int = 0) -> int:
        n = self.lib.cuFileWrite(fh, ctypes.c_void_p(ptr), size, file_off, buf_off)
        if n < 0:
            raise GdsError(f"cuFileWrite ret={n}")
        return n

    def close(self):
        self.lib.cuFileDriverClose()


def nvfs_ops():
    """nvidia-fs 누적 Read/Write op 카운터 (native 경로 증명용)."""
    for line in open("/proc/driver/nvidia-fs/stats"):
        if line.startswith("Ops"):
            p = dict(x.split("=") for x in line.split(":")[1].split())
            return int(p["Read"]), int(p["Write"])
    return 0, 0
