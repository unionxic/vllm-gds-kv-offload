# cuFile Batch API ctypes 바인딩 — /usr/local/cuda/include/cufile.h ABI 그대로.
# 시스템 libcufile 단일 로드(gdslib.Gds의 CDLL 핸들 재사용). PyDLL 금지.
# GetStatus는 CDLL 경유 native blocking → 대기 중 GIL 해제.
import ctypes
from ctypes import (POINTER, Structure, c_int, c_long, c_size_t, c_ssize_t,
                    c_uint, c_void_p, byref)

CUFILE_BATCH = 1
CUFILE_READ, CUFILE_WRITE = 0, 1
CUFILE_COMPLETE, CUFILE_FAILED, CUFILE_TIMEOUT, CUFILE_CANCELED = 0x10, 0x40, 0x20, 0x8


class CUfileError(Structure):
    _fields_ = [("err", c_int), ("cu_err", c_int)]


class CUfileIOParams(Structure):
    # mode; union{batch{devPtr_base,file_offset,devPtr_offset,size}}; fh; opcode; cookie
    _fields_ = [
        ("mode", c_int),
        ("devPtr_base", c_void_p),
        ("file_offset", c_long),
        ("devPtr_offset", c_long),
        ("size", c_size_t),
        ("fh", c_void_p),
        ("opcode", c_int),
        ("cookie", c_void_p),
    ]


class CUfileIOEvents(Structure):
    _fields_ = [("cookie", c_void_p), ("status", c_int), ("ret", c_ssize_t)]


class Timespec(Structure):
    _fields_ = [("tv_sec", c_long), ("tv_nsec", c_long)]


class BatchError(RuntimeError):
    pass


class CuFileBatch:
    """단일 batch 핸들. capacity = 동시 outstanding entry 상한."""

    def __init__(self, lib, capacity: int = 128):
        self.lib = lib
        for name, res, args in [
            ("cuFileBatchIOSetUp", CUfileError, [POINTER(c_void_p), c_uint]),
            ("cuFileBatchIOSubmit", CUfileError,
             [c_void_p, c_uint, POINTER(CUfileIOParams), c_uint]),
            ("cuFileBatchIOGetStatus", CUfileError,
             [c_void_p, c_uint, POINTER(c_uint), POINTER(CUfileIOEvents),
              POINTER(Timespec)]),
            ("cuFileBatchIOCancel", CUfileError, [c_void_p]),
            ("cuFileBatchIODestroy", None, [c_void_p]),
        ]:
            fn = getattr(lib, name)
            fn.restype = res
            fn.argtypes = args
        h = c_void_p()
        e = lib.cuFileBatchIOSetUp(byref(h), capacity)
        if e.err != 0:
            raise BatchError(f"cuFileBatchIOSetUp err={e.err}")
        self.handle = h
        self.capacity = capacity

    def submit(self, entries):
        """entries: [(fh, dev_ptr, size, file_off, opcode, cookie_int)]"""
        n = len(entries)
        arr = (CUfileIOParams * n)()
        for i, (fh, ptr, size, foff, op, ck) in enumerate(entries):
            arr[i].mode = CUFILE_BATCH
            arr[i].devPtr_base = ptr
            arr[i].file_offset = foff
            arr[i].devPtr_offset = 0
            arr[i].size = size
            arr[i].fh = fh
            arr[i].opcode = op
            arr[i].cookie = ck
        e = self.lib.cuFileBatchIOSubmit(self.handle, n, arr, 0)
        if e.err != 0:
            raise BatchError(f"cuFileBatchIOSubmit err={e.err}")
        return arr  # 완료 시까지 참조 유지 필요

    def get_status(self, min_nr: int, max_nr: int, timeout_ms: int):
        """native blocking 대기(GIL 해제). (완료목록[(cookie,status,ret)], timed_out)"""
        nr = c_uint(max_nr)
        evs = (CUfileIOEvents * max_nr)()
        ts = Timespec(timeout_ms // 1000, (timeout_ms % 1000) * 1_000_000)
        e = self.lib.cuFileBatchIOGetStatus(self.handle, min_nr, byref(nr), evs,
                                            byref(ts))
        if e.err != 0:
            raise BatchError(f"cuFileBatchIOGetStatus err={e.err}")
        out = [(evs[i].cookie, evs[i].status, evs[i].ret) for i in range(nr.value)]
        return out, nr.value == 0

    def destroy(self):
        if self.handle:
            self.lib.cuFileBatchIODestroy(self.handle)
            self.handle = None
