# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD tier for prefetch-based weight offloading.

Offloaded layers that do not fit in the host (pinned CPU) budget are kept as
files on an NVMe device and streamed into the GPU static buffers each forward.

Two transports are supported:
- ``cufile``: GPUDirect Storage. ``cuFileRead`` DMAs file data straight into
  the GPU static buffer (no host memory hop).
- ``posix``: ``pread`` (O_DIRECT) into a pinned host bounce buffer followed by
  an async H2D copy on the offloader's copy stream (the "SSD -> CPU -> GPU"
  baseline path).

Both transports are host-synchronous (they run on a thread pool), so this
tier is only usable in eager mode: it cannot be captured into a CUDA graph.
"""

import ctypes
import ctypes.util
import os
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ALIGN = 4096


def host_mem_total_bytes() -> int:
    """MemTotal from /proc/meminfo (falls back to os.sysconf)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _round_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


# --------------------------------------------------------------------------
# Minimal ctypes binding to the system libcufile.
# The cuda-python ``cuda.bindings.cufile`` wheel dlopens its own libcufile,
# which double-loads against the system library and segfaults
# non-deterministically, so we bind the system library directly.
# --------------------------------------------------------------------------
class _CUfileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int), ("cu_err", ctypes.c_int)]


class _DescrHandle(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class _CUfileDescr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _DescrHandle),
        ("fs_ops", ctypes.c_void_p),
    ]


_CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
_LIBCUFILE_CANDIDATES = (
    os.environ.get("VLLM_LIBCUFILE_PATH", ""),
    "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0",
    "/usr/local/cuda/lib64/libcufile.so.0",
)


class CuFileError(RuntimeError):
    pass


class CuFile:
    """Process-wide cuFile driver handle (lazily opened, thread-safe)."""

    _instance: "CuFile | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "CuFile":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CuFile()
            return cls._instance

    def __init__(self):
        path = next((p for p in _LIBCUFILE_CANDIDATES if p and os.path.exists(p)), None)
        if path is None:
            path = ctypes.util.find_library("cufile")
        if path is None:
            raise CuFileError("libcufile not found; set VLLM_LIBCUFILE_PATH")
        self.lib = L = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        L.cuFileDriverOpen.restype = _CUfileError
        L.cuFileDriverClose.restype = _CUfileError
        L.cuFileHandleRegister.restype = _CUfileError
        L.cuFileHandleRegister.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_CUfileDescr),
        ]
        L.cuFileHandleDeregister.restype = None
        L.cuFileHandleDeregister.argtypes = [ctypes.c_void_p]
        L.cuFileBufRegister.restype = _CUfileError
        L.cuFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        L.cuFileBufDeregister.restype = _CUfileError
        L.cuFileBufDeregister.argtypes = [ctypes.c_void_p]
        L.cuFileRead.restype = ctypes.c_ssize_t
        L.cuFileRead.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]
        self._check(L.cuFileDriverOpen(), "cuFileDriverOpen")
        logger.info("cuFile driver opened from %s", path)

    @staticmethod
    def _check(e: _CUfileError, what: str) -> None:
        if e.err != 0:
            raise CuFileError(f"{what}: err={e.err} cu_err={e.cu_err}")

    def handle_register(self, fd: int) -> ctypes.c_void_p:
        d = _CUfileDescr()
        d.type = _CU_FILE_HANDLE_TYPE_OPAQUE_FD
        d.handle.fd = fd
        fh = ctypes.c_void_p()
        self._check(
            self.lib.cuFileHandleRegister(ctypes.byref(fh), ctypes.byref(d)),
            "cuFileHandleRegister",
        )
        return fh

    def handle_deregister(self, fh: ctypes.c_void_p) -> None:
        self.lib.cuFileHandleDeregister(fh)

    def buf_register(self, ptr: int, size: int) -> bool:
        e = self.lib.cuFileBufRegister(ctypes.c_void_p(ptr), size, 0)
        return e.err == 0

    def buf_deregister(self, ptr: int) -> None:
        self.lib.cuFileBufDeregister(ctypes.c_void_p(ptr))

    def read(self, fh: ctypes.c_void_p, ptr: int, size: int, file_off: int) -> int:
        n = self.lib.cuFileRead(fh, ctypes.c_void_p(ptr), size, file_off, 0)
        if n < 0:
            raise CuFileError(f"cuFileRead ret={n}")
        return n


# --------------------------------------------------------------------------
# File records and the tier itself
# --------------------------------------------------------------------------
@dataclass
class SsdFile:
    """One parameter persisted on the SSD tier."""

    path: str
    nbytes: int  # exact tensor bytes
    padded: int  # file size (rounded up to ALIGN for O_DIRECT)
    fd: int = -1
    fh: ctypes.c_void_p | None = None  # cuFile handle (cufile transport)


class SsdTier:
    """Owns the SSD directory, the IO thread pool and the transport."""

    def __init__(
        self,
        root: str,
        transport: str,
        io_threads: int,
        copy_stream: torch.cuda.Stream,
        rank: int = 0,
        ring_mb: int = 0,
    ):
        assert transport in ("cufile", "posix"), transport
        self.root = os.path.join(root, f"rank{rank}")
        os.makedirs(self.root, exist_ok=True)
        self.transport = transport
        self.copy_stream = copy_stream
        # Registered GPU ring (cufile only): cuFileRead lands in a small
        # pre-registered slot (direct DMA regardless of the destination
        # buffer's size vs BAR1), then a D2D copy moves it into the static
        # buffer. ring_mb == 0 disables it (reads go straight to the buffer).
        self.ring_mb = ring_mb if transport == "cufile" else 0
        self._ring: torch.Tensor | None = None
        self._ring_slots: "queue.Queue[int]" = queue.Queue()
        self._ring_streams: dict[int, torch.cuda.Stream] = {}
        self._ring_registered = 0
        if self.ring_mb > 0:
            n_slots = 2 * io_threads
            slot = self.ring_mb << 20
            raw = torch.empty(n_slots * slot + ALIGN, dtype=torch.uint8, device="cuda")
            off = (-raw.data_ptr()) % ALIGN
            self._ring = raw[off : off + n_slots * slot]
            self._ring_slot_bytes = slot
            for i in range(n_slots):
                self._ring_slots.put(i)
        self.pool = ThreadPoolExecutor(
            max_workers=io_threads, thread_name_prefix="vllm-ssd-io"
        )
        # Layer-level coordinator: separate from the IO pool so that a
        # layer job waiting on its per-parameter reads can never starve them.
        self.coord = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-ssd-coord")
        self._cufile: CuFile | None = None
        self._bounce: dict[int, torch.Tensor] = {}  # posix: per-thread pinned buf
        self._bounce_lock = threading.Lock()
        self.files: list[SsdFile] = []
        self.stats = {"reads": 0, "bytes": 0}
        self.on_activity = None  # callable(start: bool), 실제 읽기 시작/끝(IoWindow.activity)
        self._registered: list[int] = []
        if transport == "cufile":
            self._cufile = CuFile.get()

    def register_buffers(self, buffers: list[torch.Tensor]) -> tuple[int, int]:
        """Try to pin the GPU static buffers for direct DMA (cufile only).

        Registration is best-effort: it needs BAR1 space for every buffer, so
        on GPUs with a small BAR1 (e.g. 256 MiB) large slots fail and cuFile
        falls back to its internal GPU bounce cache, which is still a
        GPUDirect path. Returns (registered, failed) counts.
        """
        if self._cufile is None:
            return 0, 0
        ok = failed = 0
        if self._ring is not None:
            # Ring mode: spend BAR1 on the ring slots, not on the buffers.
            slot = self._ring_slot_bytes
            for i in range(self._ring_slots.qsize()):
                if self._cufile.buf_register(self._ring.data_ptr() + i * slot, slot):
                    self._registered.append(self._ring.data_ptr() + i * slot)
                    self._ring_registered += 1
            logger.info(
                "[SsdTier] ring mode: %d/%d slots of %d MiB registered",
                self._ring_registered,
                self._ring_slots.qsize(),
                self.ring_mb,
            )
            return self._ring_registered, self._ring_slots.qsize() - self._ring_registered
        seen: set[int] = set()
        # BAR1 창(이 카드 256 MiB)을 등록 버퍼가 다 차지하면 cuFile 내부 bounce 캐시를 매핑할 자리가 없어
        # 미등록 버퍼의 읽기가 -1(-5011)로 실패한다. 등록 총량 상한(MiB). 0이면 상한 없음.
        cap = float(os.environ.get("VLLM_OFFLOAD_SSD_REGISTER_MAX_MB", "0")) * 2**20
        reg_bytes = 0
        for buf in buffers:
            ptr, size = buf.data_ptr(), buf.numel() * buf.element_size()
            if ptr in seen:
                continue
            seen.add(ptr)
            if cap and reg_bytes + size > cap:
                failed += 1
                continue
            if self._cufile.buf_register(ptr, size):
                reg_bytes += size
                self._registered.append(ptr)
                ok += 1
            else:
                failed += 1
        logger.info(
            "[SsdTier] cuFileBufRegister on static buffers: %d ok, %d failed "
            "(failed buffers use cuFile's internal GPU bounce cache)",
            ok,
            failed,
        )
        return ok, failed

    # -- creation during weight loading ------------------------------------
    def new_file_tensor(
        self, layer_idx: int, param_name: str, shape, stride, dtype
    ) -> tuple[torch.Tensor, SsdFile]:
        """Create a file-backed CPU tensor the weight loader can write into."""
        numel = 1
        for d in shape:
            numel *= d
        nbytes = numel * torch.tensor([], dtype=dtype).element_size()
        padded = _round_up(nbytes)
        path = os.path.join(self.root, f"layer{layer_idx:03d}", f"{param_name}.bin")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(padded)
        flat = torch.from_file(path, shared=True, size=numel, dtype=dtype)
        t = torch.as_strided(flat, tuple(shape), tuple(stride))
        rec = SsdFile(path=path, nbytes=nbytes, padded=padded)
        self.files.append(rec)
        return t, rec

    def finalize(self, rec: SsdFile) -> None:
        """Flush the file, drop it from the page cache and open for O_DIRECT."""
        fd = os.open(rec.path, os.O_RDONLY | os.O_DIRECT)
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        rec.fd = fd
        if self._cufile is not None:
            rec.fh = self._cufile.handle_register(fd)

    # -- reads during forward ------------------------------------------------
    def _bounce_for(self, size: int) -> torch.Tensor:
        tid = threading.get_ident()
        with self._bounce_lock:
            buf = self._bounce.get(tid)
            if buf is None or buf.numel() < size + ALIGN:
                buf = torch.empty(size + ALIGN, dtype=torch.uint8, pin_memory=True)
                self._bounce[tid] = buf
        # align the start to ALIGN for O_DIRECT
        off = (-buf.data_ptr()) % ALIGN
        return buf[off : off + size]

    def _ring_stream(self) -> torch.cuda.Stream:
        tid = threading.get_ident()
        s = self._ring_streams.get(tid)
        if s is None:
            s = torch.cuda.Stream()
            self._ring_streams[tid] = s
        return s

    def _read_via_ring(self, rec: SsdFile, gpu_buffer: torch.Tensor) -> None:
        assert self._cufile is not None and rec.fh is not None
        assert self._ring is not None
        slot_bytes = self._ring_slot_bytes
        dst = gpu_buffer.view(torch.uint8).view(-1)
        stream = self._ring_stream()
        done = 0
        while done < rec.nbytes:
            n = min(slot_bytes, rec.nbytes - done)
            i = self._ring_slots.get()
            try:
                ptr = self._ring.data_ptr() + i * slot_bytes
                got = self._cufile.read(rec.fh, ptr, n, done)
                if got != n:
                    raise CuFileError(f"short cuFileRead {got}/{n} {rec.path}@{done}")
                src = self._ring[i * slot_bytes : i * slot_bytes + n]
                with torch.cuda.stream(stream):
                    dst[done : done + n].copy_(src, non_blocking=True)
                stream.synchronize()
            finally:
                self._ring_slots.put(i)
            done += n
        # make the layer's completion event (recorded on copy_stream) order
        # after this thread's D2D copies
        self.copy_stream.wait_stream(stream)

    def _read_one(self, rec: SsdFile, gpu_buffer: torch.Tensor) -> None:
        cb = self.on_activity
        if cb is not None:
            cb(True)
        torch.cuda.nvtx.range_push(f"weight_ssd_read {os.path.basename(os.path.dirname(rec.path))}/{os.path.basename(rec.path)} {rec.nbytes >> 20}MiB")
        try:
            self._read_one_impl(rec, gpu_buffer)
        finally:
            torch.cuda.nvtx.range_pop()
            if cb is not None:
                cb(False)

    def _read_one_impl(self, rec: SsdFile, gpu_buffer: torch.Tensor) -> None:
        assert gpu_buffer.is_contiguous(), "SSD tier needs contiguous buffers"
        if self.transport == "cufile" and self._ring is not None:
            self._read_via_ring(rec, gpu_buffer)
        elif self.transport == "cufile":
            assert self._cufile is not None and rec.fh is not None
            n = self._cufile.read(rec.fh, gpu_buffer.data_ptr(), rec.nbytes, 0)
            if n != rec.nbytes:
                raise CuFileError(f"short cuFileRead {n}/{rec.nbytes} {rec.path}")
        else:
            bounce = self._bounce_for(rec.padded)
            mv = memoryview(
                (ctypes.c_char * rec.padded).from_address(bounce.data_ptr())
            )
            done = 0
            while done < rec.padded:
                n = os.preadv(rec.fd, [mv[done:]], done)
                if n <= 0:
                    raise OSError(f"short pread {done}/{rec.padded} {rec.path}")
                done += n
            src = bounce[: rec.nbytes].view(gpu_buffer.dtype).view(gpu_buffer.shape)
            with torch.cuda.stream(self.copy_stream):
                gpu_buffer.copy_(src, non_blocking=True)
            # the bounce buffer is per-thread and reused: make sure the copy
            # has consumed it before this thread picks up the next read
            self.copy_stream.synchronize()
        self.stats["reads"] += 1
        self.stats["bytes"] += rec.nbytes

    def submit_layer(
        self,
        items: list[tuple[SsdFile, torch.Tensor]],
        fork_event: torch.cuda.Event,
        done_event: torch.cuda.Event,
    ) -> Future:
        """Read every (file, gpu_buffer) pair of a layer, then record done_event.

        The read must not start before the compute stream has finished with
        the static buffer (previous occupant of the slot): we wait for
        ``fork_event`` on the host, since the transports are host-driven.
        """

        def _job() -> None:
            fork_event.synchronize()
            futs = [self.pool.submit(self._read_one, rec, buf) for rec, buf in items]
            for f in futs:
                f.result()
            done_event.record(self.copy_stream)

        return self.coord.submit(_job)

    def close(self) -> None:
        self.coord.shutdown(wait=True)
        self.pool.shutdown(wait=True)
        if self._cufile is not None:
            for ptr in self._registered:
                self._cufile.buf_deregister(ptr)
            self._registered.clear()
        for rec in self.files:
            if rec.fh is not None and self._cufile is not None:
                self._cufile.handle_deregister(rec.fh)
                rec.fh = None
            if rec.fd >= 0:
                os.close(rec.fd)
                rec.fd = -1
