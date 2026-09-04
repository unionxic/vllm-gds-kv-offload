# 4단계 원인 분해: store 경로 구성요소를 하나씩만 남긴다 (ISOLATE=... 일 때만 로드).
#   이 단계는 의도적으로 store 동작을 치환한다 — 성능 격리용, 정확성 무관.
#   파일은 truncate로 크기만 만들어 rename하므로(희소 파일) 이후 load 경로의
#   IO 볼륨은 동일하다(내용은 0 — 출력 토큰 검증은 이 단계에서 무의미).
# 변형:
#   ctrl        queue/job/rename/완료 처리만. ev.sync 없음, IO 없음 → Python 제어부·GIL
#   native_wait ctrl + WRITE_MS만큼 GIL 놓는 native 대기 → 스레드 스케줄링·convoy
#   cuda_only   ev.synchronize + span/data_ptr 접근 + truncate/rename → CUDA context·driver
#   posix_only  준비된 pinned buffer를 O_DIRECT pwrite (D2H·ev.sync 없음) → syscall·fs
#   cufile_only GPU span을 cuFile write (ev.sync 없음) → cuFile·nvidia-fs
#   full        원본 그대로
import ctypes
import os
import threading
import time

import torch

MODE = os.environ.get("ISOLATE", "full")
WRITE_MS = float(os.environ.get("WRITE_MS", "0"))

_tls = threading.local()


def _commit_sparse(path, nbytes):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.ftruncate(fd, nbytes)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _pinned_addr(nbytes):
    # O_DIRECT용 4KiB 정렬 주소 반환
    need = nbytes + 8192
    buf = getattr(_tls, "buf", None)
    if buf is None or buf.numel() < need:
        _tls.buf = torch.empty(need, dtype=torch.uint8, pin_memory=True)
    a = _tls.buf.data_ptr()
    return (a + 4095) & ~4095


def install():
    if MODE == "full":
        return
    import expfs

    def _store_task(self, ev, path, spans):
        with torch.cuda.nvtx.range(f"ISOLATE_{MODE}"):
            nbytes = sum(s[2] for s in spans)
            if MODE == "cuda_only":
                ev.synchronize()
                for t, boff, size, foff in spans:  # pointer 산술·텐서 접근만
                    _ = self.transport.flat[t].data_ptr() + boff if hasattr(
                        self.transport, "flat") else size
                _commit_sparse(path, nbytes)
            elif MODE == "ctrl":
                _commit_sparse(path, nbytes)
            elif MODE == "native_wait":
                time.sleep(WRITE_MS / 1000.0)  # sleep은 GIL을 놓는다
                _commit_sparse(path, nbytes)
            elif MODE == "posix_only":
                addr = _pinned_addr(nbytes)
                direct = getattr(os, "O_DIRECT", 0) if nbytes % 4096 == 0 else 0
                tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | direct,
                             0o644)
                try:
                    mv = memoryview((ctypes.c_char * nbytes).from_address(addr))
                    off = 0
                    while off < nbytes:
                        off += os.pwrite(fd, mv[off:], off)
                finally:
                    os.close(fd)
                os.replace(tmp, path)
            elif MODE == "cufile_only":
                self.transport.write_chunk(path, spans, self.chunk_bytes)
            else:
                raise ValueError(MODE)
    expfs.FilesystemWorker._store_task = _store_task
    print(f"ISOLATE mode={MODE} write_ms={WRITE_MS}", flush=True)
