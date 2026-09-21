# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import errno
import fcntl
import glob
import mmap
import os
import time

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_MMAP_GLOB = "/dev/shm/vllm_offload_*.mmap"

# flock이 무의미한(미지원) 파일시스템에서 나는 errno들 — 이때만 조용히 포기한다.
_FLOCK_UNSUPPORTED = (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL)


def _hold_shared_lock(fd: int, path: str) -> None:
    """Mark this region as in use for as long as the process lives.

    The lock is advisory and shared, so every participant of the same engine
    holds it at once; the kernel releases it when the process exits, however
    it exits (including SIGKILL).  _reclaim_orphaned_regions reads it to tell
    a live region from an orphan.

    The acquisition BLOCKS: the only writers that ever hold the exclusive
    lock are reclaim scanners, and only momentarily, so blocking here waits
    microseconds instead of silently leaving this participant lock-less for
    its lifetime (which would make its live region reclaimable).  Only a
    filesystem without flock support is excused, and merely costs the
    liveness signal.
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
    except OSError as e:
        if e.errno in _FLOCK_UNSUPPORTED:
            logger.debug("No flock support for %s; liveness signal disabled", path)
            return
        raise


def _unlink_if_same_inode(fd: int, path: str) -> bool:
    """Unlink path only if it still names the inode behind fd.

    Every unlink in this module follows a lock taken on an fd (an inode),
    while unlink targets a path.  Without this re-verification a scanner
    could delete a fresh LIVE region that re-occupied the path after the
    inode it actually locked was reclaimed (path-vs-inode TOCTOU).
    """
    try:
        st_fd = os.fstat(fd)
        st_path = os.stat(path)
    except OSError:
        return False
    if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
        return False
    try:
        os.unlink(path)
    except OSError:
        return False
    return True


def _try_reclaim(path: str, *, allow_empty: bool) -> bool:
    """Reclaim path if it is an orphan.  Returns True when unlinked.

    An orphan is a region nobody can be holding: every participant keeps a
    shared flock for its lifetime and the kernel drops it on process death,
    so winning LOCK_EX|LOCK_NB proves the file is dead.  The unlink is inode
    verified (see _unlink_if_same_inode).  allow_empty=False additionally
    skips zero-length files — a cross-engine scanner cannot tell a crashed
    creator's empty file from one inside the O_EXCL-create-to-flock window,
    and an empty file costs no tmpfs budget anyway.  Same-engine_id restarts
    pass allow_empty=True: leaving the empty file would wedge that engine_id
    forever in _wait_for_file_size.
    """
    try:
        # /dev/shm is world-writable; refuse to follow a planted symlink.
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        if not allow_empty and os.fstat(fd).st_size == 0:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False  # still held: a live engine owns this region
        return _unlink_if_same_inode(fd, path)
    finally:
        os.close(fd)


def _reclaim_orphaned_regions(own_path: str) -> None:
    """Remove offload regions left behind by an engine that died hard.

    cleanup() unlinks the region on graceful shutdown, but SIGKILL, an OOM
    kill or a segfault skips it, and the orphan keeps consuming the tmpfs
    budget until /dev/shm fills up (at which point new regions fail their
    MADV_POPULATE_WRITE with EFAULT).  Live regions are never touched:
    their holders keep LOCK_SH, so the LOCK_EX probe in _try_reclaim fails.
    """
    for path in glob.glob(_MMAP_GLOB):
        if path == own_path:
            continue
        if _try_reclaim(path, allow_empty=False):
            logger.info("Reclaimed orphaned offload mmap file %s", path)


def _wait_for_file_size(fd: int, expected_size: int, timeout: float = 30.0) -> None:
    """Spin-wait until the file reaches expected_size (creator truncated it)."""
    deadline = time.monotonic() + timeout
    while True:
        if os.fstat(fd).st_size >= expected_size:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for mmap file to reach {expected_size} bytes"
            )
        time.sleep(0.005)


class SharedOffloadRegion:
    """
    Single mmap-backed memory region shared across all workers for a
    vLLM instance.  Workers coordinate via the filesystem: the first worker
    to open the file with O_EXCL becomes the creator and calls ftruncate;
    the rest open the existing file and wait until it reaches the expected
    size.  Each worker then mmap()s the full file.

    File path: /dev/shm/vllm_offload_{engine_id}.mmap
    """

    BLOCK_SIZE_ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        engine_id: str,
        num_blocks: int,
        rank: int | None,
        kv_bytes_per_block: int,
        cpu_page_size: int,
    ) -> None:
        self.page_size = mmap.PAGESIZE
        assert kv_bytes_per_block % self.page_size == 0

        self.num_blocks = num_blocks
        self._row_stride = kv_bytes_per_block
        self.total_size_bytes = self.num_blocks * self._row_stride

        self.mmap_path = f"/dev/shm/vllm_offload_{engine_id}.mmap"
        self._creator = False  # set True only if this worker creates the file
        self.rank = rank
        if rank is not None:
            # byte offset to this worker's first slot within each block row
            self._worker_offset = rank * cpu_page_size
            # exclusive upper bound for this worker's area within each row
            self._worker_area_end = (rank + 1) * cpu_page_size
        # Initialize every resource field up-front so any failure below can
        # be handled by the one real teardown path, cleanup().
        self.fd: int | None = None
        self.mmap_obj: mmap.mmap | None = None
        self._base: torch.Tensor | None = None
        self._views: list[torch.Tensor] = []
        self.is_pinned: bool = False

        # Reap regions orphaned by previous engines that died without
        # cleanup.  Inode-verified and lock-gated, so concurrent scanners
        # from sibling processes are harmless (see _try_reclaim).
        _reclaim_orphaned_regions(self.mmap_path)

        try:
            self._open_or_join()
            self.mmap_obj = mmap.mmap(
                self.fd,
                self.total_size_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )

            # MADV_POPULATE_WRITE was added in Linux 5.14 (value 23).
            _MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
            if rank is not None:
                # Populate only this worker's pages (one slot per block row).
                worker_offset = rank * cpu_page_size
                _t0 = time.perf_counter()
                page_size = self.page_size
                for block in range(num_blocks):
                    raw_offset = block * self._row_stride + worker_offset
                    aligned_offset = (raw_offset // page_size) * page_size
                    end = raw_offset + cpu_page_size
                    aligned_length = end - aligned_offset
                    self.mmap_obj.madvise(
                        _MADV_POPULATE_WRITE, aligned_offset, aligned_length
                    )
                logger.debug(
                    "MADV_POPULATE_WRITE loop: %d blocks in %.3f s",
                    num_blocks,
                    time.perf_counter() - _t0,
                )
            else:
                # No rank — populate the entire shared region in one call.
                _t0 = time.perf_counter()
                self.mmap_obj.madvise(
                    _MADV_POPULATE_WRITE, 0, self.total_size_bytes
                )
                logger.debug(
                    "MADV_POPULATE_WRITE entire region: %.3f s",
                    time.perf_counter() - _t0,
                )

            self._base = torch.frombuffer(
                memoryview(self.mmap_obj), dtype=torch.int8
            )
        except BaseException:
            self.cleanup()
            raise

    def _inode_matches_path(self, fd: int) -> bool:
        try:
            st_fd = os.fstat(fd)
            st_path = os.stat(self.mmap_path)
        except OSError:
            return False
        return (st_fd.st_dev, st_fd.st_ino) == (st_path.st_dev, st_path.st_ino)

    def _open_or_join(self) -> None:
        """Create the region file or join an existing live one.

        Retries around the races this lifecycle admits: a same-engine_id
        orphan being reclaimed by a sibling, the path being unlinked between
        our EEXIST and our open, and the winner of a reclaim recreating the
        file while we were joining the old inode.  Every exit re-verifies
        that the fd we keep still names the file at mmap_path, so no
        participant can end up mapped to a detached inode (split brain).
        """
        for attempt in range(8):
            try:
                # Exclusive create — only one participant succeeds.
                fd = os.open(
                    self.mmap_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                )
            except FileExistsError:
                try:
                    fd = os.open(self.mmap_path, os.O_RDWR | os.O_NOFOLLOW)
                except FileNotFoundError:
                    continue  # unlinked between EEXIST and open — retry
                except OSError as e:
                    if e.errno == errno.ELOOP:
                        raise RuntimeError(
                            f"{self.mmap_path} is a symlink; refusing to use it"
                        ) from e
                    raise
                # A file nobody holds is an orphan of a previous run with
                # this same engine_id: reclaim it (empty ones included —
                # leaving those would wedge this engine_id in
                # _wait_for_file_size forever) and retry creating, so the
                # restart gets a fresh region with a real creator.
                # An EMPTY file gets a short grace first: a racing sibling
                # sits un-flocked between its O_EXCL create and its flock
                # for microseconds, and stealing its file in that window
                # would churn creators.  A genuinely dead empty corpse is
                # still reclaimed once the grace attempts are spent.
                if os.fstat(fd).st_size == 0 and attempt < 3:
                    os.close(fd)
                    time.sleep(0.002)
                    continue
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    pass  # held by someone — likely live, try to join below
                else:
                    reclaimed = _unlink_if_same_inode(fd, self.mmap_path)
                    os.close(fd)
                    if reclaimed:
                        logger.info(
                            "Reclaimed orphaned offload mmap file %s "
                            "(same engine_id restart)",
                            self.mmap_path,
                        )
                    continue
                _hold_shared_lock(fd, self.mmap_path)
                # The blocking lock may have waited out a reclaimer that
                # unlinked (and a sibling recreated) the file: only join if
                # our inode is still the one the path names.
                self.fd = fd
                if not self._inode_matches_path(fd):
                    self.fd = None
                    os.close(fd)
                    continue
                _wait_for_file_size(fd, self.total_size_bytes)
                logger.info("Opened existing mmap file %s", self.mmap_path)
                return
            # Creator path.  Lock before ftruncate: from here the file is
            # either zero-length (never reclaimed by cross-engine scanners)
            # or sized-and-locked.  A same-id sibling may still have stolen
            # the empty file in the open-to-flock window, so re-verify.
            _hold_shared_lock(fd, self.mmap_path)
            self.fd = fd
            if not self._inode_matches_path(fd):
                self.fd = None
                os.close(fd)
                continue
            # Creator flag before ftruncate: if sizing fails, cleanup() then
            # removes the empty file instead of leaking it.
            self._creator = True
            os.ftruncate(fd, self.total_size_bytes)
            logger.info(
                "Created mmap file %s (%.2f GB)",
                self.mmap_path,
                self.total_size_bytes / 1e9,
            )
            return
        raise RuntimeError(
            f"Could not create or join offload region {self.mmap_path}"
        )

    def create_next_view(self, tensor_page_size: int) -> torch.Tensor:
        """Allocate a strided int8 view for this worker, one canonical tensor.

        Must be called once per canonical tensor. The full mmap layout is:

            worker0_block0 | worker1_block0 | ... | worker{M-1}_block0
            worker0_block1 | worker1_block1 | ... | worker{M-1}_block1
            ...

        Each worker_block cell is cpu_page_size bytes and holds all canonical
        tensors for that worker and block concatenated:
            [ tensor0_data | tensor1_data | ... | tensor{L-1}_data ]

        Consecutive rows are separated by row_stride = cpu_page_size * M.

        Returns an int8 tensor of shape (num_blocks, tensor_page_size) with stride
        (row_stride, 1).  Using int8 keeps stride == bytes, so swap_blocks
        address arithmetic works without any dtype conversion.

        Args:
            tensor_page_size: Bytes per block for this  tensor.
        """
        assert self.rank is not None
        new_offset = self._worker_offset + tensor_page_size
        assert new_offset <= self._worker_area_end, (
            f"Worker offset {new_offset} exceeds worker area end "
            f"{self._worker_area_end} (overflowed by "
            f"{new_offset - self._worker_area_end} bytes)"
        )
        worker_layer_view = torch.as_strided(
            self._base,
            size=(self.num_blocks, tensor_page_size),
            stride=(self._row_stride, 1),
            storage_offset=self._worker_offset,
        )
        self._worker_offset = new_offset
        self._views.append(worker_layer_view)
        return worker_layer_view

    def create_kv_memoryview(self) -> memoryview:
        """Return a zero-copy memoryview over the entire KV buffer.

        Shape: (num_blocks, row_stride_bytes). Secondary tiers address
        block *b* as ``view[b]``.
        """
        kv_tensor = self._base.view(self.num_blocks, self._row_stride)
        np_arr = kv_tensor.numpy()
        assert np_arr.ctypes.data == self._base.data_ptr(), (
            "view()/numpy() created a copy instead of sharing the mmap buffer; "
            "secondary tiers require zero-copy access to primary KV data"
        )
        return memoryview(np_arr)

    def cleanup(self) -> None:
        if self.is_pinned and self._base is not None:
            if current_platform.is_cuda_alike():
                base_ptr = self._base.data_ptr()
                result = torch.cuda.cudart().cudaHostUnregister(base_ptr)
                if result.value != 0:
                    logger.warning(
                        "cudaHostUnregister failed for rank=%d (code=%d)",
                        self.rank,
                        result,
                    )
            self.is_pinned = False
        # Release views before _base: each view holds a _base reference and a
        # direct StorageImpl reference.  Freeing views first lets both refcounts
        # drop so the storage (which holds the mmap_obj buffer export) is freed
        # before mmap_obj.close() is called below.
        if self._views is not None:
            self._views.clear()
        self._base = None
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except Exception:
                logger.warning("Failed to close mmap_obj", exc_info=True)
            self.mmap_obj = None
        # Unlink BEFORE closing the fd: while the fd (and our flock) is still
        # held, the inode check below cannot race a reclaimer, and closing
        # first would drop our liveness lock while the file is still linked.
        if self._creator and getattr(self, "mmap_path", None):
            if self.fd is not None and _unlink_if_same_inode(self.fd, self.mmap_path):
                logger.info("Removed mmap file %s", self.mmap_path)
            elif self.fd is None:
                # No fd to verify against (already closed by an earlier
                # cleanup attempt) — best-effort by path.
                with contextlib.suppress(OSError):
                    os.unlink(self.mmap_path)
            else:
                logger.debug(
                    "Not unlinking %s: path no longer names our inode",
                    self.mmap_path,
                )
            self._creator = False
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                logger.warning("Failed to close fd %s", self.fd, exc_info=True)
            self.fd = None
