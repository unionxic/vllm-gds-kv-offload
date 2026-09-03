# ExperimentalFilesystemSpec — worker-side GDS filesystem KV offload backend.
#
# v0.26.1.dev(568afb3a13)·TP=1·단일 KV group·uniform full-attention 전용
# 성능 검증 prototype. GPU pointer/span 해석을 backend가 직접 수행하며
# upstream-ready 범용 구현(RFC #48504의 공통 worker refactor)을 목표로 하지 않는다.
#
# v2 (2026-08-31): blocks_per_chunk>1 지원 + 인접 GPU 블록 span 병합.
#   chunk 파일 레이아웃 = [tensor0: bpc pages][tensor1: bpc pages]... 연결.
#   병합: 논리 연속 페이지의 GPU 블록 id가 연속이면 단일 대형 IO — GIL 콘보이
#   (nsys 진단: 엔진 busy loop의 스위치 인터벌 독점)와 호출 오버헤드를 함께 줄인다.
#
# v3 (2026-09-03): cufile_staged — GPU staging ring transport.
#   원인 규명 결론(tail = store의 GPU 블록 점유가 요청 경계를 침범)의 설계 응답.
#   KV 블록 → (D2D, side stream) → ring slot → cuFile 단일 대형 write.
#   store job 완료 = 복사 완료(블록 즉시 반환, tiering의 수명 분리와 동일 의미).
#   SSD 쓰기는 슬롯에서 writer 스레드가 비동기 수행 — 산발 span이 슬롯에서
#   연속화되어 chunk당 1회 대형 IO(마이크로벤치의 등록 GDS 우위 레짐).
#   의미론: 파일은 쓰기 완료 시 rename으로 나타남 — 완료 보고~파일 출현 사이
#   lookup은 miss로 안전(재계산 폴백). 쓰기 실패도 파일 부재 = miss로 안전.
#
# 사용:
#   PYTHONPATH+=:~/experiments/vllm-gds-kv/phase2:~/experiments/vllm-gds-kv/phase1
#   kv_connector_extra_config = {
#     "spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
#     "expfs_root_dir": ..., "expfs_transport": "cufile"|"posix",
#     "expfs_cufile_mode": "unregistered"|"registered_tensors",
#     "block_size": 16|64|256,  # GPU블록 16의 배수 → blocks_per_chunk = block_size/16
#   }
import os
import queue
import sys
import threading
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import partial

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1"))
from gdslib import Gds, GdsError  # noqa: E402

from vllm.logger import init_logger  # noqa: E402
from vllm.v1.kv_offload.base import (  # noqa: E402
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig  # noqa: E402
from vllm.v1.kv_offload.file_mapper import FileMapper  # noqa: E402
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool  # noqa: E402

logger = init_logger(__name__)

ALIGN = 4096

# 하네스 계측용: 마지막으로 생성된 worker (in-process 실험 전용 훅)
LAST_WORKER = None


@dataclass
class FileLoadStoreSpec(LoadStoreSpec):
    """chunk당 파일 1개. paths[i] = i번째 chunk의 최종 경로."""

    paths: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"FileLoadStoreSpec(n={len(self.paths)})"


# ---------------- scheduler-side ----------------


class FilesystemManager(OffloadingManager):
    def __init__(self, file_mapper: FileMapper):
        self.file_mapper = file_mapper
        # 최종 파일은 temp→rename 원자적 생성이라 exists()=로드 가능.
        # pending은 동일 chunk 중복 store 방지용(정확성 아닌 효율).
        self._pending_stores: set[OffloadKey] = set()

    def _path(self, key: OffloadKey) -> str:
        return self.file_mapper.get_file_name(key)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return LookupResult.HIT if os.path.exists(self._path(key)) else LookupResult.MISS

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        return FileLoadStoreSpec([self._path(k) for k in keys])

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        to_store = [
            k for k in keys
            if k not in self._pending_stores and not os.path.exists(self._path(k))
        ]
        self._pending_stores.update(to_store)
        return PrepareStoreOutput(
            keys_to_store=to_store,
            store_spec=FileLoadStoreSpec([self._path(k) for k in to_store]),
            evicted_keys=[],
        )

    def complete_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True
    ):
        self._pending_stores.difference_update(keys)

    def reset_cache(self) -> None:
        self._pending_stores.clear()


# ---------------- transports ----------------
# 통일 span 형식: (tensor_idx, gpu_byte_off, size, file_off)
#   gpu_byte_off = flat[t] 텐서 내 바이트 오프셋 (병합 span은 여러 페이지 연속)


class CuFileTransport:
    name = "cufile"

    def __init__(self, flat_tensors: list[torch.Tensor], mode: str):
        self.flat = flat_tensors
        self.g = Gds()
        self.registered = False
        if mode == "registered_tensors":
            done = []
            try:
                for t in flat_tensors:
                    self.g.buf_register(t.data_ptr(), t.numel())
                    done.append(t.data_ptr())
                self.registered = True
                logger.info("expfs: registered %d KV tensors as GDS buffers", len(done))
            except GdsError as e:
                for p in done:
                    self.g.buf_deregister(p)
                logger.warning(
                    "expfs: KV tensor GDS registration failed (%s) — "
                    "falling back to unregistered cuFile IO", e)

    def write_chunk(self, path: str, spans, chunk_bytes: int):
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_DIRECT", 0), 0o644)
        try:
            fh = self.g.handle_register(fd)
            try:
                for t, boff, size, foff in spans:
                    n = self.g.write(fh, self.flat[t].data_ptr() + boff, size, foff)
                    if n != size:
                        raise OSError(f"short cuFileWrite {n}/{size} @{path}+{foff}")
            finally:
                self.g.handle_deregister(fh)
            os.close(fd)
            fd = None
            os.replace(tmp, path)
        except Exception:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def read_chunk(self, path: str, spans, chunk_bytes: int):
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        try:
            fh = self.g.handle_register(fd)
            try:
                for t, boff, size, foff in spans:
                    n = self.g.read(fh, self.flat[t].data_ptr() + boff, size, foff)
                    if n != size:
                        raise OSError(f"short cuFileRead {n}/{size} @{path}+{foff}")
            finally:
                self.g.handle_deregister(fh)
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)

    def close(self):
        self.g.close()


class StagedCuFileTransport(CuFileTransport):
    """GPU staging ring: 수명 분리 + GPU측 코얼레싱 + 스핀 없는 대기 (v3).

    store 경로: stage_chunk()이 IO 스레드에서 슬롯 확보 → 스레드별 side stream에
    compute event 의존을 걸고 span들을 슬롯에 연속 배치로 D2D 복사 → blocking
    event로 복사 완료 대기(스핀 없음) → 슬롯을 writer 큐에 넘기고 반환.
    반환 시점 = job 완료 = KV 블록 반환 가능. writer 스레드가 슬롯 전체를
    cuFile 1회 IO로 temp에 쓰고 rename, 슬롯 반납.
    load 경로: CuFileTransport.read_chunk 상속(무변경).
    """

    name = "cufile_staged"

    def __init__(self, flat_tensors: list[torch.Tensor], chunk_bytes: int,
                 n_slots: int, n_writers: int, register_ring: bool,
                 io_kind: str = "cufile", policy: str = "block",
                 max_outstanding_write_bytes: int | None = None,
                 cpu_fallback_slots: int = 0):
        super().__init__(flat_tensors, "unregistered")
        self.io_kind = io_kind  # "cufile" | "posix" (호스트 pinned 링 + pwrite)
        # policy: "block"(v3.1, 슬롯 대기) | "skip"(포화 시 저장 생략)
        #         | "cpu_fallback"(포화 시 D2H 후 CPU에서 write)
        self.policy = policy
        self.max_ob = max_outstanding_write_bytes
        self._ob = 0  # outstanding write bytes (in-flight SSD write)
        self.stats = dict(ring_used=0, fallback=0, dropped=0, slot_full=0,
                          ob_full=0, read_deferred=0)
        self._read_active = 0  # SSD read in-flight (read-priority용)
        if io_kind == "posix":
            self.name = "posix_staged"
        self.flat1d = [t.view(-1) for t in flat_tensors]
        self.chunk_bytes = chunk_bytes
        self.n_slots = n_slots
        # cuFile·O_DIRECT용 4KiB 정렬 링 (chunk_bytes는 page 합이라 ALIGN 배수)
        if io_kind == "posix":
            raw = torch.zeros(n_slots * chunk_bytes + ALIGN, dtype=torch.uint8,
                              pin_memory=True)
        else:
            raw = torch.zeros(n_slots * chunk_bytes + ALIGN, dtype=torch.uint8,
                              device="cuda")
        ali = (-raw.data_ptr()) % ALIGN
        self._raw = raw
        self.ring = raw[ali: ali + n_slots * chunk_bytes].view(n_slots, chunk_bytes)
        self.ring_registered = False
        if register_ring and io_kind == "cufile":
            try:
                self.g.buf_register(self.ring.data_ptr(), self.ring.numel())
                self.ring_registered = True
                logger.info("expfs: staging ring %.0f MiB GDS-registered (%d slots)",
                            self.ring.numel() / 2**20, n_slots)
            except GdsError as e:
                logger.warning(
                    "expfs: ring GDS registration failed (%s) — unregistered IO", e)
        self._tls = threading.local()
        self._free: queue.Queue[int] = queue.Queue()
        for i in range(n_slots):
            self._free.put(i)
        # CPU fallback 풀: 포화 시 D2H 복사 대상 (호스트 pinned, 4KiB 정렬)
        self.n_cpu = cpu_fallback_slots
        self._cpu_free: queue.Queue[int] = queue.Queue()
        if cpu_fallback_slots > 0:
            craw = torch.zeros(cpu_fallback_slots * chunk_bytes + ALIGN,
                               dtype=torch.uint8, pin_memory=True)
            cali = (-craw.data_ptr()) % ALIGN
            self._craw = craw
            self.cpu_ring = craw[cali: cali + cpu_fallback_slots * chunk_bytes].view(
                cpu_fallback_slots, chunk_bytes)
            for i in range(cpu_fallback_slots):
                self._cpu_free.put(i)
        self._wq: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.write_errors = 0
        self._writers = [
            threading.Thread(target=self._writer_loop, daemon=True,
                             name=f"expfs_stw{i}")
            for i in range(n_writers)]
        for th in self._writers:
            th.start()

    def _side_stream(self) -> torch.cuda.Stream:
        s = getattr(self._tls, "stream", None)
        if s is None:
            s = torch.cuda.Stream()
            self._tls.stream = s
        return s

    # ---- v3.1: 스레드 점유 없는 staging 프리미티브 ----
    # v3의 stage_chunk는 IO 스레드가 compute event를 기다리며 스레드를 점유해
    # 큐 대기가 블록 점유 시간에 더해지는 결함이 있었다(계측: queue_wait 1.13s).
    # v3.1은 제출 스레드가 복사를 비동기 인큐만 하고, 완료는 worker의 _drain이
    # event.query() 비차단 폴링으로 보고한다. IO 스레드 관여 0.

    def try_acquire_slot(self):
        try:
            return self._free.get_nowait()
        except queue.Empty:
            return None

    def try_acquire_cpu(self):
        try:
            return self._cpu_free.get_nowait()
        except queue.Empty:
            return None

    def _launch_into(self, dst, ev, spans):
        """dst(ring slot 또는 cpu buffer)로 span들을 비동기 복사, 완료 event 반환."""
        side = self._side_stream()
        with torch.cuda.stream(side):
            side.wait_event(ev)  # compute 의존은 GPU 안에서
            with torch.cuda.nvtx.range("STAGE_COPY"):
                for t, boff, size, foff in spans:
                    dst[foff:foff + size].copy_(
                        self.flat1d[t][boff:boff + size], non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
        return done

    def launch_copy(self, ev: torch.cuda.Event, slot: int, spans):
        """ring slot으로 D2D 복사 (source_released = 이 event fire)."""
        return self._launch_into(self.ring[slot], ev, spans)

    def launch_cpu_copy(self, ev: torch.cuda.Event, cslot: int, spans):
        """CPU fallback buffer로 D2H 복사 (source_released = 이 event fire)."""
        return self._launch_into(self.cpu_ring[cslot], ev, spans)

    def writer_submit(self, slot: int, path: str, kind: str = "ring"):
        # store_committed = writer가 SSD write+rename 완료 시점 (D2D/D2H 완료 ≠ commit)
        with self._lock:
            self._ob += self.chunk_bytes
        self._wq.put((kind, slot, path))

    def outstanding_write_bytes(self):
        with self._lock:
            return self._ob

    def note_read_begin(self):
        with self._lock:
            self._read_active += 1

    def note_read_end(self):
        with self._lock:
            self._read_active = max(0, self._read_active - 1)

    def reads_active(self):
        with self._lock:
            return self._read_active > 0

    def stage_chunk(self, ev: torch.cuda.Event, path: str, spans):
        """(v3 경로, 참조용 보존) IO 스레드에서 실행. 반환 = 복사 완료."""
        slot = self._free.get()  # ring 포화 시 여기서 대기 (blocking, GIL 해제)
        try:
            dst = self.ring[slot]
            side = self._side_stream()
            with torch.cuda.stream(side):
                side.wait_event(ev)  # GPU 안에서 compute 의존 — CPU 대기 아님
                with torch.cuda.nvtx.range("STAGE_COPY"):
                    for t, boff, size, foff in spans:
                        dst[foff:foff + size].copy_(
                            self.flat1d[t][boff:boff + size], non_blocking=True)
                done = torch.cuda.Event(blocking=True)
                done.record(side)
            done.synchronize()  # blocking flag → sleep 대기, 스핀 없음
        except Exception:
            # 인큐된 복사가 남아 있으면 늦게 착지해 재사용 슬롯을 오염시킬 수
            # 있으므로, side stream을 비운 뒤에만 슬롯을 반납한다 (QA 결함 3).
            try:
                self._side_stream().synchronize()
            except Exception:
                pass
            self._free.put(slot)
            raise
        self.writer_submit(slot, path, "ring")

    def _writer_loop(self):
        while True:
            item = self._wq.get()
            if item is None:
                return
            kind, slot, path = item
            try:
                with torch.cuda.nvtx.range("STAGE_WRITE"):
                    self.write_slot(slot, path, kind)
            except Exception as e:
                # 파일 부재 = 이후 lookup miss = 재계산 폴백이라 안전. 계수만.
                self.write_errors += 1
                logger.error("expfs staged write failed for %s: %s", path, e)
            finally:
                with self._lock:
                    self._ob -= self.chunk_bytes
                if kind == "cpu":
                    self._cpu_free.put(slot)
                else:
                    self._free.put(slot)

    def write_slot(self, slot: int, path: str, kind: str = "ring"):
        # kind="cpu"면 호스트 pinned buffer에서 pwrite (GPU-direct 불가)
        src_ptr = (self.cpu_ring[slot].data_ptr() if kind == "cpu"
                   else self.ring[slot].data_ptr())
        use_cufile = (self.io_kind == "cufile" and kind == "ring")
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR
                     | getattr(os, "O_DIRECT", 0), 0o644)
        try:
            if not use_cufile:
                import ctypes
                mv = memoryview((ctypes.c_char * self.chunk_bytes).from_address(
                    src_ptr))
                off = 0
                while off < self.chunk_bytes:
                    off += os.pwrite(fd, mv[off:], off)
            else:
                fh = self.g.handle_register(fd)
                try:
                    n = self.g.write(fh, self.ring[slot].data_ptr(),
                                     self.chunk_bytes, 0)
                    if n != self.chunk_bytes:
                        raise OSError(
                            f"short cuFileWrite {n}/{self.chunk_bytes} @{path}")
                finally:
                    self.g.handle_deregister(fh)
            os.close(fd)
            fd = None
            os.replace(tmp, path)
        except Exception:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def flush(self, timeout: float = 60.0) -> bool:
        """잔여 쓰기 완료 대기 — 종료·검증·wall 확정용. True = 전량 배출."""
        deadline = time.monotonic() + timeout
        while (not self._wq.empty() or self._free.qsize() < self.n_slots
               or self._cpu_free.qsize() < self.n_cpu):
            if time.monotonic() > deadline:
                logger.warning("expfs staged flush timeout: wq=%d free=%d/%d cpu=%d/%d",
                               self._wq.qsize(), self._free.qsize(), self.n_slots,
                               self._cpu_free.qsize(), self.n_cpu)
                return False
            time.sleep(0.002)
        return True

    def close(self):
        drained = self.flush()
        for _ in self._writers:
            self._wq.put(None)
        joined = True
        for th in self._writers:
            th.join(timeout=10)
            joined = joined and not th.is_alive()
        if not (drained and joined):
            # in-flight write가 남았을 수 있음 — 드라이버 자원 해제를 건너뛰어
            # 쓰기 도중 deregister/close로 인한 크래시를 피한다 (QA 결함 4).
            logger.warning("expfs staged close: writers not fully drained — "
                           "skipping cuFile deregister/close")
            return
        if self.ring_registered:
            try:
                self.g.buf_deregister(self.ring.data_ptr())
            except Exception:
                pass
        super().close()


class PosixBounceTransport:
    """control plane 동일, 전송만 pinned host bounce + O_DIRECT (Phase 3 E군)."""

    name = "posix"

    def __init__(self, flat_tensors: list[torch.Tensor], chunk_bytes: int):
        self.flat = [t.view(-1) for t in flat_tensors]  # 바이트 오프셋 슬라이스용 1-D 뷰
        self.chunk_bytes = chunk_bytes
        self._tls = threading.local()

    def _pin(self):
        if not hasattr(self._tls, "buf"):
            self._tls.buf = torch.empty(self.chunk_bytes, dtype=torch.uint8, pin_memory=True)
            self._tls.mv = memoryview(self._tls.buf.numpy())
        return self._tls.buf, self._tls.mv

    @staticmethod
    def _contiguous(spans):
        """foff 정렬 기준으로 파일 구간이 빈틈없이 연속인가 (단일 IO 가능 여부)."""
        s = sorted(spans, key=lambda x: x[3])
        return all(s[i][3] + s[i][2] == s[i + 1][3] for i in range(len(s) - 1)), s

    def write_chunk(self, path: str, spans, chunk_bytes: int):
        buf, mv = self._pin()
        contig, s = self._contiguous(spans)
        base = s[0][3]
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_DIRECT", 0), 0o644)
        try:
            if contig:
                for t, boff, size, foff in s:
                    buf[foff - base:foff - base + size].copy_(
                        self.flat[t][boff:boff + size], non_blocking=True)
                torch.cuda.synchronize()
                nbytes = sum(x[2] for x in s)
                n = os.pwrite(fd, mv[:nbytes], base)
                if n != nbytes:
                    raise OSError(f"short write {n}/{nbytes} @{path}")
            else:  # 불연속 (다중 tensor + partial chunk): span별 IO
                for t, boff, size, foff in s:
                    buf[:size].copy_(self.flat[t][boff:boff + size], non_blocking=True)
                    torch.cuda.synchronize()
                    n = os.pwrite(fd, mv[:size], foff)
                    if n != size:
                        raise OSError(f"short write {n}/{size} @{path}+{foff}")
            os.close(fd)
            fd = None
            os.replace(tmp, path)
        except Exception:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def read_chunk(self, path: str, spans, chunk_bytes: int):
        buf, mv = self._pin()
        contig, s = self._contiguous(spans)
        base = s[0][3]
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        try:
            if contig:
                nbytes = sum(x[2] for x in s)
                n = os.preadv(fd, [mv[:nbytes]], base)
                if n != nbytes:
                    raise OSError(f"short read {n}/{nbytes} @{path}")
                for t, boff, size, foff in s:
                    self.flat[t][boff:boff + size].copy_(
                        buf[foff - base:foff - base + size], non_blocking=True)
                torch.cuda.synchronize()
            else:
                for t, boff, size, foff in s:
                    n = os.preadv(fd, [mv[:size]], foff)
                    if n != size:
                        raise OSError(f"short read {n}/{size} @{path}+{foff}")
                    self.flat[t][boff:boff + size].copy_(buf[:size], non_blocking=True)
                    torch.cuda.synchronize()
        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        finally:
            os.close(fd)

    def close(self):
        pass


# ---------------- worker-side ----------------


class FilesystemWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        transport_name: str,
        cufile_mode: str,
        n_read_threads: int,
        n_write_threads: int,
        staging_slots: int = 6,
        staging_writers: int = 2,
        staging_policy: str = "block",
        max_outstanding_write_bytes: int | None = None,
        cpu_fallback_slots: int = 0,
    ):
        assert len(kv_caches.group_data_refs) == 1, (
            "expfs prototype: single KV cache group only")
        self.bpc = blocks_per_chunk
        self.staging_policy = staging_policy

        self.flat: list[torch.Tensor] = []
        self.page_sizes: list[int] = []
        self.region_off: list[int] = []  # chunk 파일 내 tensor region 시작 오프셋
        off = 0
        for ct in kv_caches.tensors:
            page = ct.page_size_bytes
            assert page % ALIGN == 0, (
                f"expfs: page_size_bytes {page} not {ALIGN}-aligned — "
                "direct IO geometry unsupported for this model/backend")
            self.flat.append(ct.tensor.view(torch.int8).view(-1, page))
            self.page_sizes.append(page)
            self.region_off.append(off)
            off += page * self.bpc
        self.chunk_bytes = off

        if transport_name == "cufile":
            self.transport = CuFileTransport(self.flat, cufile_mode)
        elif transport_name in ("cufile_staged", "posix_staged"):
            self.transport = StagedCuFileTransport(
                self.flat, self.chunk_bytes, staging_slots, staging_writers,
                register_ring=(cufile_mode != "unregistered_ring"),
                io_kind=("posix" if transport_name == "posix_staged" else "cufile"),
                policy=staging_policy,
                max_outstanding_write_bytes=max_outstanding_write_bytes,
                cpu_fallback_slots=cpu_fallback_slots)
        elif transport_name == "posix":
            self.transport = PosixBounceTransport(self.flat, self.chunk_bytes)
        else:
            raise ValueError(f"unknown expfs_transport: {transport_name}")

        self.pool = DualQueueThreadPool(n_read_threads, n_write_threads,
                                        thread_name_prefix="expfs")
        self._pending: set[int] = set()
        self._drained: list[TransferResult] = []
        # v3.1 staged: job_id -> [entry], entry = {ev, path, spans, slot, done, committed}
        self._staged_jobs: dict[int, list[dict]] = {}
        self._load_jobs: set[int] = set()
        global LAST_WORKER
        LAST_WORKER = self
        logger.info(
            "expfs worker: %d tensors, bpc=%d, chunk=%d bytes, transport=%s",
            len(self.flat), self.bpc, self.chunk_bytes, transport_name)

    def _chunk_spans(self, bids: list[int], j0: int):
        """chunk의 span 목록 (병합 포함). bids[k] = 논리 페이지 (j0+k)의 GPU 블록 id."""
        spans = []
        for t in range(len(self.flat)):
            page = self.page_sizes[t]
            region = self.region_off[t]
            k = 0
            while k < len(bids):
                bid0, n = bids[k], 1
                while (k + n < len(bids)) and bids[k + n] == bid0 + n:
                    n += 1
                spans.append((t, bid0 * page, n * page, region + (j0 + k) * page))
                k += n
        return spans

    def _partition(self, gpu_spec: GPULoadStoreSpec, file_spec: FileLoadStoreSpec,
                   is_store: bool):
        """GPU 블록 목록을 chunk별 (path, bids, j0)로 분할."""
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        assert isinstance(file_spec, FileLoadStoreSpec)
        assert len(gpu_spec.group_sizes) == 1, "expfs prototype: single group only"
        bids = [int(b) for b in gpu_spec.block_ids]
        n_chunks = len(file_spec.paths)
        skip = int(gpu_spec.block_indices[0]) % self.bpc
        if is_store:
            assert skip == 0, "expfs: store must be chunk-aligned"
        assert len(bids) == n_chunks * self.bpc - skip, (
            f"expfs: {len(bids)} blocks vs {n_chunks} chunks (bpc={self.bpc}, skip={skip})")
        out, pos = [], 0
        for c, path in enumerate(file_spec.paths):
            j0 = skip if c == 0 else 0
            take = self.bpc - j0
            out.append((path, bids[pos:pos + take], j0))
            pos += take
        return out

    def _store_task(self, ev: torch.cuda.Event, path: str, spans):
        with torch.cuda.nvtx.range(f"expfs.store[{self.transport.name}]"):
            ev.synchronize()  # compute stream이 이 블록들의 KV를 완성한 뒤에만 읽기
            self.transport.write_chunk(path, spans, self.chunk_bytes)

    def _staged_store_task(self, ev: torch.cuda.Event, path: str, spans):
        # 반환 = 복사 완료 = job 완료 = 블록 반환. SSD 쓰기는 transport가 비동기로.
        with torch.cuda.nvtx.range("expfs.store[staged]"):
            self.transport.stage_chunk(ev, path, spans)

    def _load_task(self, path: str, spans):
        # 동기 IO — 반환 시 GPU 상주. 완료 보고 전엔 스케줄러가 요청을 안 돌림.
        with torch.cuda.nvtx.range(f"expfs.load[{self.transport.name}]"):
            self.transport.read_chunk(path, spans, self.chunk_bytes)

    def submit_store(self, job_id: int, src_spec: GPULoadStoreSpec,
                     dst_spec: LoadStoreSpec) -> bool:
        parts = self._partition(src_spec, dst_spec, is_store=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream())
        if isinstance(self.transport, StagedCuFileTransport):
            # v3.1: IO 스레드 미사용. 복사 비동기 인큐 → 완료는 _drain의 event
            # 폴링으로 보고. 블록 점유 = 실제 복사 완료 시점.
            self._staged_jobs[job_id] = [
                dict(ev=ev, path=path, spans=self._chunk_spans(bids, j0),
                     slot=None, done=None, committed=False, mode=None)
                for path, bids, j0 in parts]
            self._pending.add(job_id)
            self._advance_staged()
            return True
        tasks = [partial(self._store_task, ev, path, self._chunk_spans(bids, j0))
                 for path, bids, j0 in parts]
        self._pending.add(job_id)
        self.pool.enqueue_store(job_id, len(tasks), tasks)
        return True

    def _admit(self, e):
        """한 chunk entry의 복사를 시작(비차단). ring→cpu fallback→skip 순.
        어느 경로든 원본 블록은 이 복사(D2D/D2H) 완료로 해제 — 슬롯 대기 없음.
        반환 False = 아직 시작 못함(block 정책에서 슬롯 대기 재시도)."""
        tr = self.transport
        pol = self.staging_policy
        # (6) outstanding write bytes 상한 + (5) read-priority
        # 주의: _admit/_advance_staged/submit_store는 엔진 단일 스레드에서만 호출됨.
        # 아래 cap 검사와 writer_submit의 _ob 증가가 별도 lock이라도 admission이
        # 단일 스레드라 오버슛 없음. 멀티워커 포팅 시 이 전제가 깨지니 유의.
        ob_full = (tr.max_ob is not None
                   and tr.outstanding_write_bytes() + self.chunk_bytes > tr.max_ob)
        read_hold = (pol != "block" and tr.reads_active())
        if not ob_full and not read_hold:
            s = tr.try_acquire_slot()
            if s is not None:
                e["slot"], e["mode"] = s, "ring"
                e["done"] = tr.launch_copy(e["ev"], s, e["spans"])
                tr.stats["ring_used"] += 1
                return True
            tr.stats["slot_full"] += 1
        elif ob_full:
            tr.stats["ob_full"] += 1
        elif read_hold:
            tr.stats["read_deferred"] += 1
        # 포화/보류 상태. block 정책이면 대기(재시도), 아니면 비차단 처리:
        if pol == "block":
            return False
        if pol == "cpu_fallback" and tr.n_cpu > 0:
            c = tr.try_acquire_cpu()
            if c is not None:
                e["slot"], e["mode"] = c, "cpu"
                e["done"] = tr.launch_cpu_copy(e["ev"], c, e["spans"])
                tr.stats["fallback"] += 1
                return True
        # skip 정책 또는 cpu 버퍼도 없음 → 저장 생략(원본 즉시 해제, 파일 없음=miss)
        e["mode"] = "drop"
        e["committed"] = True
        tr.stats["dropped"] += 1
        return True

    def _advance_staged(self):
        """슬롯 획득·복사 인큐·완료 폴링을 한 바퀴 진행 (비차단, 엔진 스레드)."""
        tr = self.transport
        done_jobs = []
        for job_id, entries in self._staged_jobs.items():
            for e in entries:
                if e["committed"]:
                    continue
                if e["slot"] is None and e["mode"] != "drop":
                    if not self._admit(e):
                        continue  # block 정책: 슬롯 없음 — 다음 폴링 재시도
                    if e["mode"] == "drop":
                        continue
                if e["done"].query():  # 비차단
                    tr.writer_submit(e["slot"], e["path"], e["mode"])
                    e["committed"] = True
            if all(e["committed"] for e in entries):
                done_jobs.append(job_id)
        for j in done_jobs:
            del self._staged_jobs[j]
            self._pending.discard(j)
            self._drained.append(TransferResult(job_id=j, success=True))

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec,
                    dst_spec: GPULoadStoreSpec) -> bool:
        parts = self._partition(dst_spec, src_spec, is_store=False)
        tasks = [partial(self._load_task, path, self._chunk_spans(bids, j0))
                 for path, bids, j0 in parts]
        self._pending.add(job_id)
        if isinstance(self.transport, StagedCuFileTransport):
            self._load_jobs.add(job_id)
            self.transport.note_read_begin()  # (5) read-priority: write 보류 신호
        self.pool.enqueue_load(job_id, len(tasks), tasks)
        return True

    def _drain(self):
        if self._staged_jobs:
            self._advance_staged()
        for job_id, success in self.pool.get_finished():
            self._pending.discard(job_id)
            if job_id in self._load_jobs:
                self._load_jobs.discard(job_id)
                self.transport.note_read_end()
            self._drained.append(TransferResult(job_id=job_id, success=success))

    def get_finished(self) -> list[TransferResult]:
        self._drain()
        out, self._drained = self._drained, []
        return out

    def wait(self, job_ids: set[int]) -> None:
        while job_ids & self._pending:
            self._drain()
            if job_ids & self._pending:
                time.sleep(0.0005)

    def shutdown(self) -> None:
        # block 정책에서 슬롯을 못 받고 남은 staged job을 강제 완주 (QA 결함 2:
        # 미admit entry가 flush 조건에 안 걸려 영구 미완료되는 것 방지)
        if self._staged_jobs:
            deadline = time.monotonic() + 60.0
            while self._staged_jobs and time.monotonic() < deadline:
                self._advance_staged()
                if self._staged_jobs:
                    time.sleep(0.001)
        self.pool.shutdown(wait=True)
        self.transport.close()


# ---------------- spec ----------------


class ExperimentalFilesystemSpec(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        root_dir = self.extra_config.get("expfs_root_dir")
        if not root_dir:
            raise ValueError("expfs_root_dir must be set in kv_connector_extra_config")
        self.root_dir = root_dir
        self.transport_name = self.extra_config.get("expfs_transport", "cufile")
        self.cufile_mode = self.extra_config.get("expfs_cufile_mode", "unregistered")
        self.n_read_threads = int(self.extra_config.get("expfs_read_threads", 8))
        self.n_write_threads = int(self.extra_config.get("expfs_write_threads", 8))
        self.staging_slots = int(self.extra_config.get("expfs_staging_slots", 6))
        self.staging_writers = int(self.extra_config.get("expfs_staging_writers", 2))
        self.staging_policy = self.extra_config.get("expfs_staging_policy", "block")
        mob = self.extra_config.get("expfs_max_outstanding_write_bytes")
        self.max_ob = int(mob) if mob else None
        self.cpu_fallback_slots = int(
            self.extra_config.get("expfs_cpu_fallback_slots", 0))
        self._manager: FilesystemManager | None = None
        self._worker: FilesystemWorker | None = None

    def _make_mapper(self) -> FileMapper:
        return FileMapper.from_offloading_spec(
            root_dir=self.root_dir, offloading_spec=self,
            blocks_per_file=self.blocks_per_chunk, parallel_agnostic=True)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            mapper = self._make_mapper()
            cfg = mapper.get_config_file_path()
            os.makedirs(os.path.dirname(cfg), exist_ok=True)
            if not os.path.exists(cfg):
                import json
                with open(cfg, "w") as f:
                    json.dump(mapper.get_run_config(), f, indent=2, sort_keys=True)
            self._manager = FilesystemManager(mapper)
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = FilesystemWorker(
                kv_caches, self.blocks_per_chunk, self.transport_name, self.cufile_mode,
                self.n_read_threads, self.n_write_threads,
                self.staging_slots, self.staging_writers,
                self.staging_policy, self.max_ob, self.cpu_fallback_slots)
        return self._worker
