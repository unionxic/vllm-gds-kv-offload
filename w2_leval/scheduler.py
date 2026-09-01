# W2 I/O 스케줄러 레이어 — expfs 무수정, 몽키패치 주입.
#
# 모드:
#   S0 current-concurrent-sync  현행 그대로 + 계측만
#   S1 read-priority admission  read 대기/실행 중이면 신규 write 제출 보류(기제출분은 완료)
#   S2 strict phase             read와 write 동시 outstanding 금지 + write quantum
#   S3 batch + blocking wait    S1 정책 + chunk의 span들을 cuFile Batch 제출,
#                               완료는 전용 completion thread의 blocking GetStatus
#                               (min_nr=1, finite timeout; busy poll 금지)
#
# 공통: outstanding read/write IO 수와 bytes 추적, bounded admission(bytes 상한),
#       시간 분해(ns): queue_wait / python_prepare / native_submit / native_wait /
#       python_completion / scheduler_report. NVTX: CUFILE_SUBMIT, CUFILE_WAIT,
#       READ_QUEUE_WAIT, WRITE_QUEUE_WAIT, STORE_DEFERRED, READ_BLOCKED_BY_WRITE.
# 안전장치: wait(job_ids) fence 시 보류 write 강제 제출. temp→rename은 chunk의
# 모든 span 성공 후에만. load 완료 보고는 모든 span 성공 후에만.
import os
import threading
import time
from collections import defaultdict

import torch

import expfs

nvtx = torch.cuda.nvtx

S = {
    "mode": "S0",
    "out_r_ios": 0, "out_r_bytes": 0, "out_w_ios": 0, "out_w_bytes": 0,
    "max_out_r_ios": 0, "max_out_r_bytes": 0, "max_out_w_ios": 0, "max_out_w_bytes": 0,
    "deferred_writes": 0, "forced_flushes": 0,
    "read_blocked_by_write_ns": 0, "read_blocked_events": 0,
    "batch_submits": 0, "batch_entries": 0, "batch_bytes": 0,
    "getstatus_calls": 0, "getstatus_timeouts": 0,
    "cufile_sync_calls": 0,
}
T = defaultdict(int)   # ns 합계
N = defaultdict(int)   # 호출 수
_lock = threading.Lock()
_orig = {}
_deferred = []            # (worker, job_id, src, dst, bytes)
_task_enqueue_ts = {}     # id(task closure) 대신 (job_id) -> enqueue ts
_job_last_task_end = {}
_kind = {}

_cfg = {"max_w_bytes": None, "max_r_bytes": None, "write_quantum_chunks": 2}


def _acct(kind, ios, nbytes):
    p = "r" if kind == "load" else "w"
    S[f"out_{p}_ios"] += ios
    S[f"out_{p}_bytes"] += nbytes
    S[f"max_out_{p}_ios"] = max(S[f"max_out_{p}_ios"], S[f"out_{p}_ios"])
    S[f"max_out_{p}_bytes"] = max(S[f"max_out_{p}_bytes"], S[f"out_{p}_bytes"])


class _Phase:
    """S2: readers 우선 상호배제 + write quantum."""

    def __init__(self):
        self.cv = threading.Condition()
        self.readers = 0
        self.writers = 0
        self.readers_waiting = 0
        self.write_quantum_left = _cfg["write_quantum_chunks"]

    def enter_read(self):
        with self.cv:
            self.readers_waiting += 1
            while self.writers > 0:
                self.cv.wait()
            self.readers_waiting -= 1
            self.readers += 1

    def leave_read(self):
        with self.cv:
            self.readers -= 1
            if self.readers == 0:
                self.write_quantum_left = _cfg["write_quantum_chunks"]
                self.cv.notify_all()

    def enter_write(self):
        with self.cv:
            while (self.readers > 0 or self.readers_waiting > 0
                   or self.write_quantum_left <= 0):
                if self.readers == 0 and self.readers_waiting == 0:
                    self.write_quantum_left = _cfg["write_quantum_chunks"]
                    continue
                self.cv.wait()
            self.write_quantum_left -= 1
            self.writers += 1

    def leave_write(self):
        with self.cv:
            self.writers -= 1
            self.cv.notify_all()


_phase = _Phase()

# ---------- batch: 스레드-로컬 핸들 ----------
# 단일 공유 핸들에 다중 스레드 Submit + 별도 GetStatus는 libcufile 1.13에서
# 이벤트 유실을 일으켰다(standalone 파일럿은 정상 → 핸들 동시성 문제).
# 따라서 IO 스레드마다 자기 핸들을 갖고, 같은 스레드에서 Submit 후
# blocking GetStatus로 drain한다(min_nr≥1, finite timeout — busy poll 아님).
_batch_tls = threading.local()


def _batch_chunk_io(worker, path, spans, opcode):
    """한 chunk의 span들을 스레드-로컬 핸들로 batch 제출, 같은 스레드에서 drain."""
    from cufile_batch import CUFILE_COMPLETE, CuFileBatch

    g = worker.transport.g
    b = getattr(_batch_tls, "b", None)
    if b is None:
        b = _batch_tls.b = CuFileBatch(g.lib, _cfg.get("batch_capacity", 128))

    t0 = time.perf_counter_ns()
    if opcode == 1:  # write → temp에 쓰고 전체 성공 후 rename
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o644)
    else:
        tmp = None
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    fh = g.handle_register(fd)
    entries = [(fh.value, worker.flat[t].data_ptr() + boff, size, foff, opcode, i + 1)
               for i, (t, boff, size, foff) in enumerate(spans)]
    prep_ns = time.perf_counter_ns() - t0

    results = []
    try:
        nvtx.range_push("CUFILE_SUBMIT")
        t1 = time.perf_counter_ns()
        arr = b.submit(entries)  # noqa: F841  (완료까지 참조 유지)
        submit_ns = time.perf_counter_ns() - t1
        nvtx.range_pop()
        with _lock:
            S["batch_submits"] += 1
            S["batch_entries"] += len(entries)
            S["batch_bytes"] += sum(e[2] for e in entries)

        nvtx.range_push("CUFILE_WAIT")
        t2 = time.perf_counter_ns()
        deadline = time.monotonic() + 120
        while len(results) < len(entries):
            evs, timed_out = b.get_status(1, len(entries), 500)  # native blocking
            with _lock:
                S["getstatus_calls"] += 1
                if timed_out:
                    S["getstatus_timeouts"] += 1
            results.extend(evs)
            if time.monotonic() > deadline:
                raise RuntimeError(f"batch completion timeout @{path}")
        wait_ns = time.perf_counter_ns() - t2
        nvtx.range_pop()

        t3 = time.perf_counter_ns()
        for _, status, ret in results:
            if status != CUFILE_COMPLETE:
                raise RuntimeError(f"batch entry status=0x{status:x} @{path}")
        got = sum(max(0, r) for _, _, r in results)
        want = sum(s[2] for s in spans)
        if got != want:
            raise RuntimeError(f"batch short IO {got}/{want} @{path}")
    except Exception:
        g.handle_deregister(fh)
        os.close(fd)
        if tmp:
            try:
                os.remove(tmp)  # 실패 시 rename 금지 + temp 정리
            except OSError:
                pass
        raise
    g.handle_deregister(fh)
    os.close(fd)
    if tmp:
        os.replace(tmp, path)  # 전체 span 성공 후에만 rename
    fin_ns = time.perf_counter_ns() - t3
    with _lock:
        T["python_prepare_ns"] += prep_ns
        T["native_submit_ns"] += submit_ns
        T["native_wait_ns"] += wait_ns
        T["python_completion_ns"] += fin_ns
        N["batch_chunks"] += 1
    return sum(s[2] for s in spans)


def install(mode, max_w_bytes=None, max_r_bytes=None, write_quantum_chunks=2,
            batch_capacity=128):
    S["mode"] = mode
    _cfg.update(max_w_bytes=max_w_bytes, max_r_bytes=max_r_bytes,
                write_quantum_chunks=write_quantum_chunks)
    W = expfs.FilesystemWorker
    for n in ("submit_store", "submit_load", "get_finished", "wait",
              "_load_task", "_store_task"):
        _orig[n] = getattr(W, n)

    def _chunk_bytes(spans_list):
        return sum(s[2] for chunk in spans_list for s in chunk)

    def submit_load(self, job_id, src, dst):
        with _lock:
            _kind[job_id] = "load"
            nb = 0  # span 계산은 태스크 생성 내부라 job 단위 추정: chunk_bytes×paths
            nb = len(src.paths) * self.chunk_bytes
            _acct("load", len(src.paths), nb)
            _task_enqueue_ts[job_id] = time.perf_counter_ns()
            if S["out_w_ios"] > 0:
                S["read_blocked_events"] += 1
        return _orig["submit_load"](self, job_id, src, dst)

    def submit_store(self, job_id, src, dst):
        nb = len(dst.paths) * self.chunk_bytes
        hold = False
        with _lock:
            if mode in ("S1", "S3") and S["out_r_ios"] > 0:
                hold = True
            if mode in ("DEF", "DEFB"):
                hold = True  # foreground-aware admission: gap에서만 실행
            if _cfg["max_w_bytes"] and S["out_w_bytes"] + nb > _cfg["max_w_bytes"]:
                hold = True
            if hold:
                _deferred.append((self, job_id, src, dst, nb))
                S["deferred_writes"] += 1
                return True
            _kind[job_id] = "store"
            _acct("store", len(dst.paths), nb)
            _task_enqueue_ts[job_id] = time.perf_counter_ns()
        return _orig["submit_store"](self, job_id, src, dst)

    def _release_deferred(self):
        if mode in ("DEF", "DEFB"):
            return  # gap 또는 fence에서만 방출
        take = []
        with _lock:
            while _deferred:
                w, j, s, d, nb = _deferred[0]
                if S["out_r_ios"] > 0 and mode in ("S1", "S3"):
                    break
                if _cfg["max_w_bytes"] and S["out_w_bytes"] + nb > _cfg["max_w_bytes"]:
                    break
                _deferred.pop(0)
                _kind[j] = "store"
                _acct("store", len(d.paths), nb)
                _task_enqueue_ts[j] = time.perf_counter_ns()
                take.append((w, j, s, d))
        for w, j, s, d in take:
            nvtx.range_push("STORE_DEFERRED_RELEASE")
            _orig["submit_store"](w, j, s, d)
            nvtx.range_pop()

    def get_finished(self):
        res = _orig["get_finished"](self)
        now = time.perf_counter_ns()
        if res:
            with _lock:
                for r in res:
                    k = _kind.pop(r.job_id, None)
                    end = _job_last_task_end.pop(r.job_id, None)
                    if end:
                        T["scheduler_report_ns"] += now - end
                        N["scheduler_report"] += 1
                    if k == "load":
                        S["out_r_ios"] = max(0, S["out_r_ios"] - 1)
                    elif k == "store":
                        S["out_w_ios"] = max(0, S["out_w_ios"] - 1)
            _release_deferred(self)
        return res

    def wait(self, job_ids):
        take = []
        with _lock:
            keep = []
            for item in _deferred:
                if item[1] in job_ids:
                    take.append(item)
                    S["forced_flushes"] += 1
                else:
                    keep.append(item)
            _deferred[:] = keep
            for w, j, s, d, nb in take:
                _kind[j] = "store"
                _acct("store", len(d.paths), nb)
        for w, j, s, d, nb in take:
            _orig["submit_store"](w, j, s, d)
        return _orig["wait"](self, job_ids)

    def _timed(kind, body, self, path, spans, is_batch=False):
        tq0 = time.perf_counter_ns()
        # queue_wait ≈ (task 시작) − (job enqueue). job 단위 근사.
        nvtx.range_push("READ_QUEUE_WAIT" if kind == "load" else "WRITE_QUEUE_WAIT")
        nvtx.range_pop()
        nb = sum(s[2] for s in spans)
        t0 = time.perf_counter_ns()
        if mode == "S3" and self.transport.name == "cufile":
            _batch_chunk_io(self, path, spans, 0 if kind == "load" else 1)
        elif is_batch:
            body()  # batch 내부에서 자체 계측
        else:
            nvtx.range_push("CUFILE_SUBMIT")
            body()
            nvtx.range_pop()
            with _lock:
                S["cufile_sync_calls"] += len(spans)
                T["native_wait_ns"] += time.perf_counter_ns() - t0
                N["native_sync"] += len(spans)
        end = time.perf_counter_ns()
        with _lock:
            p = "r" if kind == "load" else "w"
            S[f"out_{p}_bytes"] = max(0, S[f"out_{p}_bytes"] - nb)
            if kind == "load" and S["out_w_ios"] > 0:
                S["read_blocked_by_write_ns"] += end - t0
        return end

    def load_task(self, path, spans):
        if mode == "S2":
            _phase.enter_read()
        try:
            end = _timed("load", lambda: _orig["_load_task"](self, path, spans),
                         self, path, spans)
        finally:
            if mode == "S2":
                _phase.leave_read()
        # job 완료시각 근사 기록 (마지막 task가 덮어씀)
        for j, k in list(_kind.items()):
            if k == "load":
                _job_last_task_end[j] = end
        _release_deferred(self)

    def store_task(self, ev, path, spans):
        if mode == "S2":
            _phase.enter_write()
        try:
            if mode == "DEFB" and self.transport.name == "cufile":
                def body():
                    ev.synchronize()  # compute fence 유지
                    _batch_chunk_io(self, path, spans, 1)
                end = _timed("store", body, self, path, spans, is_batch=True)
            else:
                end = _timed(
                    "store", lambda: _orig["_store_task"](self, ev, path, spans),
                    self, path, spans)
        finally:
            if mode == "S2":
                _phase.leave_write()
        for j, k in list(_kind.items()):
            if k == "store":
                _job_last_task_end[j] = end

    W.submit_load, W.submit_store = submit_load, submit_store
    W.get_finished, W.wait = get_finished, wait
    W._load_task, W._store_task = load_task, store_task

    # queue_wait 계측: 스레드풀 enqueue를 감싸 task별 대기시간을 기록
    from vllm.v1.kv_offload.tiering.fs import thread_pool as tp_mod
    if not getattr(tp_mod.DualQueueThreadPool, "_w2_wrapped", False):
        for meth, kind in (("enqueue_load", "read"), ("enqueue_store", "write")):
            orig_enq = getattr(tp_mod.DualQueueThreadPool, meth)

            def make(orig_enq, kind):
                def enq(self, job_id, n_tasks, tasks):
                    ts = time.perf_counter_ns()

                    def wrap(fn):
                        def run():
                            waited = time.perf_counter_ns() - ts
                            with _lock:
                                T[f"queue_wait_{kind}_ns"] += waited
                                N[f"queue_wait_{kind}"] += 1
                            nvtx.range_push(
                                "READ_QUEUE_WAIT" if kind == "read"
                                else "WRITE_QUEUE_WAIT")
                            nvtx.range_pop()
                            return fn()
                        return run
                    return orig_enq(self, job_id, n_tasks,
                                    [wrap(f) for f in tasks])
                return enq
            setattr(tp_mod.DualQueueThreadPool, meth, make(orig_enq, kind))
        tp_mod.DualQueueThreadPool._w2_wrapped = True

    _cfg["batch_capacity"] = batch_capacity


def ensure_batch(worker):
    return  # 스레드-로컬 핸들로 대체됨


def release_gap(worker_holder: list):
    """foreground gap: 보류 store 전부 방출 후 완료까지 drain.
    admission gate의 '허용된 store 구간' 실행 지점이며 마지막 drain에도 사용."""
    take = []
    with _lock:
        while _deferred:
            w, j, s, d, nb = _deferred.pop(0)
            _kind[j] = "store"
            _acct("store", len(d.paths), nb)
            _task_enqueue_ts[j] = time.perf_counter_ns()
            take.append((w, j, s, d))
        S["deferred_writes_released_gap"] = S.get("deferred_writes_released_gap", 0) + len(take)
    if not take:
        return 0
    nvtx.range_push("STORE_DEFERRED")
    ids = set()
    for w, j, s, d in take:
        _orig["submit_store"](w, j, s, d)
        ids.add(j)
        if not worker_holder:
            worker_holder.append(w)
    if worker_holder:
        worker_holder[0].wait(ids)
    nvtx.range_pop()
    return len(take)


def summary():
    with _lock:
        out = dict(S)
        out["timing_ns"] = dict(T)
        out["timing_n"] = dict(N)
        out["deferred_pending"] = len(_deferred)
        return out
