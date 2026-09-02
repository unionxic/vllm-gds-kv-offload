# 원인 분석용 관찰 전용 계측 (PROF_INSTRUMENT=1일 때만 run_bench가 로드).
#   store 동작은 바꾸지 않는다 — 동일 호출을 NVTX range와 ns 누적으로 감쌀 뿐.
# STORE_JOB 하위: QUEUE_WAIT / CUDA_EVENT_WAIT / FILE_OPEN_REGISTER /
#                 CUFILE_WRITE(또는 TRANSPORT_WRITE) / COMMIT_RENAME(잔여로 추정) /
#                 COMPLETION_REPORT(get_finished 지연)
# ENGINE 하위: SCHEDULE / MODEL_EXECUTE / OUTPUT_PROCESS
import threading
import time

import torch

nvtx = torch.cuda.nvtx

T = {}   # 누적 ns
N = {}   # 횟수
_lock = threading.Lock()
_enq_ts = {}          # (kind, job_id) -> enqueue ns
_task_end = {}        # job_id -> 마지막 task 종료 ns


def _acc(key, ns):
    with _lock:
        T[key] = T.get(key, 0) + ns
        N[key] = N.get(key, 0) + 1


def _timed(key):
    class _R:
        def __enter__(self):
            nvtx.range_push(key)
            self.t0 = time.perf_counter_ns()
        def __exit__(self, *a):
            _acc(key, time.perf_counter_ns() - self.t0)
            nvtx.range_pop()
    return _R()


def install():
    import expfs
    import gdslib

    # --- pool enqueue 시각 기록 (QUEUE_WAIT의 시작점) ---
    P = expfs.DualQueueThreadPool
    for name in ("enqueue_store", "enqueue_load"):
        orig = getattr(P, name)
        def mk(orig, kind):
            def enq(self, job_id, n_tasks, tasks):
                now = time.perf_counter_ns()
                with _lock:
                    _enq_ts[(kind, job_id)] = now
                wrapped = []
                for t in tasks:
                    def w(t=t, kind=kind, job_id=job_id):
                        ts = None
                        with _lock:
                            ts = _enq_ts.get((kind, job_id))
                        if ts is not None:
                            _acc(f"{kind}.QUEUE_WAIT", time.perf_counter_ns() - ts)
                        nvtx.range_push(f"{kind}.QUEUE_WAIT_END")
                        nvtx.range_pop()
                        r = t()
                        with _lock:
                            _task_end[job_id] = time.perf_counter_ns()
                        return r
                    wrapped.append(w)
                return orig(self, job_id, n_tasks, wrapped)
            return enq
        setattr(P, name, mk(orig, "store" if "store" in name else "load"))

    # --- _store_task: 동일 본체를 단계별 NVTX로 재구성 (동작 불변) ---
    def _store_task(self, ev, path, spans):
        with torch.cuda.nvtx.range(f"STORE_JOB[{self.transport.name}]"):
            with _timed("store.CUDA_EVENT_WAIT"):
                ev.synchronize()
            with _timed("store.TRANSPORT_WRITE"):
                self.transport.write_chunk(path, spans, self.chunk_bytes)
    expfs.FilesystemWorker._store_task = _store_task

    def _load_task(self, path, spans):
        with torch.cuda.nvtx.range(f"LOAD_JOB[{self.transport.name}]"):
            with _timed("load.TRANSPORT_READ"):
                self.transport.read_chunk(path, spans, self.chunk_bytes)
    expfs.FilesystemWorker._load_task = _load_task

    # --- cuFile 프리미티브: FILE_OPEN_REGISTER / CUFILE_WRITE / CUFILE_READ ---
    G = gdslib.Gds
    _hr, _wr, _rd = G.handle_register, G.write, G.read
    def handle_register(self, fd):
        with _timed("cufile.FILE_OPEN_REGISTER"):
            return _hr(self, fd)
    def write(self, fh, ptr, size, off):
        with _timed("cufile.CUFILE_WRITE"):
            return _wr(self, fh, ptr, size, off)
    def read(self, fh, ptr, size, off):
        with _timed("cufile.CUFILE_READ"):
            return _rd(self, fh, ptr, size, off)
    G.handle_register, G.write, G.read = handle_register, write, read
    # COMMIT_RENAME ≈ store.TRANSPORT_WRITE − (FILE_OPEN_REGISTER+CUFILE_WRITE) 로 사후 추정

    # --- COMPLETION_REPORT: 마지막 task 종료 → get_finished 보고 지연 ---
    W = expfs.FilesystemWorker
    _gf = W.get_finished
    def get_finished(self):
        res = _gf(self)
        now = time.perf_counter_ns()
        for r in (res or []):
            with _lock:
                ts = _task_end.pop(r.job_id, None)
            if ts:
                _acc("store.COMPLETION_REPORT", now - ts)
        return res
    W.get_finished = get_finished

    # --- 엔진 3단계 ---
    from vllm.v1.core.sched.scheduler import Scheduler
    _sch = Scheduler.schedule
    def schedule(self, *a, **k):
        with _timed("engine.SCHEDULE"):
            return _sch(self, *a, **k)
    Scheduler.schedule = schedule
    _upd = Scheduler.update_from_output
    def upd(self, *a, **k):
        with _timed("engine.UPDATE_FROM_OUTPUT"):
            return _upd(self, *a, **k)
    Scheduler.update_from_output = upd
    from vllm.v1.engine.core import EngineCore
    _st = EngineCore.step
    def step(self, *a, **k):
        with _timed("engine.STEP"):
            return _st(self, *a, **k)
    EngineCore.step = step
    # MODEL_EXECUTE(GPU 실행 대기) ≈ STEP − SCHEDULE − UPDATE_FROM_OUTPUT (사후 추정);
    # 커널 제출 공백은 nsys CUDA trace에서 직접 본다.
    from vllm.v1.engine.output_processor import OutputProcessor
    _po = OutputProcessor.process_outputs
    def proc(self, *a, **k):
        with _timed("engine.OUTPUT_PROCESS"):
            return _po(self, *a, **k)
    OutputProcessor.process_outputs = proc


def summary():
    with _lock:
        out = {}
        for k in sorted(T):
            out[k] = dict(total_ms=round(T[k] / 1e6, 1), n=N[k],
                          mean_us=round(T[k] / N[k] / 1e3, 1))
        return out
