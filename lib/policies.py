# read/write 스케줄링 정책 레이어 — lib/expfs.py를 수정하지 않고 몽키패치로 주입.
#
# 정책:
#   baseline        현행 동시 실행 (계측만 추가)
#   read_priority   load가 하나라도 대기/실행 중이면 store 제출을 보류,
#                   load가 모두 빠지면 보류분을 일괄 제출
#   deferred_store  store를 요청 사이 gap에서만 제출하고 그 gap에서 완료까지 대기
#                   (foreground와 store의 시간 중첩 제거)
#   phase_sep       task 수준 상호배제 — load task와 store task의 동시 실행 금지
#
# 공통 안전장치: 커넥터가 wait(job_ids)로 특정 store 완료를 요구하면(블록 재사용 fence)
# 보류 중인 해당 job을 강제 제출한다(forced_flush로 계수). GPU 블록 정합성 유지.
import threading

import expfs

state = {
    "policy": "baseline",
    "loads_pending": 0,
    "outstanding_store": 0, "max_outstanding_store": 0,
    "outstanding_load": 0, "max_outstanding_load": 0,
    "deferred_jobs": 0, "forced_flushes": 0, "gap_flushes": 0,
}
_lock = threading.Lock()
_deferred: list = []          # (worker, job_id, src, dst)
_kind: dict = {}              # job_id -> "load"|"store"

_orig = {}


class _PhaseGate:
    def __init__(self):
        self.cv = threading.Condition()
        self.readers = 0
        self.writers = 0

    def enter(self, kind):
        with self.cv:
            other = "writers" if kind == "readers" else "readers"
            while getattr(self, other) > 0:
                self.cv.wait()
            setattr(self, kind, getattr(self, kind) + 1)

    def leave(self, kind):
        with self.cv:
            setattr(self, kind, getattr(self, kind) - 1)
            self.cv.notify_all()


_gate = _PhaseGate()


def _submit_store_now(worker, job_id, src, dst):
    with _lock:
        _kind[job_id] = "store"
        state["outstanding_store"] += 1
        state["max_outstanding_store"] = max(
            state["max_outstanding_store"], state["outstanding_store"])
    return _orig["submit_store"](worker, job_id, src, dst)


def _flush_deferred_locked(pred=None):
    """_lock 보유 상태에서 조건에 맞는 보류 store를 꺼내 목록으로 반환."""
    take, keep = [], []
    for item in _deferred:
        if pred is None or pred(item[1]):
            take.append(item)
        else:
            keep.append(item)
    _deferred[:] = keep
    state["deferred_jobs"] = len(keep)
    return take


def install(policy: str):
    state["policy"] = policy
    W = expfs.FilesystemWorker
    for name in ("submit_store", "submit_load", "get_finished", "wait",
                 "_load_task", "_store_task"):
        _orig[name] = getattr(W, name)

    def submit_load(self, job_id, src, dst):
        with _lock:
            _kind[job_id] = "load"
            state["loads_pending"] += 1
            state["outstanding_load"] += 1
            state["max_outstanding_load"] = max(
                state["max_outstanding_load"], state["outstanding_load"])
        return _orig["submit_load"](self, job_id, src, dst)

    def submit_store(self, job_id, src, dst):
        if policy in ("read_priority", "deferred_store"):
            with _lock:
                hold = (state["loads_pending"] > 0) if policy == "read_priority" else True
                if hold:
                    _deferred.append((self, job_id, src, dst))
                    state["deferred_jobs"] = len(_deferred)
                    return True
        return _submit_store_now(self, job_id, src, dst)

    def get_finished(self):
        res = _orig["get_finished"](self)
        if res:
            release = False
            with _lock:
                for r in res:
                    k = _kind.pop(r.job_id, None)
                    if k == "load":
                        state["loads_pending"] -= 1
                        state["outstanding_load"] -= 1
                    elif k == "store":
                        state["outstanding_store"] -= 1
                if (policy == "read_priority" and state["loads_pending"] == 0
                        and _deferred):
                    release = True
            if release:
                with _lock:
                    take = _flush_deferred_locked()
                for w, j, s, d in take:
                    _submit_store_now(w, j, s, d)
        return res

    def wait(self, job_ids):
        with _lock:
            take = _flush_deferred_locked(pred=lambda j: j in job_ids)
            if take:
                state["forced_flushes"] += len(take)
        for w, j, s, d in take:
            _submit_store_now(w, j, s, d)
        return _orig["wait"](self, job_ids)

    W.submit_load, W.submit_store = submit_load, submit_store
    W.get_finished, W.wait = get_finished, wait

    if policy == "phase_sep":
        def load_task(self, path, spans):
            _gate.enter("readers")
            try:
                return _orig["_load_task"](self, path, spans)
            finally:
                _gate.leave("readers")

        def store_task(self, ev, path, spans):
            _gate.enter("writers")
            try:
                return _orig["_store_task"](self, ev, path, spans)
            finally:
                _gate.leave("writers")

        W._load_task, W._store_task = load_task, store_task


def on_request_gap(worker_holder: list, timeout: float = 120.0):
    """deferred_store: 요청 사이 gap에서 보류분을 제출하고 완료까지 대기."""
    if state["policy"] != "deferred_store":
        return
    with _lock:
        take = _flush_deferred_locked()
        state["gap_flushes"] += len(take)
    ids = set()
    for w, j, s, d in take:
        _submit_store_now(w, j, s, d)
        ids.add(j)
        if not worker_holder:
            worker_holder.append(w)
    if ids and worker_holder:
        worker_holder[0].wait(ids)


def summary():
    with _lock:
        return dict(state)
