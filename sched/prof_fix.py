# 5단계 원인 제거 검증 (FIX=... 일 때만 로드). 한 번에 한 요인만 제거한다.
#   blocking_event  torch.cuda.Event(blocking=True) → cudaEventBlockingSync,
#                   event 대기가 스핀 대신 sleep — 다른 동작은 submit_store 원본과 동일
#   step_throttle   엔진 step이 빈 결과(스케줄 0)를 낸 직후 0.5ms 양보 —
#                   빈 step 공회전의 GIL 점유만 완화
import os
import time
from functools import partial

import torch

MODE = os.environ.get("FIX", "")


def install():
    if "blocking_event" in MODE:
        import expfs

        def submit_store(self, job_id, src_spec, dst_spec):
            parts = self._partition(src_spec, dst_spec, is_store=True)
            ev = torch.cuda.Event(blocking=True)  # 원본과의 유일한 차이
            ev.record(torch.cuda.current_stream())
            tasks = [partial(self._store_task, ev, path,
                             self._chunk_spans(bids, j0))
                     for path, bids, j0 in parts]
            self._pending.add(job_id)
            self.pool.enqueue_store(job_id, len(tasks), tasks)
            return True
        expfs.FilesystemWorker.submit_store = submit_store
        print("FIX: blocking_event 적용", flush=True)

    if "step_throttle" in MODE:
        from vllm.v1.engine.core import EngineCore
        _st = EngineCore.step

        def step(self, *a, **k):
            out = _st(self, *a, **k)
            executed = isinstance(out, tuple) and len(out) == 2 and out[1]
            if not executed:
                time.sleep(0.0005)  # 빈 step: GIL을 놓고 잠깐 양보
            return out
        EngineCore.step = step
        print("FIX: step_throttle 적용", flush=True)
