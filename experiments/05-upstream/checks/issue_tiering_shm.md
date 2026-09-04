# [초안 — 제출 전 검토용] Upstream issue draft

Title: [Bug] TieringOffloadingSpec still leaks its /dev/shm region on crash —
barrier-unlink from #52596 is not wired into the tiering path

## Your current environment

vLLM main @ 1f1f628859 (0.26.1rc1.dev1488), Ubuntu 20.04, kernel 5.15,
single node, TP=1, in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0).

## Description

#52596 fixed the /dev/shm leak reported in #51579 by unlinking
`/dev/shm/vllm_offload_{engine_id}.mmap` once all workers have mmapped it
(unlink-after-barrier). This works for `CPUOffloadingSpec`, but
`TieringOffloadingSpec` still leaks the region when the process dies
without cleanup (SIGKILL, OOM-kill, node crash).

Verified on current main with the same experiment for both specs
(start engine → run one request → SIGKILL → inspect /dev/shm):

```
cpu:     during_run=[]                              leftover_after_SIGKILL=[]        -> CLEAN
tiering: during_run=['vllm_offload_<id>.mmap']      leftover_after_SIGKILL=['vllm_offload_<id>.mmap'] -> LEAK
```

Note the `during_run` column: with `CPUOffloadingSpec` the file is already
unlinked while the engine is running (barrier fired), while with
`TieringOffloadingSpec` it is still linked, i.e. the barrier-unlink never
happens on this path.

## Root cause

`vllm/v1/kv_offload/cpu/spec.py` passes `barrier=_all_workers_barrier` when
constructing `SharedOffloadRegion`, but `vllm/v1/kv_offload/tiering/spec.py`
constructs `SharedOffloadRegion` at two sites — the scheduler-side region
(`rank=None`) and the worker-side region — and neither passes `barrier`,
so `SharedOffloadRegion.__init__` takes the no-barrier path and never
unlinks.

Wiring the same barrier in is not enough by itself: the tiering spec's
scheduler-side opener runs in the scheduler process, which is not part of
the worker collective that `_all_workers_barrier` synchronizes. If a worker
unlinked the file after the worker barrier, a scheduler that opens by path
afterwards would fail. So the tiering path needs either (a) a rendezvous
that includes the scheduler-side opener, or (b) a liveness mechanism that
doesn't depend on unlink ordering, e.g. the flock-based reclaim proposed in
#54124, which we independently implemented and tested against this exact
scenario (SIGKILL orphan reclaim, same-engine-id restart, live-region
protection).

## Minimal repro

(첨부: shm_repro.py + run_shm_check.py — 제출 시 gist 또는 코드블록으로)
