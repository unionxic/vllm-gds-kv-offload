# SPDX-License-Identifier: Apache-2.0
"""HybridWorker: GPU 쪽에서 CPUOffloadingWorker와 CuFileFsWorker를 함께 돌린다.

스케줄러가 준 job 하나가 두 티어에 걸쳐 있으므로, 티어별 하위 job으로 쪼개 제출하고
모두 끝났을 때만 바깥 job_id를 완료로 보고한다. 하나라도 실패하면 바깥 job도 실패다.
GPU 블록 목록은 HybridLoadStoreSpec의 cpu_pos/file_pos가 가리키는 청크 단위로 자른다.
"""
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec

logger = init_logger(__name__)

LAST_WORKER = None  # in-process 계측용

_HOST = 0
_SSD = 1


class HybridWorker(OffloadingWorker):
    def __init__(self, host_worker, ssd_worker, blocks_per_chunk: int):
        self.host = host_worker
        self.ssd = ssd_worker
        self.bpc = int(blocks_per_chunk)
        self._next_inner = 1
        self._pending: dict[int, set[int]] = {}  # 바깥 job -> 안 끝난 하위 job id
        self._owner: dict[int, tuple[int, int]] = {}  # 하위 job id -> (바깥 job, 티어)
        self._agg: dict[int, list] = {}  # 바깥 job -> [ok, bytes, time]
        self._ready: list[TransferResult] = []
        self.n_sub_host = 0
        self.n_sub_ssd = 0
        global LAST_WORKER
        LAST_WORKER = self
        logger.info("Hybrid worker: host+SSD, bpc=%d", self.bpc)

    # --- GPU 블록 쪼개기 ---

    def _chunks(self, gpu_spec: GPULoadStoreSpec, n_chunks: int) -> list[tuple]:
        """GPULoadStoreSpec의 블록 목록을 청크(=오프로드 키) 단위로 나눈다.

        각 항목은 (그 청크 첫 블록의 논리 인덱스, GPU 블록 id 목록).
        첫 청크만 앞이 잘릴 수 있고(block_indices[0] % bpc), 마지막 청크만 뒤가 짧을 수 있다.
        """
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        assert len(gpu_spec.group_sizes) == 1, "Hybrid: single KV cache group only"
        bids = [int(b) for b in gpu_spec.block_ids]
        base = int(gpu_spec.block_indices[0])
        skip = base % self.bpc
        chunk0 = base // self.bpc
        out: list[tuple] = []
        pos = 0
        for c in range(n_chunks):
            j0 = skip if c == 0 else 0
            take = min(self.bpc - j0, len(bids) - pos)
            out.append(((chunk0 + c) * self.bpc + j0, bids[pos : pos + take]))
            pos += take
        assert pos == len(bids), (
            f"Hybrid: {len(bids)} blocks do not fill {n_chunks} chunks (bpc={self.bpc})"
        )
        return out

    @staticmethod
    def _sub_gpu(chunks: list[tuple], positions: list[int]) -> GPULoadStoreSpec:
        blocks: list[int] = []
        for p in positions:
            blocks.extend(chunks[p][1])
        return GPULoadStoreSpec(
            blocks, group_sizes=[len(blocks)], block_indices=[chunks[positions[0]][0]]
        )

    # --- 하위 job 관리 ---

    def _alloc(self, outer: int, tier: int) -> int:
        inner = self._next_inner
        self._next_inner += 1
        self._owner[inner] = (outer, tier)
        self._pending.setdefault(outer, set()).add(inner)
        return inner

    def _submit(self, job_id: int, gpu_spec, hyb: HybridLoadStoreSpec, is_store: bool):
        assert isinstance(hyb, HybridLoadStoreSpec), f"Hybrid: got {type(hyb)}"
        self._agg[job_id] = [True, 0, 0.0]
        subs = []
        chunks = self._chunks(gpu_spec, hyb.n_keys) if hyb.n_keys else []
        if hyb.cpu is not None and hyb.cpu_pos:
            subs.append((_HOST, self.host, self._sub_gpu(chunks, hyb.cpu_pos), hyb.cpu))
        if hyb.file is not None and hyb.file_pos:
            subs.append((_SSD, self.ssd, self._sub_gpu(chunks, hyb.file_pos), hyb.file))
        if not subs:
            del self._agg[job_id]
            self._ready.append(TransferResult(job_id=job_id, success=True))
            return True
        # 먼저 모든 하위 id를 등록해 두어야, 첫 제출이 곧바로 끝나도 바깥 job이
        # 완료로 보고되지 않는다.
        inners = [(tier, w, gs, ts, self._alloc(job_id, tier)) for tier, w, gs, ts in subs]
        for tier, w, gs, ts, inner in inners:
            if tier == _HOST:
                self.n_sub_host += 1
            else:
                self.n_sub_ssd += 1
            ok = (
                w.submit_store(inner, gs, ts) if is_store else w.submit_load(inner, ts, gs)
            )
            if not ok:
                self._agg[job_id][0] = False
                self._owner.pop(inner, None)
                self._pending[job_id].discard(inner)
        if not self._pending[job_id]:
            del self._pending[job_id]
            ok = self._agg.pop(job_id)[0]
            self._ready.append(TransferResult(job_id=job_id, success=ok))
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(job_id, src_spec, dst_spec, is_store=True)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(job_id, dst_spec, src_spec, is_store=False)

    def get_finished(self) -> list[TransferResult]:
        out: list[TransferResult] = []
        for w in (self.host, self.ssd):
            for res in w.get_finished():
                owner = self._owner.pop(res.job_id, None)
                if owner is None:
                    continue
                outer = owner[0]
                agg = self._agg[outer]
                agg[0] = agg[0] and res.success
                if res.transfer_size:
                    agg[1] += res.transfer_size
                if res.transfer_time:
                    agg[2] = max(agg[2], res.transfer_time)
                left = self._pending[outer]
                left.discard(res.job_id)
                if not left:
                    del self._pending[outer]
                    ok, nbytes, secs = self._agg.pop(outer)
                    out.append(
                        TransferResult(
                            job_id=outer,
                            success=ok,
                            transfer_size=nbytes or None,
                            transfer_time=secs or None,
                        )
                    )
        if self._ready:
            out.extend(self._ready)
            self._ready.clear()
        return out

    def wait(self, job_ids: set[int]) -> None:
        per_tier: tuple[set[int], set[int]] = (set(), set())
        for outer in job_ids:
            for inner in self._pending.get(outer, ()):  # type: ignore[union-attr]
                owner = self._owner.get(inner)
                if owner is not None:
                    per_tier[owner[1]].add(inner)
        if per_tier[_HOST]:
            self.host.wait(per_tier[_HOST])
        if per_tier[_SSD]:
            self.ssd.wait(per_tier[_SSD])

    def stats(self) -> dict:
        ssd_stats = self.ssd.stats() if hasattr(self.ssd, "stats") else None
        host_stats = self.host.stats() if hasattr(self.host, "stats") else None
        return dict(
            sub_jobs_host=self.n_sub_host,
            sub_jobs_ssd=self.n_sub_ssd,
            inflight=len(self._pending),
            host=host_stats,
            ssd=ssd_stats,
        )

    def shutdown(self) -> None:
        self.host.shutdown()
        self.ssd.shutdown()
