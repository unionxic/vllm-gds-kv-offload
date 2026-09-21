# SPDX-License-Identifier: Apache-2.0
"""HybridManager 단위 테스트. GPU 없이 스케줄러 쪽 로직만 확인한다.

워커가 하는 일(실제 파일 쓰기)은 _put_ssd_file로 흉내 낸다. CuFileFsManager는
파일 존재를 적중 판정으로 쓰므로 이걸로 충분하다.
"""
import json
import os
import random

import pytest

from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    ReqContext,
    TransferResult,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cufile_fs.spec import CuFileFsManager, FileLoadStoreSpec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec
from vllm.v1.kv_offload.hybrid.manager import HybridManager
from vllm.v1.kv_offload.hybrid.worker import HybridWorker

CTX = ReqContext(req_id="t")
CHUNK = 4096


def key(i: int):
    return make_offload_key(bytes([i]) * 8, 0)


def build(tmp_path, host_blocks=8, write_through=True, placement="host_first",
          profile_path=None, host_share=None, roots=None):
    mapper = FileMapper(
        root_dir=str(tmp_path),
        model_name="test/model",
        tokens_per_hash=16,
        blocks_per_file=1,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype="float16",
    )
    return HybridManager(
        host=CPUOffloadingManager(num_blocks=host_blocks, cache_policy="lru"),
        ssd=CuFileFsManager(mapper),
        write_through=write_through,
        placement=placement,
        profile_path=profile_path,
        host_share=host_share,
    )


def _put_ssd_file(mgr, k):
    """워커가 SSD 파일을 다 쓴 상태를 흉내 낸다."""
    path = mgr.ssd.mapper.get_file_name(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * CHUNK)


def _put_host(mgr, keys):
    """host 티어에만 올라가 있고 읽을 수 있는 상태를 만든다."""
    out = mgr.host.prepare_store(keys, CTX)
    assert out is not None
    mgr.host.complete_store(out.keys_to_store, CTX, True)
    mgr._host_keys.update(out.keys_to_store)


def _finish_store(mgr, out):
    """prepare_store 결과대로 워커가 성공했다고 치고 마무리한다."""
    spec = out.store_spec
    for pos in spec.file_pos:
        _put_ssd_file(mgr, out.keys_to_store[pos])
    mgr.complete_store(set(out.keys_to_store), CTX, True)


def test_lookup_priority(tmp_path):
    m = build(tmp_path)
    k_host, k_ssd, k_none = key(1), key(2), key(3)
    _put_host(m, [k_host])
    _put_ssd_file(m, k_ssd)

    assert m.lookup(k_host, CTX) == LookupResult.HIT
    assert m.lookup(k_ssd, CTX) == LookupResult.HIT
    assert m.lookup(k_none, CTX) == LookupResult.MISS
    assert (m.n_hit_host, m.n_hit_ssd, m.n_miss) == (1, 1, 1)

    # host와 SSD 양쪽에 있으면 host가 이긴다.
    _put_ssd_file(m, k_host)
    assert m.lookup(k_host, CTX) == LookupResult.HIT
    assert (m.n_hit_host, m.n_hit_ssd) == (2, 1)


def test_prepare_load_split(tmp_path):
    m = build(tmp_path)
    k0, k1, k2 = key(10), key(11), key(12)
    _put_host(m, [k0, k2])
    _put_ssd_file(m, k1)

    spec = m.prepare_load([k0, k1, k2], CTX)
    assert isinstance(spec, HybridLoadStoreSpec)
    assert spec.n_keys == 3
    assert spec.cpu_pos == [0, 2]
    assert spec.file_pos == [1]
    assert len(spec.cpu.block_ids) == 2
    assert spec.file.paths == [m.ssd.mapper.get_file_name(k1)]

    # ref 보호가 걸렸다가 complete_load로 풀린다.
    assert m.host._policy.get(k0).ref_cnt == 1
    m.complete_load({k0, k1, k2}, CTX)
    assert m.host._policy.get(k0).ref_cnt == 0
    assert not m._load_host and not m._load_ssd
    assert not m.ssd._loading


def test_prepare_store_host_first_write_through(tmp_path):
    m = build(tmp_path)
    keys = [key(20), key(21), key(22)]
    out = m.prepare_store(keys, CTX)
    assert out is not None
    assert out.keys_to_store == keys  # 입력 순서 유지
    spec = out.store_spec
    assert spec.cpu_pos == [0, 1, 2]
    assert spec.file_pos == [0, 1, 2]  # write-through: 양쪽 모두
    assert len(spec.cpu.block_ids) == 3
    assert len(spec.file.paths) == 3
    assert out.evicted_keys == []

    _finish_store(m, out)
    for k in keys:
        assert m.lookup(k, CTX) == LookupResult.HIT
    assert m.n_store_host == 3 and m.n_store_ssd == 3


def test_placement_profile(tmp_path):
    keys = [key(30), key(31), key(32), key(33)]
    hot = [keys[0], keys[2]]
    prof = tmp_path / "prof.json"
    prof.write_text(
        json.dumps(
            {
                "hashes": {k[:-4].hex(): 3 for k in hot},
                "min_reuse": 2,
            }
        )
    )

    m = build(tmp_path, placement="profile", profile_path=str(prof))
    out = m.prepare_store(keys, CTX)
    assert out.keys_to_store == keys
    assert out.store_spec.cpu_pos == [0, 2]  # 재사용 많은 키만 host
    assert out.store_spec.file_pos == [0, 1, 2, 3]  # write-through: 전부 SSD

    m2 = build(tmp_path / "wb", placement="profile", profile_path=str(prof),
               write_through=False)
    out2 = m2.prepare_store(keys, CTX)
    assert out2.store_spec.cpu_pos == [0, 2]
    assert out2.store_spec.file_pos == [1, 3]  # 겹치지 않는 배치
    assert len(out2.store_spec.file.paths) == 2


def test_eviction_not_lost_with_write_through(tmp_path):
    m = build(tmp_path, host_blocks=2)
    first = [key(40), key(41)]
    _finish_store(m, m.prepare_store(first, CTX))
    assert m.host._num_evictable_cache_blocks == 2

    out = m.prepare_store([key(42)], CTX)
    assert out is not None
    assert m.n_evicted_host == 1
    # SSD에 사본이 있으므로 host 축출은 손실이 아니다.
    assert out.evicted_keys == []
    assert m.n_lost == 0
    _finish_store(m, out)
    for k in first + [key(42)]:
        assert m.lookup(k, CTX) == LookupResult.HIT


def test_eviction_reported_without_write_through(tmp_path):
    m = build(tmp_path, host_blocks=2, write_through=False)
    first = [key(50), key(51)]
    _finish_store(m, m.prepare_store(first, CTX))
    # write_through=False + host_first면 host에만 올라간다.
    assert m.n_store_ssd == 0

    out = m.prepare_store([key(52)], CTX)
    assert m.n_evicted_host == 1
    assert len(out.evicted_keys) == 1  # 사본이 없으니 진짜 손실
    assert m.n_lost == 1


def test_host_allocation_failure_falls_back_to_ssd(tmp_path):
    m = build(tmp_path, host_blocks=1)
    # 쓰기 중(ref_cnt -1)이라 축출할 수 없는 블록으로 host를 채운다.
    m.host.prepare_store([key(60)], CTX)
    out = m.prepare_store([key(61)], CTX)
    assert out is not None
    assert out.store_spec.cpu is None
    assert out.store_spec.file_pos == [0]
    assert m.n_host_alloc_fail == 1


def test_stats(tmp_path):
    m = build(tmp_path)
    _finish_store(m, m.prepare_store([key(70), key(71)], CTX))
    m.lookup(key(70), CTX)
    m.lookup(key(99), CTX)
    s = m.stats()
    for field in (
        "host_blocks_used",
        "host_blocks_total",
        "ssd_files",
        "ssd_gib",
        "hits_host",
        "hits_ssd",
        "misses",
        "stores_host",
        "stores_ssd",
        "demotions",
        "evictions_ssd",
    ):
        assert field in s, field
    assert s["host_blocks_used"] == 2
    assert s["host_blocks_total"] == 8
    assert s["ssd_files"] == 2
    assert s["hits_host"] == 1
    assert s["misses"] == 1
    assert s["demotions"] == 0


class _FakeWorker:
    """제출된 하위 job을 기록만 하는 가짜 워커(GPU 없이 분할 로직만 본다)."""

    def __init__(self):
        self.submitted = []
        self.finished = []
        self.waited = None

    def submit_store(self, job_id, gpu_spec, tier_spec):
        self.submitted.append(
            (job_id, [int(b) for b in gpu_spec.block_ids], list(gpu_spec.block_indices))
        )
        return True

    def submit_load(self, job_id, tier_spec, gpu_spec):
        return self.submit_store(job_id, gpu_spec, tier_spec)

    def get_finished(self):
        out = [TransferResult(job_id=j, success=ok) for j, ok in self.finished]
        self.finished = []
        return out

    def wait(self, job_ids):
        self.waited = set(job_ids)

    def shutdown(self):
        pass


def _hybrid_worker(bpc):
    host, ssd = _FakeWorker(), _FakeWorker()
    return HybridWorker(host, ssd, blocks_per_chunk=bpc), host, ssd


def test_worker_splits_gpu_blocks_by_tier():
    w, host, ssd = _hybrid_worker(2)
    gpu = GPULoadStoreSpec([10, 11, 20, 21, 30, 31], group_sizes=[6], block_indices=[0])
    spec = HybridLoadStoreSpec(
        n_keys=3,
        cpu=CPULoadStoreSpec([0, 1]),
        file=FileLoadStoreSpec(["/f1"]),
        cpu_pos=[0, 2],
        file_pos=[1],
    )
    w.submit_store(7, gpu, spec)
    assert host.submitted[0][1:] == ([10, 11, 30, 31], [0])
    assert ssd.submitted[0][1:] == ([20, 21], [2])
    # 하위 job id는 서로 다르고 바깥 id와 분리돼 있다.
    assert host.submitted[0][0] != ssd.submitted[0][0]


def test_worker_splits_with_unaligned_first_chunk():
    w, host, ssd = _hybrid_worker(2)
    gpu = GPULoadStoreSpec([11, 20, 21, 30, 31], group_sizes=[5], block_indices=[1])
    spec = HybridLoadStoreSpec(
        n_keys=3,
        cpu=CPULoadStoreSpec([0]),
        file=FileLoadStoreSpec(["/f1", "/f2"]),
        cpu_pos=[0],
        file_pos=[1, 2],
    )
    w.submit_load(9, spec, gpu)
    assert host.submitted[0][1:] == ([11], [1])  # 첫 청크만 앞이 잘린다
    assert ssd.submitted[0][1:] == ([20, 21, 30, 31], [2])


def test_worker_reports_outer_job_only_when_all_subjobs_finish():
    w, host, ssd = _hybrid_worker(2)
    gpu = GPULoadStoreSpec([10, 11, 20, 21], group_sizes=[4], block_indices=[0])
    spec = HybridLoadStoreSpec(
        n_keys=2,
        cpu=CPULoadStoreSpec([0]),
        file=FileLoadStoreSpec(["/f1"]),
        cpu_pos=[0],
        file_pos=[1],
    )
    w.submit_store(5, gpu, spec)
    h_id = host.submitted[0][0]
    s_id = ssd.submitted[0][0]

    w.wait({5})
    assert host.waited == {h_id} and ssd.waited == {s_id}

    host.finished = [(h_id, True)]
    assert w.get_finished() == []  # 아직 SSD가 안 끝났다
    ssd.finished = [(s_id, False)]
    res = w.get_finished()
    assert len(res) == 1 and res[0].job_id == 5
    assert res[0].success is False  # 하나라도 실패하면 실패


def test_worker_empty_spec_finishes_immediately():
    w, host, ssd = _hybrid_worker(2)
    gpu = GPULoadStoreSpec([], group_sizes=[0], block_indices=[0])
    w.submit_store(3, gpu, HybridLoadStoreSpec())
    res = w.get_finished()
    assert len(res) == 1 and res[0].job_id == 3 and res[0].success
    assert not host.submitted and not ssd.submitted


def test_placement_ratio_share(tmp_path):
    """몫 0.4면 20k 키 중 host 후보가 40 % ±3 %(상대). 같은 키는 늘 같은 티어."""
    m = build(tmp_path, placement="ratio", host_share=0.4)
    ks = [make_offload_key(bytes(random.Random(i).getrandbits(8) for _ in range(32)), 0)
          for i in range(20000)]
    want = [m._wants_host(k) for k in ks]
    share = sum(want) / len(ks)
    assert abs(share - 0.4) <= 0.03 * 0.4, share
    assert [m._wants_host(k) for k in ks] == want  # 결정적
    m2 = build(tmp_path, placement="ratio", host_share=0.4)
    assert [m2._wants_host(k) for k in ks] == want  # 인스턴스가 달라도 같음


def test_placement_ratio_extremes(tmp_path):
    ks = [key(i) for i in range(64)]
    assert all(build(tmp_path, placement="ratio", host_share=1.0)._wants_host(k) for k in ks)
    assert not any(build(tmp_path, placement="ratio", host_share=0.0)._wants_host(k) for k in ks)


def test_placement_ratio_requires_share(tmp_path):
    with pytest.raises(ValueError):
        build(tmp_path, placement="ratio")
    with pytest.raises(ValueError):
        build(tmp_path, placement="ratio", host_share=1.5)


def test_placement_ratio_splits_tiers_without_write_through(tmp_path):
    """write_through=false면 키마다 한 티어에만 간다. 두 티어 모두 몫을 받는다."""
    m = build(tmp_path, host_blocks=64, write_through=False, placement="ratio",
              host_share=0.5)
    ks = [key(i) for i in range(1, 60)]
    out = m.prepare_store(ks, CTX)
    assert out is not None
    assert set(out.store_spec.cpu_pos) & set(out.store_spec.file_pos) == set()
    assert out.store_spec.cpu_pos and out.store_spec.file_pos
    for pos in out.store_spec.cpu_pos:
        assert m._wants_host(out.keys_to_store[pos])
    for pos in out.store_spec.file_pos:
        assert not m._wants_host(out.keys_to_store[pos])


def test_placement_ratio_host_full_falls_back_to_ssd(tmp_path):
    """host 자리가 없어도 파일 티어로 떨어진다(write_through=false에서도)."""
    m = build(tmp_path, host_blocks=1, write_through=False, placement="ratio",
              host_share=1.0)
    out = m.prepare_store([key(i) for i in range(1, 40)], CTX)
    assert out is not None
    assert m.n_host_alloc_fail == 1
    assert len(out.store_spec.file_pos) == len(out.keys_to_store) > 0


def test_stats_reports_host_share(tmp_path):
    m = build(tmp_path, placement="ratio", host_share=0.47)
    st = m.stats()
    assert st["placement"] == "ratio" and st["host_share"] == 0.47
    assert build(tmp_path, placement="host_first").stats()["host_share"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
