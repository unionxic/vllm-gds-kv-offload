# SPDX-License-Identifier: Apache-2.0
"""shadow store 단위 테스트. GPU·cuFile 없이 CuFileFsManager의 배치 규칙만 확인한다.

shadow_store를 켜면 이미 티어에 있는 키도 같은 루트의 shadow/ 아래로 다시 쓰이고,
complete_store 뒤에는 그 파일이 지워지며 캐시 상태(_entries·total_bytes·루트별 사용량·lookup)는
켜기 전과 같아야 한다(저장 간섭 실험).
"""
import os

from vllm.v1.kv_offload.base import get_offload_block_hash, make_offload_key
from vllm.v1.kv_offload.cufile_fs.multi_root import MultiRootFileMapper
from vllm.v1.kv_offload.cufile_fs.spec import CuFileFsManager
from vllm.v1.kv_offload.file_mapper import FileMapper

CHUNK = 4096


def mapper(root_dir):
    return FileMapper(root_dir=root_dir, model_name="test/model", tokens_per_hash=16, blocks_per_file=1,
                      tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, dtype="float16")


def build(roots):
    return MultiRootFileMapper([mapper(d) for d, *_ in roots], roots)


def keys(n, seed=0):
    import random
    rng = random.Random(seed)
    return [make_offload_key(rng.randbytes(32), 0) for _ in range(n)]


def store_all(mgr, ks, ctx=None):
    """prepare_store → 파일 생성 → complete_store 를 한 번에. 쓴 경로 목록을 돌려준다."""
    out = mgr.prepare_store(ks, ctx)
    for path in out.store_spec.paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\0" * CHUNK)
    paths = list(out.store_spec.paths)
    mgr.complete_store(out.keys_to_store, ctx, success=True)
    return list(out.keys_to_store), paths


def snapshot(mgr):
    # _Entry는 참조가 아니라 값으로 떠 둬야 shadow 경로가 건드렸는지 보인다.
    entries = {k: (e.size, e.hits, e.last_ns) for k, e in mgr._entries.items()}
    return dict(entries=entries, total=mgr.total_bytes, root_bytes=list(mgr._root_bytes),
                root_files=list(mgr._root_files), root_of=dict(mgr._root_of), pending=set(mgr._pending))


def test_shadow_restores_all_keys_and_keeps_state(tmp_path):
    """켜기 전 저장한 키 전부가 다시 저장 대상이 되고, 캐시 상태는 그대로다."""
    m = mapper(str(tmp_path / "a"))
    mgr = CuFileFsManager(m)
    ks = keys(30)
    stored, real_paths = store_all(mgr, ks)
    assert len(stored) == 30
    before = snapshot(mgr)

    # (d) shadow 꺼진 상태에서는 이미 있는 키를 다시 저장하지 않는다.
    assert mgr.prepare_store(ks, None).keys_to_store == []

    # (a) shadow를 켜면 같은 키 전부가 대상이고 경로가 shadow/ 아래이며 실제 파일과 다르다.
    mgr.shadow_store = True
    out = mgr.prepare_store(ks, None)
    assert list(out.keys_to_store) == ks
    assert out.evicted_keys == []
    assert len(out.store_spec.paths) == 30
    for k, path in zip(out.keys_to_store, out.store_spec.paths):
        assert os.path.dirname(path).endswith("/shadow")
        assert path.startswith(str(tmp_path / "a") + "/")
        assert path != mgr._path(k)
        assert os.path.basename(path) == get_offload_block_hash(k).hex() + ".bin"
        assert os.path.isdir(os.path.dirname(path))  # makedirs(exist_ok)
    assert mgr.n_shadow_stores == 30
    assert len(mgr._shadow_pending) == 30
    assert mgr._pending == before["pending"], "shadow 키는 _pending 에 들어가지 않는다"

    # (6) 진행 중인 shadow 키를 다시 주면 건너뛴다.
    assert mgr.prepare_store(ks, None).keys_to_store == []

    # 워커가 쓴 것처럼 파일을 만들고 완료 보고.
    for path in out.store_spec.paths:
        with open(path, "wb") as f:
            f.write(b"\0" * CHUNK)
    mgr.complete_store(out.keys_to_store, None, success=True)

    # (b) shadow 파일은 지워지고 상태는 그대로다.
    for path in out.store_spec.paths:
        assert not os.path.exists(path)
    for path in real_paths:
        assert os.path.exists(path)
    after = snapshot(mgr)
    assert after == before
    assert mgr.n_shadow_done == 30 and mgr.shadow_bytes == 30 * CHUNK
    assert mgr._shadow_pending == set()

    # (c) lookup 은 전부 HIT 그대로다.
    for k in ks:
        assert mgr.lookup(k, None).name == "HIT"
    assert mgr.total_bytes == before["total"] and len(mgr._entries) == 30

    st = mgr.stats()["shadow"]
    assert st == dict(on=True, stores=30, done=30, gib=round(30 * CHUNK / 2**30, 3), pending=0)
    mgr.reset_cache()
    assert mgr._shadow_pending == set()


def test_shadow_on_unstored_keys(tmp_path):
    """켜기 전에 없던 키도 shadow 로 쓰이지만 캐시에는 남지 않는다."""
    m = mapper(str(tmp_path / "a"))
    mgr = CuFileFsManager(m)
    ks = keys(10, seed=7)
    mgr.shadow_store = True
    _, paths = store_all(mgr, ks)
    assert len(paths) == 10
    assert mgr._entries == {} and mgr.total_bytes == 0 and mgr.chunk_bytes == 0
    for k, path in zip(ks, paths):
        assert not os.path.exists(path)
        assert not os.path.exists(mgr._path(k))
        assert mgr.lookup(k, None).name == "MISS"


def test_shadow_uses_placed_root_in_spill(tmp_path):
    """다중 루트 spill 에서 shadow 경로의 루트가 _root_of 의 루트와 같고 용량 상태가 그대로다."""
    cap_a = 20 * CHUNK
    roots = [(str(tmp_path / "a"), 1, cap_a), (str(tmp_path / "b"), 3, 0), (str(tmp_path / "c"), 1, 0)]
    m = build(roots)
    mgr = CuFileFsManager(m)
    assert mgr._spill
    ks = keys(300, seed=3)
    store_all(mgr, ks[:1])  # chunk_bytes 확정
    store_all(mgr, ks[1:])
    assert len(mgr._entries) == 300
    before = snapshot(mgr)
    st_before = mgr._roots_stats()
    assert before["root_bytes"][0] <= cap_a

    mgr.shadow_store = True
    out = mgr.prepare_store(ks, None)
    assert list(out.keys_to_store) == ks
    moved = 0
    for k, path in zip(out.keys_to_store, out.store_spec.paths):
        i = mgr._root_of[k]
        # (e) shadow 경로의 루트 = 키가 놓인 루트
        assert path.startswith(roots[i][0] + "/")
        assert path.startswith(m.mappers[i].base_path)
        assert os.path.dirname(path).endswith("/shadow")
        assert os.path.dirname(path) != os.path.dirname(m.file_name(k, i))
        if i != m.hash_root(k):
            moved += 1
        with open(path, "wb") as f:
            f.write(b"\0" * CHUNK)
    assert moved, "용량이 있는 루트가 차서 넘어간 키가 있어야 한다"
    assert mgr.n_spill_refused == 0, "shadow 는 spill 배치를 쓰지 않는다"

    mgr.complete_store(out.keys_to_store, None, success=True)
    for path in out.store_spec.paths:
        assert not os.path.exists(path)
    # (b) 다중 루트 용량 상태까지 그대로
    assert snapshot(mgr) == before
    assert mgr._roots_stats() == st_before
    assert sum(mgr._root_bytes) == mgr.total_bytes
    # (c) lookup 은 전부 HIT, 실제 파일 경로도 그대로
    for k in ks:
        assert mgr.lookup(k, None).name == "HIT"
        assert mgr._path(k) == m.file_name(k, mgr._root_of[k])
        assert os.path.exists(mgr._path(k))

    # (d) 끄면 기존 동작으로 돌아온다.
    mgr.shadow_store = False
    assert mgr.prepare_store(ks, None).keys_to_store == []


def test_shadow_does_not_evict_under_capacity(tmp_path):
    """전체 용량 상한이 있어도 shadow 저장은 축출·거절을 일으키지 않는다."""
    m = mapper(str(tmp_path / "a"))
    mgr = CuFileFsManager(m, capacity_bytes=12 * CHUNK)
    ks = keys(10, seed=11)
    store_all(mgr, ks[:1])
    store_all(mgr, ks[1:])
    before = snapshot(mgr)
    n_ev, n_ref = mgr.n_evicted, mgr.n_refused

    mgr.shadow_store = True
    out = mgr.prepare_store(ks, None)
    assert len(out.keys_to_store) == 10 and out.evicted_keys == []
    for path in out.store_spec.paths:
        with open(path, "wb") as f:
            f.write(b"\0" * CHUNK)
    mgr.complete_store(out.keys_to_store, None, success=True)
    assert (mgr.n_evicted, mgr.n_refused) == (n_ev, n_ref)
    assert snapshot(mgr) == before
