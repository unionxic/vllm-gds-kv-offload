# SPDX-License-Identifier: Apache-2.0
"""루트 용량 spill 단위 테스트. GPU·cuFile 없이 CuFileFsManager의 배치 규칙만 확인한다.

루트 A(용량 있음)가 차면 그 뒤 키는 B·C 사이에서 가중치대로 갈리고, 이미 A에 놓인 키는
lookup·prepare_load에서 A 경로로 나와야 한다.
"""
import os

import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.cufile_fs.multi_root import MultiRootFileMapper, parse_root_dirs
from vllm.v1.kv_offload.cufile_fs.spec import CuFileFsManager
from vllm.v1.kv_offload.file_mapper import FileMapper

CHUNK = 4096


def build(roots):
    mappers = [
        FileMapper(root_dir=d, model_name="test/model", tokens_per_hash=16, blocks_per_file=1,
                   tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, dtype="float16")
        for d, *_ in roots
    ]
    return MultiRootFileMapper(mappers, roots)


def keys(n, seed=0):
    import random
    rng = random.Random(seed)
    return [make_offload_key(rng.randbytes(32), 0) for _ in range(n)]


def store_all(mgr, ks, ctx=None):
    """prepare_store → 파일 생성 → complete_store 를 한 번에. 저장된 키 목록을 돌려준다."""
    out = mgr.prepare_store(ks, ctx)
    for k, path in zip(out.keys_to_store, out.store_spec.paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\0" * CHUNK)
    mgr.complete_store(out.keys_to_store, ctx, success=True)
    return list(out.keys_to_store)


def test_parse_capacity():
    r = parse_root_dirs([{"dir": "/a", "weight": 2, "capacity_gb": 1.5}, "/b"])
    assert r == [("/a", 2, int(1.5 * 2**30)), ("/b", 1, 0)]
    with pytest.raises(ValueError):
        parse_root_dirs([{"dir": "/a", "capacity_gb": -1}])


def test_no_capacity_keeps_hash_placement(tmp_path):
    roots = [(str(tmp_path / "a"), 1, 0), (str(tmp_path / "b"), 3, 0)]
    m = build(roots)
    assert not m.has_capacity
    mgr = CuFileFsManager(m)
    assert not mgr._spill
    ks = keys(200)
    stored = store_all(mgr, ks)
    assert len(stored) == 200
    for k in ks:
        assert mgr._path(k) == m.file_name(k, m.hash_root(k))


def test_spill_when_root_full(tmp_path):
    cap_a = 20 * CHUNK  # A에는 청크 20개만 들어간다
    roots = [(str(tmp_path / "a"), 1, cap_a), (str(tmp_path / "b"), 3, 0), (str(tmp_path / "c"), 1, 0)]
    m = build(roots)
    mgr = CuFileFsManager(m)
    assert mgr._spill
    ks = keys(2000)
    # chunk_bytes는 첫 완료 파일로 정해지므로 첫 배치는 하나만 저장해 크기를 알린다.
    first = store_all(mgr, ks[:1])
    assert mgr.chunk_bytes == CHUNK
    rest = store_all(mgr, ks[1:])
    assert len(first) + len(rest) == 2000, "용량이 없는 루트가 있으니 거르는 키가 없어야 한다"
    st = mgr._roots_stats()
    assert st[0]["files"] <= 20 and st[0]["bytes_gib"] <= cap_a / 2**30
    assert st[0]["files"] == 20, "A는 용량까지 찬다"
    n_b, n_c = st[1]["files"], st[2]["files"]
    # A가 찬 뒤의 키는 B:C = 3:1 근처로 갈린다(허용 오차 15 %).
    assert abs(n_b / (n_b + n_c) - 0.75) < 0.15
    assert st[1]["spilled"] + st[2]["spilled"] > 0
    # 놓인 루트 경로가 lookup·prepare_load 경로와 같고 실제로 파일이 있다.
    for k in ks:
        i = mgr._root_of[k]
        p = m.file_name(k, i)
        assert os.path.exists(p)
        assert mgr._path(k) == p
        assert mgr.lookup(k, None).name == "HIT"
    assert mgr.prepare_load(ks[:5], None).paths == [mgr._path(k) for k in ks[:5]]
    # A에 놓인 키는 해시 루트가 A인 키뿐이고, A가 찬 뒤 해시 루트 A였던 키는 다른 루트에 있다.
    moved = [k for k in ks if m.hash_root(k) == 0 and mgr._root_of[k] != 0]
    assert moved, "A로 갔어야 할 키 일부가 다른 루트로 넘어가야 한다"
    assert mgr.n_spill_refused == 0


def test_all_full_refuses(tmp_path):
    roots = [(str(tmp_path / "a"), 1, 3 * CHUNK), (str(tmp_path / "b"), 1, 3 * CHUNK)]
    m = build(roots)
    mgr = CuFileFsManager(m)
    ks = keys(50)
    store_all(mgr, ks[:1])
    stored = store_all(mgr, ks[1:])
    assert len(stored) == 5
    assert mgr.n_spill_refused == 44


def test_evict_frees_root_room(tmp_path):
    """전체 용량 상한(capacity_bytes)으로 축출되면 루트별 사용량도 같이 줄어야 한다."""
    roots = [(str(tmp_path / "a"), 1, 4 * CHUNK), (str(tmp_path / "b"), 1, 0)]
    m = build(roots)
    mgr = CuFileFsManager(m, capacity_bytes=10 * CHUNK)
    ks = keys(60)
    store_all(mgr, ks[:1])
    store_all(mgr, ks[1:])
    st = mgr._roots_stats()
    assert st[0]["files"] + st[1]["files"] == len(mgr._entries) <= 10
    assert st[0]["files"] <= 4
    assert sum(mgr._root_bytes) == mgr.total_bytes
