# SPDX-License-Identifier: Apache-2.0
"""MultiRootFileMapper 단위 테스트. GPU·cuFile 없이 매핑 규칙만 확인한다."""
import json
import os
import random

import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.cufile_fs.multi_root import (
    MultiRootFileMapper,
    parse_root_dirs,
)
from vllm.v1.kv_offload.file_mapper import FileMapper

N_KEYS = 20000
TOL = 0.05  # 분포 허용 오차(몫 대비 상대값)


def build(roots):
    """루트 목록으로 MultiRootFileMapper 하나를 만든다. 모든 필드는 루트만 다르다."""
    mappers = [
        FileMapper(
            root_dir=d,
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
        for d, _ in roots
    ]
    return MultiRootFileMapper(mappers, roots)


def keys(n, seed=0):
    rnd = random.Random(seed)
    return [make_offload_key(bytes(rnd.getrandbits(8) for _ in range(32)), 0) for _ in range(n)]


def test_parse_root_dirs():
    assert parse_root_dirs([{"dir": "/a", "weight": 13}, {"dir": "/b", "weight": 40}]) == [
        ("/a", 13, 0),
        ("/b", 40, 0),
    ]
    assert parse_root_dirs(["/a", "/b"]) == [("/a", 1, 0), ("/b", 1, 0)]
    with pytest.raises(ValueError):
        parse_root_dirs([])
    with pytest.raises(ValueError):
        parse_root_dirs([{"dir": "/a", "weight": 0}])
    with pytest.raises(ValueError):
        parse_root_dirs([{"weight": 1}])


def test_deterministic_across_instances():
    """같은 키는 인스턴스가 달라도 같은 루트·같은 경로."""
    roots = [("/r0", 13), ("/r1", 40)]
    m1, m2 = build(roots), build(roots)
    ks = keys(2000, seed=1)
    p1 = [m1.get_file_name(k) for k in ks]
    p2 = [m2.get_file_name(k) for k in ks]
    assert p1 == p2
    # 같은 키를 다시 물어도 같은 답(호출 순서에 의존하지 않음).
    assert [m1.get_file_name(k) for k in ks] == p1
    for k, p in zip(ks, p1):
        i = m1.root_index(k)
        assert p.startswith(roots[i][0] + "/")
        # 루트 아래 경로 모양은 단일 루트 FileMapper와 같다.
        assert p == m1.mappers[i].get_file_name(k)


def test_weighted_distribution():
    """가중치 1:3이면 20k 키가 25 % / 75 %에서 ±5 % 안."""
    roots = [("/r0", 1), ("/r1", 3)]
    m = build(roots)
    for k in keys(N_KEYS, seed=2):
        m.get_file_name(k)
    share = [n / N_KEYS for n in m.mapped]
    assert sum(m.mapped) == N_KEYS
    for got, want in zip(share, (0.25, 0.75)):
        assert abs(got - want) <= TOL * want, f"{share} vs (0.25, 0.75)"


def test_roots_stats():
    m = build([("/r0", 13), ("/r1", 40)])
    for k in keys(100, seed=3):
        m.get_file_name(k)
    st = m.roots_stats()
    assert [x["root"] for x in st] == ["/r0", "/r1"]
    assert [x["weight"] for x in st] == [13, 40]
    assert sum(x["mapped"] for x in st) == 100


def test_config_written_to_every_root(tmp_path):
    roots = [(str(tmp_path / f"r{i}"), w) for i, w in enumerate((13, 40))]
    m = build(roots)
    paths = m.get_config_file_paths()
    assert len(paths) == 2
    for cfg, (d, _) in zip(paths, roots):
        assert cfg.startswith(d + "/")
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        with open(cfg, "w") as f:
            json.dump(m.get_run_config(), f)
    for cfg in paths:
        assert json.load(open(cfg))["model_name"] == "test/model"
    # 루트 경로는 해시에 들어가지 않으므로 루트 아래 폴더 이름이 같다.
    assert len({os.path.basename(os.path.dirname(c)) for c in paths}) == 1


def test_single_root_matches_file_mapper():
    """루트가 하나면 단일 루트 FileMapper와 경로가 완전히 같다."""
    m = build([("/r0", 7)])
    fm = m.mappers[0]
    for k in keys(200, seed=4):
        assert m.get_file_name(k) == fm.get_file_name(k)
    assert FileMapper.get_config_file_paths(fm) == m.get_config_file_paths()
