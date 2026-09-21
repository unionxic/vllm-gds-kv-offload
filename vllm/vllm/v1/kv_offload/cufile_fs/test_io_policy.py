# SPDX-License-Identifier: Apache-2.0
"""경로별 동시 발행 조절(cufile_fs_root_inflight)과 읽기 우선(cufile_fs_read_priority) 통합 테스트.

네이티브 확장(csrc/kv_offload/cufile_fs.cpp)을 실제로 빌드해 GPU 텐서 하나를 루트 두 개에
청크 수십 개로 저장·적재한다. GPU나 cuFile이 없으면 통째로 skip한다.
확인하는 것: (a) 정책이 꺼지면 예전과 같은 왕복 일치, (b) 루트 상한이 in-flight 최댓값을 묶는지,
(c) 읽기가 남아 있는 동안 쓰기 동시 발행이 K 이하인지.
"""
import os
import tempfile

import pytest
import torch

PAGE = 1 << 20  # 청크 하나(= 텐서 1개 × bpc 1페이지). 4096 정렬이면 된다.
N_SRC = 48  # 저장할 청크 수(루트 두 개에 나눠 놓는다)


def _native():
    from vllm.v1.kv_offload.cufile_fs.native import load

    return load()


@pytest.fixture(scope="module")
def mod():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    try:
        return _native()
    except Exception as e:  # 빌드 실패·libcufile 없음
        pytest.skip(f"cufile_fs native extension unavailable: {e}")


@pytest.fixture(scope="module")
def buf():
    """앞쪽 N_SRC 페이지는 저장 원본, 뒤쪽 N_SRC 페이지는 적재 목적지."""
    t = torch.empty(2 * N_SRC * PAGE, dtype=torch.uint8, device="cuda")
    t[: N_SRC * PAGE] = torch.randint(0, 256, (N_SRC * PAGE,), dtype=torch.uint8, device="cuda")
    t[N_SRC * PAGE :] = 0
    torch.cuda.synchronize()
    return t


def make_fs(mod, buf, *, n_read=4, n_write=4, root_inflight=None, read_priority=0):
    from vllm.v1.kv_offload.cufile_fs.spec import parse_root_inflight

    roots, limits = parse_root_inflight(root_inflight)
    return mod.CuFileFs([buf.data_ptr()], [buf.numel()], [PAGE], 1, n_read, n_write, False,
                        roots, limits, read_priority)


def run_job(fs, job_id, is_store, paths, bids):
    fs.submit(job_id, is_store, paths, [[b] for b in bids], [0] * len(paths), 0)
    fs.wait([job_id])
    fin = dict(fs.get_finished())
    assert fin.get(job_id) is True, f"job {job_id} failed: {fs.stats()['last_error']}"


def paths_for(root_a, root_b):
    """청크 절반씩 루트 A/B에. 인덱스 i의 원본 블록은 i, 적재 목적지 블록은 N_SRC + i."""
    return [os.path.join(root_a if i % 2 == 0 else root_b, f"c{i:04d}.kv") for i in range(N_SRC)]


@pytest.fixture(scope="module")
def roots():
    with tempfile.TemporaryDirectory(prefix="cufile_fs_policy_") as d:
        a, b = os.path.join(d, "rootA"), os.path.join(d, "rootB")
        os.makedirs(a)
        os.makedirs(b)
        yield a, b


def _store_all(fs, paths):
    run_job(fs, 1, True, paths, list(range(N_SRC)))


def _load_all(fs, paths, job_id=2):
    run_job(fs, job_id, False, paths, [N_SRC + i for i in range(N_SRC)])


def test_policy_off_roundtrip(mod, buf, roots):
    """정책이 꺼지면 예전 동작 그대로: 파일 내용이 왕복으로 일치한다."""
    paths = paths_for(*roots)
    fs = make_fs(mod, buf)
    try:
        _store_all(fs, paths)
        buf[N_SRC * PAGE :] = 0
        torch.cuda.synchronize()
        _load_all(fs, paths)
        st = fs.stats()
        assert st["gate_enabled"] is False
        assert st["root_inflight"] == []
        assert st["errors"] == 0
    finally:
        fs.shutdown()
    torch.cuda.synchronize()
    assert torch.equal(buf[: N_SRC * PAGE], buf[N_SRC * PAGE :])


def test_root_inflight_limit(mod, buf, roots):
    """루트 A만 상한 1을 주면 A의 in-flight 최댓값이 1이고, B는 통계에 없다(무제한)."""
    root_a, root_b = roots
    paths = paths_for(root_a, root_b)
    fs = make_fs(mod, buf, root_inflight={root_a: 1})
    try:
        _store_all(fs, paths)
        buf[N_SRC * PAGE :] = 0
        torch.cuda.synchronize()
        _load_all(fs, paths)
        st = fs.stats()
        assert st["gate_enabled"] is True
        assert st["errors"] == 0
        assert [r["root"] for r in st["root_inflight"]] == [root_a]
        ra = st["root_inflight"][0]
        assert ra["limit"] == 1
        assert ra["peak_inflight"] == 1
        assert ra["chunks"] == N_SRC  # 저장 24 + 적재 24 = A에 놓인 청크 전부
        assert ra["gate_wait_ns"] >= 0
    finally:
        fs.shutdown()
    torch.cuda.synchronize()
    assert torch.equal(buf[: N_SRC * PAGE], buf[N_SRC * PAGE :])


def test_root_inflight_json_string(mod, buf, roots):
    """JSON dict 문자열도 같은 매핑으로 읽힌다."""
    root_a, _ = roots
    fs = make_fs(mod, buf, root_inflight=f'{{"{root_a}": 2}}')
    try:
        st = fs.stats()
        assert [(r["root"], r["limit"]) for r in st["root_inflight"]] == [(root_a, 2)]
    finally:
        fs.shutdown()


def test_read_priority_caps_writes(mod, buf, roots):
    """읽기가 남아 있는 동안 쓰기 동시 발행이 K(=1) 이하로 묶인다."""
    root_a, root_b = roots
    src = paths_for(root_a, root_b)
    dst = [p + ".w" for p in src]  # 같은 루트에 새로 쓰는 경로(읽기 대상과 겹치지 않게)
    fs0 = make_fs(mod, buf)
    try:
        _store_all(fs0, src)
    finally:
        fs0.shutdown()

    fs = make_fs(mod, buf, read_priority=1)
    try:
        # 읽기를 먼저 넣어 r_out_을 올린 뒤 쓰기를 넣는다. 쓰기는 읽기가 끝날 때까지 1개씩만 나간다.
        fs.submit(10, False, src, [[N_SRC + i] for i in range(N_SRC)], [0] * N_SRC, 0)
        fs.submit(11, True, dst, [[i] for i in range(N_SRC)], [0] * N_SRC, 0)
        fs.wait([10, 11])
        fin = dict(fs.get_finished())
        assert fin.get(10) is True and fin.get(11) is True, fs.stats()["last_error"]
        st = fs.stats()
        assert st["read_priority"] == 1
        assert st["gate_enabled"] is True
        assert st["errors"] == 0
        assert st["write_inflight_peak_during_reads"] <= 1
        assert st["write_inflight_peak"] >= 1
    finally:
        fs.shutdown()
    torch.cuda.synchronize()
    assert torch.equal(buf[: N_SRC * PAGE], buf[N_SRC * PAGE :])
