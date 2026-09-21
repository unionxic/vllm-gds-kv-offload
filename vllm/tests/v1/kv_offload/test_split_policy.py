# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the split-source KV policy."""

import pytest

from vllm.v1.kv_offload.split_policy import SplitContext, SplitPolicy

CHUNK_TOKENS = 256
# 16 MiB per chunk: at 3.4 GB/s one chunk is ~4.9 ms.
CHUNK_BYTES = 16 * 2**20


def _ctx(**kw):
    kw.setdefault("chunk_bytes", CHUNK_BYTES)
    return SplitContext(req_id="req", **kw)


def _fixed(frac: float) -> SplitPolicy:
    return SplitPolicy(mode="fixed", frac=frac, log_every=0)


def _model(**kw) -> SplitPolicy:
    return SplitPolicy(mode="model", log_every=0, **kw)


def test_off_never_splits():
    pol = SplitPolicy(mode="off", log_every=0)
    assert not pol.enabled
    assert pol.decide(8, CHUNK_TOKENS, ["ssd"] * 8, _ctx()) == 0


@pytest.mark.parametrize(
    "frac,H,expected",
    [
        (0.0, 8, 0),
        (0.5, 8, 4),
        (1.0, 8, 8),
        (0.25, 8, 2),
        (0.5, 7, 4),  # round(3.5) == 4 under banker's rounding of 3.5 -> 4
        (0.5, 1, 0),  # round(0.5) == 0
        (0.5, 3, 2),  # round(1.5) == 2
    ],
)
def test_fixed_fraction(frac, H, expected):
    pol = _fixed(frac)
    assert pol.decide(H, CHUNK_TOKENS, ["ssd"] * H, _ctx()) == expected


def test_fixed_clamped_to_H():
    assert _fixed(1.0).decide(4, CHUNK_TOKENS, ["ssd"] * 4, _ctx()) == 4
    assert _fixed(0.5).decide(0, CHUNK_TOKENS, [], _ctx()) == 0


def test_model_fast_load_does_not_recompute():
    # Loading 8 chunks from host costs ~11 ms; recomputing one chunk at
    # 215 tok/s costs ~1.2 s. Recompute is never worth it.
    pol = _model()
    assert pol.decide(8, CHUNK_TOKENS, ["host"] * 8, _ctx()) == 0


def test_model_slow_load_recomputes():
    # A very slow SSD and a fast prefill: loading dominates, so the policy
    # moves work onto the GPU until the two sides balance.
    pol = _model(rate_toks=100_000.0, bw_ssd_gbs=0.1)
    k = pol.decide(8, CHUNK_TOKENS, ["ssd"] * 8, _ctx())
    assert 0 < k <= 8
    # Balance point: T_compute(k) ~= T_load(H-k).
    t_compute = k * CHUNK_TOKENS / 100_000.0
    t_load = (8 - k) * CHUNK_BYTES / 0.1e9
    assert abs(t_compute - t_load) <= max(
        CHUNK_TOKENS / 100_000.0, CHUNK_BYTES / 0.1e9
    )


def test_model_is_tier_aware():
    """Chunks are loaded from the BACK, so only the tail's tiers matter.

    Two prefixes with the same host/ssd counts but opposite order must get
    different answers: putting the slow (ssd) chunks in the head lets the
    recompute retire them, leaving a fast host-only load.
    """
    pol = _model(rate_toks=100_000.0, bw_ssd_gbs=0.1, bw_host_gbs=12.3)
    ssd_first = ["ssd"] * 4 + ["host"] * 4
    host_first = ["host"] * 4 + ["ssd"] * 4
    k_ssd_first = pol.decide(8, CHUNK_TOKENS, ssd_first, _ctx())
    k_host_first = pol.decide(8, CHUNK_TOKENS, host_first, _ctx())
    # With the slow chunks up front, recomputing exactly them is enough.
    assert k_ssd_first == 4
    # With the slow chunks at the back they can only be avoided by
    # recomputing everything before them too.
    assert k_host_first > k_ssd_first


def test_model_prefers_smaller_k_on_ties():
    # Zero-cost load on both channels: every k >= 0 costs T_compute(k),
    # minimized at k = 0.
    pol = _model()
    assert pol.decide(6, CHUNK_TOKENS, ["host"] * 6, _ctx(chunk_bytes=0)) == 0


def test_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_KV_SPLIT", "off")
    assert not SplitPolicy.from_env().enabled

    monkeypatch.setenv("VLLM_KV_SPLIT", "fixed:0.25")
    pol = SplitPolicy.from_env()
    assert pol.mode == "fixed" and pol.frac == 0.25 and pol.describe() == "fixed:0.25"

    monkeypatch.setenv("VLLM_KV_SPLIT", "model")
    monkeypatch.setenv("VLLM_KV_SPLIT_RATE_TOKS", "512")
    pol = SplitPolicy.from_env()
    assert pol.mode == "model" and pol.rate_toks == 512.0

    monkeypatch.setenv("VLLM_KV_SPLIT", "bogus")
    with pytest.raises(ValueError):
        SplitPolicy.from_env()

    monkeypatch.setenv("VLLM_KV_SPLIT", "fixed:1.5")
    with pytest.raises(ValueError):
        SplitPolicy.from_env()


def test_stats_counters():
    from vllm.v1.kv_offload.split_policy import LAST_SPLIT_STATS, reset_split_stats

    reset_split_stats("fixed:0.5")
    pol = _fixed(0.5)
    for _ in range(3):
        pol.decide(4, CHUNK_TOKENS, ["ssd"] * 4, _ctx())
    assert LAST_SPLIT_STATS["decisions"] == 3
    assert pol.n_split == 3
