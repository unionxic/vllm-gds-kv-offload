# SPDX-License-Identifier: Apache-2.0
"""Split-source KV: head recompute on GPU, tail load from the offload tiers.

For a request whose prefix hits H chunks in the offload tiers, the first k
chunks ("head") are recomputed on the GPU while the remaining H-k chunks
("tail") are read from host/SSD at the same time. KV of a token depends only
on preceding tokens, so a loaded tail of an identical prefix is valid.

Modes (env VLLM_KV_SPLIT):
  off           k = 0, load everything (previous behavior)
  fixed:<frac>  k = round(frac * H)
  serial:<frac> k = round(frac * H), 다만 겹치지 않음: 앞 (H-k)를 보통의 프리픽스 적중으로 적재한 뒤 뒤 k를 계산(합 모형의 대조군)
  model         k minimizing max(T_compute(k), T_load_host(k), T_load_ssd(k))

The load runs concurrently with the head compute, so the cost of a split is
the max of the two, not the sum. Host and SSD are separate channels and also
run concurrently, hence three terms.
"""

import os
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)

HOST = "host"
SSD = "ssd"

# In-process observability, read by the runner (see 11-observability/run_obs.py).
LAST_SPLIT_STATS: dict = dict(
    mode="off",
    decisions=0,
    requests_split=0,
    head_tokens_recomputed_total=0,
    tail_tokens_loaded_total=0,
    tail_wait_events=0,
    tail_wait_s_total=0.0,
)


def nvtx_mark(msg: str) -> None:
    """NVTX marker for the split decision / tail arrival, cheap and optional."""
    try:
        import torch

        torch.cuda.nvtx.mark(msg)
    except Exception:  # pragma: no cover - never break scheduling on a marker
        pass


def reset_split_stats(mode: str) -> None:
    LAST_SPLIT_STATS.update(
        mode=mode,
        decisions=0,
        requests_split=0,
        head_tokens_recomputed_total=0,
        tail_tokens_loaded_total=0,
        tail_wait_events=0,
        tail_wait_s_total=0.0,
    )


@dataclass
class SplitContext:
    """Per-decision inputs that are not part of the chunk list."""

    # KV bytes of one offloaded chunk (both K and V, all layers).
    chunk_bytes: int = 0
    req_id: str = ""
    # Optional per-request overrides; None means use the policy default.
    rate_toks: float | None = None
    bw_host_gbs: float | None = None
    bw_ssd_gbs: float | None = None


@dataclass
class SplitPolicy:
    """Decides how many head chunks to recompute."""

    mode: str = "off"
    frac: float = 0.0
    # Prefill throughput used to price the head recompute (tokens/s).
    rate_toks: float = 215.0
    bw_host_gbs: float = 12.3
    bw_ssd_gbs: float = 3.4
    log_every: int = 16
    n_decisions: int = field(default=0, init=False)
    n_split: int = field(default=0, init=False)

    @classmethod
    def from_env(cls) -> "SplitPolicy":
        raw = os.getenv("VLLM_KV_SPLIT", "off").strip().lower()
        mode, frac = "off", 0.0
        if raw in ("", "off", "0", "none"):
            mode = "off"
        elif raw == "model":
            mode = "model"
        elif raw.startswith("fixed") or raw.startswith("serial"):
            mode = "serial" if raw.startswith("serial") else "fixed"
            _, _, tail = raw.partition(":")
            try:
                frac = float(tail) if tail else 0.5
            except ValueError:
                raise ValueError(f"bad VLLM_KV_SPLIT fraction: {raw!r}") from None
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"VLLM_KV_SPLIT fraction out of [0,1]: {frac}")
        else:
            raise ValueError(
                f"unknown VLLM_KV_SPLIT: {raw!r} (off | fixed:<frac> | serial:<frac> | model)"
            )
        pol = cls(
            mode=mode,
            frac=frac,
            rate_toks=float(os.getenv("VLLM_KV_SPLIT_RATE_TOKS", "215")),
            bw_host_gbs=float(os.getenv("VLLM_KV_SPLIT_BW_HOST_GBS", "12.3")),
            bw_ssd_gbs=float(os.getenv("VLLM_KV_SPLIT_BW_SSD_GBS", "3.4")),
            log_every=int(os.getenv("VLLM_KV_SPLIT_LOG_EVERY", "16")),
        )
        reset_split_stats(pol.describe())
        if mode != "off":
            logger.info(
                "KV split-source: %s (rate %.1f tok/s, host %.1f GB/s, ssd %.1f GB/s)",
                pol.describe(),
                pol.rate_toks,
                pol.bw_host_gbs,
                pol.bw_ssd_gbs,
            )
        return pol

    def describe(self) -> str:
        if self.mode in ("fixed", "serial"):
            return f"{self.mode}:{self.frac:g}"
        return self.mode

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _cost(
        self,
        k: int,
        H: int,
        chunk_tokens: int,
        tiers: list[str],
        ctx: SplitContext,
    ) -> float:
        rate = ctx.rate_toks or self.rate_toks
        b_host = ctx.bw_host_gbs or self.bw_host_gbs
        b_ssd = ctx.bw_ssd_gbs or self.bw_ssd_gbs
        # The loaded chunks are the LAST H-k, so their tiers are tiers[k:H].
        tail = tiers[k:H]
        n_host = sum(1 for t in tail if t == HOST)
        n_ssd = len(tail) - n_host
        t_compute = (k * chunk_tokens) / rate if rate > 0 else 0.0
        t_host = (n_host * ctx.chunk_bytes) / (b_host * 1e9) if b_host > 0 else 0.0
        t_ssd = (n_ssd * ctx.chunk_bytes) / (b_ssd * 1e9) if b_ssd > 0 else 0.0
        return max(t_compute, t_host, t_ssd)

    def decide(
        self,
        H_chunks: int,
        chunk_tokens: int,
        tiers_per_chunk: list[str],
        ctx: SplitContext,
    ) -> int:
        """Return the number of leading (head) chunks to recompute."""
        H = int(H_chunks)
        if H <= 0 or not self.enabled:
            return 0
        assert len(tiers_per_chunk) >= H, (
            f"tiers_per_chunk has {len(tiers_per_chunk)} entries for H={H}"
        )

        if self.mode in ("fixed", "serial"):
            k = int(round(self.frac * H))
        else:
            # Ties resolve to the smaller k (less recompute).
            k = min(
                range(H + 1),
                key=lambda c: (self._cost(c, H, chunk_tokens, tiers_per_chunk, ctx), c),
            )
        k = max(0, min(H, k))

        self.n_decisions += 1
        LAST_SPLIT_STATS["decisions"] = self.n_decisions
        if k > 0:
            self.n_split += 1
        if self.log_every > 0 and self.n_decisions % self.log_every == 1:
            logger.info(
                "KV split %s: req %s H=%d chunks (%d host / %d ssd) -> head k=%d",
                self.describe(),
                ctx.req_id,
                H,
                sum(1 for t in tiers_per_chunk[:H] if t == HOST),
                sum(1 for t in tiers_per_chunk[:H] if t != HOST),
                k,
            )
        return k
