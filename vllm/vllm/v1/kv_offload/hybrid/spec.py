# SPDX-License-Identifier: Apache-2.0
"""HybridSpec: KV 블록을 host(pinned) 티어와 SSD(GDS) 티어 두 단으로 오프로드하는 스펙.

티어 1은 vLLM in-tree CPU 티어(pinned host memory)를 그대로 쓰고, 티어 2는 포크의
cuFile 백엔드(csrc/kv_offload/cufile_fs.cpp)로 파일에 직접 읽고 쓴다. 적재 경로는
host→GPU 또는 SSD→GPU(GDS) 둘 중 하나이고, SSD→host 승격 단계는 두지 않는다.

kv_connector_extra_config 키
  spec_name: "HybridSpec"
  hybrid_host_gb: host 티어 크기(GB, 10^9 바이트). 블록 수는 CPUOffloadingSpec과 같은
      산식(worker_kv_bytes_per_block * world_size * blocks_per_chunk 정렬)으로 구한다
  hybrid_host_policy: lru | arc (기본 lru)
  hybrid_placement: host_first(기본) | profile | ratio
  hybrid_profile: placement=profile일 때 읽을 JSON 경로
      {"hashes": {"<block hash hex>": reuse, ...}, "min_reuse": N}
  hybrid_host_share: placement=ratio일 때 host 티어로 보낼 블록 몫(0..1, 필수)
  hybrid_write_through: true(기본)면 store 때 host와 SSD 양쪽에 모두 쓴다
  cufile_fs_* : 파일 티어에 그대로 전달(cufile_fs_root_dir 또는 cufile_fs_root_dirs 필수).
      cufile_fs_root_dirs를 주면 파일 티어가 루트 여러 개로 갈린다. cufile_fs/spec.py 참고

host 티어 축출분을 SSD로 내리는 write-back(demotion)은 구현하지 않았다. 이유는
hybrid/manager.py 모듈 docstring 참고.
"""
from dataclasses import replace
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.cufile_fs.spec import CuFileFsSpec
from vllm.v1.kv_offload.hybrid.manager import HybridManager

logger = init_logger(__name__)

LAST_WORKER = None  # in-process 계측용
LAST_MANAGER = None


def _as_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes")


class HybridSpec(OffloadingSpec):
    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        # get_stats()가 host 티어 지표를 그대로 내보내므로 정의도 그대로 쓴다.
        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        host_gb = float(self.extra_config.get("hybrid_host_gb", 0))
        if host_gb <= 0:
            raise ValueError(
                "hybrid_host_gb must be > 0 in kv_connector_extra_config"
            )
        self.host_gb = host_gb
        self.host_policy = str(self.extra_config.get("hybrid_host_policy", "lru")).lower()
        self.placement = str(self.extra_config.get("hybrid_placement", "host_first")).lower()
        self.profile_path = self.extra_config.get("hybrid_profile")
        _share = self.extra_config.get("hybrid_host_share")
        self.host_share = None if _share is None else float(_share)
        self.write_through = _as_bool(self.extra_config.get("hybrid_write_through"), True)

        # host 티어는 in-tree CPUOffloadingSpec을 그대로 재사용한다(블록 수 산식,
        # 워커 생성, 지표 정의가 한 곳에만 있도록).
        cpu_extra = dict(self.extra_config)
        cpu_extra["cpu_bytes_to_use"] = int(host_gb * 1e9)
        cpu_extra["eviction_policy"] = self.host_policy
        self._cpu_spec = CPUOffloadingSpec(replace(config, extra_config=cpu_extra))
        # SSD 티어는 포크의 CuFileFsSpec을 그대로 재사용한다(cufile_fs_* 키를 그대로 읽음).
        self._fs_spec = CuFileFsSpec(config)

        self.num_host_blocks = self._cpu_spec.num_blocks
        self.kv_bytes_per_chunk = self._cpu_spec.kv_bytes_per_chunk

        self._manager: HybridManager | None = None
        self._worker: OffloadingWorker | None = None

        logger.info(
            "HybridSpec: host %.1f GB -> %d blocks (%d bytes/chunk, policy=%s), "
            "file roots=%s, placement=%s, host_share=%s, write_through=%s",
            host_gb,
            self.num_host_blocks,
            self.kv_bytes_per_chunk,
            self.host_policy,
            self._fs_spec.root_dirs or self._fs_spec.root_dir,
            self.placement,
            self.host_share,
            self.write_through,
        )

    def get_manager(self) -> OffloadingManager:
        global LAST_MANAGER
        if self._manager is None:
            self._manager = HybridManager(
                host=self._cpu_spec.get_manager(),  # type: ignore[arg-type]
                ssd=self._fs_spec.get_manager(),  # type: ignore[arg-type]
                write_through=self.write_through,
                placement=self.placement,
                profile_path=self.profile_path,
                host_share=self.host_share,
            )
            LAST_MANAGER = self._manager
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        global LAST_WORKER
        if self._worker is None:
            from vllm.v1.kv_offload.hybrid.worker import HybridWorker

            self._worker = HybridWorker(
                host_worker=self._cpu_spec.get_worker(kv_caches),
                ssd_worker=self._fs_spec.get_worker(kv_caches),
                blocks_per_chunk=self.blocks_per_chunk,
            )
            LAST_WORKER = self._worker
        return self._worker
