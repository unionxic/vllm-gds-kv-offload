# SPDX-License-Identifier: Apache-2.0
"""HybridSpec의 LoadStoreSpec: 한 요청의 키 목록을 host 티어와 SSD 티어로 쪼갠 결과."""
from dataclasses import dataclass, field

from vllm.v1.kv_offload.base import LoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cufile_fs.spec import FileLoadStoreSpec

MEDIUM_HYBRID = "HYBRID"


@dataclass
class HybridLoadStoreSpec(LoadStoreSpec):
    """host/SSD 두 티어의 하위 spec과 각 티어가 맡은 키 위치.

    cpu_pos/file_pos는 이 spec이 가리키는 키 목록(load면 prepare_load의 keys,
    store면 PrepareStoreOutput.keys_to_store) 안에서의 인덱스다. 워커는 이 위치로
    GPU 블록 목록(GPULoadStoreSpec)을 같은 순서로 잘라 두 하위 워커에 넘긴다.
    load에서는 두 위치 집합이 서로소이고, write-through store에서는 겹칠 수 있다
    (같은 GPU 블록을 host와 SSD가 각각 읽는다).
    위치는 항상 오름차순이고, 하위 spec의 블록/경로 순서와 1:1로 맞는다.
    """

    n_keys: int = 0
    cpu: CPULoadStoreSpec | None = None
    file: FileLoadStoreSpec | None = None
    cpu_pos: list[int] = field(default_factory=list)
    file_pos: list[int] = field(default_factory=list)

    @staticmethod
    def medium() -> str:
        return MEDIUM_HYBRID

    def __repr__(self) -> str:
        return (
            f"HybridLoadStoreSpec(n={self.n_keys}, "
            f"host={len(self.cpu_pos)}, ssd={len(self.file_pos)})"
        )
