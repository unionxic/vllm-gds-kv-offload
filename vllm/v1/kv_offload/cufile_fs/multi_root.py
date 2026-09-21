# SPDX-License-Identifier: Apache-2.0
"""MultiRootFileMapper: KV 블록 하나를 여러 캐시 루트 중 하나에 결정적으로 배치한다.

루트 하나가 하나의 공급 경로다(로컬 NVMe, 원격 호스트 DRAM ramdisk, 원격 NVMe).
블록 해시 앞 8바이트를 가중치 합으로 나눈 나머지로 루트를 고르므로, 같은 키는 프로세스와
런을 넘어 늘 같은 루트·같은 경로가 된다. 가중치는 루트별 몫(GB/s 비 등)을 그대로 쓴다.

FileMapper와 같은 메서드(get_file_name, get_run_config, get_config_file_path)만 노출하므로
CuFileFsManager/CuFileFsWorker 쪽은 손대지 않는다. config.json은 루트마다 하나씩 쓴다
(get_config_file_paths). 루트별 매핑 횟수는 roots_stats()로 내보내 결과가 스스로를 설명하게 한다.
"""
import bisect

from vllm.v1.kv_offload.base import OffloadingSpec, OffloadKey, get_offload_block_hash
from vllm.v1.kv_offload.file_mapper import FileMapper

_BUCKET_BYTES = 8


def bucket_of(key: OffloadKey, modulo: int) -> int:
    """블록 해시 앞 8바이트를 big-endian 정수로 읽어 modulo로 나눈 나머지."""
    return int.from_bytes(get_offload_block_hash(key)[:_BUCKET_BYTES], "big") % modulo


def parse_root_dirs(spec: object) -> list[tuple[str, int]]:
    """cufile_fs_root_dirs 설정을 (디렉터리, 가중치) 목록으로 읽는다.

    받는 형식: [{"dir": str, "weight": int>0}, ...]. 문자열만 있으면 가중치 1로 본다.
    """
    if not isinstance(spec, (list, tuple)) or not spec:
        raise ValueError("cufile_fs_root_dirs must be a non-empty list")
    out: list[tuple[str, int]] = []
    for item in spec:
        if isinstance(item, str):
            d, w = item, 1
        elif isinstance(item, dict):
            d = item.get("dir")
            w = int(item.get("weight", 1))
        else:
            raise ValueError(f"cufile_fs_root_dirs entry must be str or dict: {item!r}")
        if not d:
            raise ValueError(f"cufile_fs_root_dirs entry has no dir: {item!r}")
        if w <= 0:
            raise ValueError(f"cufile_fs_root_dirs weight must be > 0: {item!r}")
        out.append((str(d), w))
    return out


class MultiRootFileMapper:
    """루트 여러 개를 가중치대로 나눠 쓰는 FileMapper 묶음."""

    def __init__(self, mappers: list[FileMapper], roots: list[tuple[str, int]]):
        if len(mappers) != len(roots) or not mappers:
            raise ValueError("MultiRootFileMapper: mappers and roots must match")
        self.mappers = mappers
        self.roots = [d for d, _ in roots]
        self.weights = [w for _, w in roots]
        self.total_weight = sum(self.weights)
        # cum[i] = 0..i 가중치 누적. bisect_right(cum, bucket)이 곧 루트 인덱스다.
        self.cum: list[int] = []
        acc = 0
        for w in self.weights:
            acc += w
            self.cum.append(acc)
        self.mapped: list[int] = [0] * len(mappers)

    @classmethod
    def from_offloading_spec(
        cls,
        roots: list[tuple[str, int]],
        offloading_spec: OffloadingSpec,
        blocks_per_file: int = 1,
        parallel_agnostic: bool = False,
    ) -> "MultiRootFileMapper":
        mappers = [
            FileMapper.from_offloading_spec(
                root_dir=d,
                offloading_spec=offloading_spec,
                blocks_per_file=blocks_per_file,
                parallel_agnostic=parallel_agnostic,
            )
            for d, _ in roots
        ]
        return cls(mappers, roots)

    def root_index(self, key: OffloadKey) -> int:
        return bisect.bisect_right(self.cum, bucket_of(key, self.total_weight))

    def get_file_name(self, key: OffloadKey) -> str:
        i = self.root_index(key)
        self.mapped[i] += 1
        return self.mappers[i].get_file_name(key)

    def get_run_config(self) -> dict:
        # 루트가 달라도 fields는 같다(루트 경로는 해시에 들어가지 않음).
        return self.mappers[0].get_run_config()

    def get_config_file_path(self) -> str:
        return self.mappers[0].get_config_file_path()

    def get_config_file_paths(self) -> list[str]:
        return [m.get_config_file_path() for m in self.mappers]

    def roots_stats(self) -> list[dict]:
        return [
            dict(root=r, weight=w, mapped=n)
            for r, w, n in zip(self.roots, self.weights, self.mapped)
        ]
