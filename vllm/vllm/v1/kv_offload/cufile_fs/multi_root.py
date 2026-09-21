# SPDX-License-Identifier: Apache-2.0
"""MultiRootFileMapper: KV 블록 하나를 여러 캐시 루트 중 하나에 배치한다.

루트 하나가 하나의 공급 경로다(로컬 NVMe, 원격 호스트 DRAM ramdisk, 원격 NVMe).
블록 해시 앞 8바이트를 가중치 합으로 나눈 나머지로 루트를 고르므로, 같은 키는 프로세스와
런을 넘어 늘 같은 루트·같은 경로가 된다. 가중치는 루트별 몫(GB/s 비 등)을 그대로 쓴다.

루트에 용량(capacity_bytes > 0)이 있으면 그 루트가 찼을 때 해시가 고른 루트 대신 아직 자리가
있는 루트들 사이에서 같은 규칙(가중치 비례 해시)으로 다시 고른다(spill). 그래서 어느 루트에
놓였는지는 매니저가 키별로 기억해야 한다(CuFileFsManager._root_of). 용량이 없는 루트만 있으면
예전과 같이 해시만으로 정해진다.

FileMapper와 같은 메서드(get_file_name, get_run_config, get_config_file_path)만 노출하므로
CuFileFsManager/CuFileFsWorker 쪽은 손대지 않는다. config.json은 루트마다 하나씩 쓴다
(get_config_file_paths). 루트별 매핑 횟수는 roots_stats()로 내보내 결과가 스스로를 설명하게 한다.
"""
import bisect
from collections.abc import Sequence

from vllm.v1.kv_offload.base import OffloadingSpec, OffloadKey, get_offload_block_hash
from vllm.v1.kv_offload.file_mapper import FileMapper

_BUCKET_BYTES = 8


def bucket_of(key: OffloadKey, modulo: int) -> int:
    """블록 해시 앞 8바이트를 big-endian 정수로 읽어 modulo로 나눈 나머지."""
    return int.from_bytes(get_offload_block_hash(key)[:_BUCKET_BYTES], "big") % modulo


def parse_root_dirs(spec: object) -> list[tuple[str, int, int]]:
    """cufile_fs_root_dirs 설정을 (디렉터리, 가중치, 용량 바이트) 목록으로 읽는다.

    받는 형식: [{"dir": str, "weight": int>0, "capacity_gb": float>=0}, ...].
    문자열만 있으면 가중치 1·용량 무제한으로 본다. capacity_gb 0(기본)은 무제한.
    """
    if not isinstance(spec, (list, tuple)) or not spec:
        raise ValueError("cufile_fs_root_dirs must be a non-empty list")
    out: list[tuple[str, int, int]] = []
    for item in spec:
        if isinstance(item, str):
            d, w, cap = item, 1, 0.0
        elif isinstance(item, dict):
            d = item.get("dir")
            w = int(item.get("weight", 1))
            cap = float(item.get("capacity_gb", 0) or 0)
        else:
            raise ValueError(f"cufile_fs_root_dirs entry must be str or dict: {item!r}")
        if not d:
            raise ValueError(f"cufile_fs_root_dirs entry has no dir: {item!r}")
        if w <= 0:
            raise ValueError(f"cufile_fs_root_dirs weight must be > 0: {item!r}")
        if cap < 0:
            raise ValueError(f"cufile_fs_root_dirs capacity_gb must be >= 0: {item!r}")
        out.append((str(d), w, int(cap * 2**30)))
    return out


class MultiRootFileMapper:
    """루트 여러 개를 가중치대로 나눠 쓰는 FileMapper 묶음."""

    def __init__(self, mappers: list[FileMapper], roots: Sequence[tuple]):
        if len(mappers) != len(roots) or not mappers:
            raise ValueError("MultiRootFileMapper: mappers and roots must match")
        self.mappers = mappers
        self.roots = [r[0] for r in roots]
        self.weights = [int(r[1]) for r in roots]
        # 루트별 용량(바이트). 0이면 무제한. (dir, weight) 두 짝만 주면 전부 무제한.
        self.capacities = [int(r[2]) if len(r) > 2 else 0 for r in roots]
        self.total_weight = sum(self.weights)
        # cum[i] = 0..i 가중치 누적. bisect_right(cum, bucket)이 곧 루트 인덱스다.
        self.cum: list[int] = []
        acc = 0
        for w in self.weights:
            acc += w
            self.cum.append(acc)
        self.mapped: list[int] = [0] * len(mappers)
        self.spilled: list[int] = [0] * len(mappers)  # 해시 루트가 차서 이 루트로 넘어온 횟수

    @classmethod
    def from_offloading_spec(
        cls,
        roots: Sequence[tuple],
        offloading_spec: OffloadingSpec,
        blocks_per_file: int = 1,
        parallel_agnostic: bool = False,
    ) -> "MultiRootFileMapper":
        mappers = [
            FileMapper.from_offloading_spec(
                root_dir=r[0],
                offloading_spec=offloading_spec,
                blocks_per_file=blocks_per_file,
                parallel_agnostic=parallel_agnostic,
            )
            for r in roots
        ]
        return cls(mappers, roots)

    @property
    def has_capacity(self) -> bool:
        return any(c > 0 for c in self.capacities)

    def hash_root(self, key: OffloadKey) -> int:
        """용량을 보지 않고 해시만으로 고른 루트."""
        return bisect.bisect_right(self.cum, bucket_of(key, self.total_weight))

    def root_index(self, key: OffloadKey, allowed: Sequence[bool] | None = None) -> int:
        """키를 놓을 루트. allowed[i]가 False인 루트(찬 루트)는 빼고 나머지 가중치로 다시 고른다.

        해시 루트가 allowed면 그대로다(용량이 없을 때와 같은 결과). 전부 막혀 있으면
        해시 루트를 돌려주고, 자리를 못 만드는 처리는 매니저 몫이다.
        """
        i = self.hash_root(key)
        if allowed is None or allowed[i]:
            return i
        idx = [j for j, ok in enumerate(allowed) if ok]
        if not idx:
            return i
        tw = sum(self.weights[j] for j in idx)
        b = bucket_of(key, tw)
        acc = 0
        for j in idx:
            acc += self.weights[j]
            if b < acc:
                return j
        return idx[-1]

    def file_name(self, key: OffloadKey, i: int) -> str:
        return self.mappers[i].get_file_name(key)

    def get_file_name(self, key: OffloadKey) -> str:
        """해시 루트의 경로. 용량 spill을 쓰는 매니저는 file_name(key, i)를 쓴다."""
        i = self.hash_root(key)
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
            dict(root=r, weight=w, capacity_gib=round(c / 2**30, 3), mapped=n, spilled=s)
            for r, w, c, n, s in zip(self.roots, self.weights, self.capacities, self.mapped, self.spilled)
        ]
