# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BAR1이 작은 GPU에서 Mooncake store를 쓰기 위한 host pinned 경유 래퍼.

MooncakeStoreWorker는 KV 텐서를 통째로 transfer engine에 등록해 GPUDirect RDMA로
주고받는다. 이 등록은 GPU BAR1 구멍을 소모하므로 BAR1이 KV 텐서보다 작은 GPU
(예: Quadro RTX 5000, BAR1 256 MiB)에서는 ibv_reg_mr이 EFAULT로 실패하고, 이후
모든 put/get이 AddressNotRegistered → TRANSFER_FAIL로 떨어진다.

이 래퍼는 그런 GPU에서 KV 텐서 대신 pinned host 슬랩을 등록해 두고 GPU↔host
복사를 끼워 넣는다. 원격으로 나가는 구간은 그대로 Mooncake transfer engine의
RDMA이고, 키 이름·객체 경계·store 프로토콜은 전혀 바뀌지 않는다.

슬랩은 스레드마다 하나씩 잡는다(송신 스레드 1개 + 수신 스레드 N개가 동시에
들어오므로). 한 배치가 슬랩보다 크면 키 단위로 잘라 여러 번 왕복한다.
"""

import threading
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 4096


def _align_up(value: int, alignment: int = _ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


class HostStagingStore:
    """``batch_put_from_multi_buffers``/``batch_get_into_multi_buffers``만 가로채는 래퍼.

    나머지 호출은 원본 store에 그대로 위임한다.
    """

    def __init__(
        self,
        store: Any,
        regions: list[tuple[int, int, Any]],
        slab_bytes: int,
        num_slabs: int,
    ) -> None:
        self._store = store
        # (base_addr, nbytes, untyped_storage). 주소 → (storage, offset) 역매핑용.
        self._regions = sorted(regions, key=lambda r: r[0])
        self._slab_bytes = int(slab_bytes)
        self._num_slabs = max(1, int(num_slabs))
        self._local = threading.local()
        self._slabs: list[Any] = []  # 프로세스가 끝날 때까지 붙잡아 둘 참조
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        # __init__에서 세팅하는 속성들은 여기로 오지 않는다(이미 인스턴스에 있음).
        return getattr(object.__getattribute__(self, "_store"), name)

    # ---- 슬랩 ----

    def _slab(self) -> tuple[torch.Tensor, torch.cuda.Stream]:
        slab = getattr(self._local, "slab", None)
        if slab is not None:
            return slab
        with self._lock:
            if len(self._slabs) >= self._num_slabs:
                logger.warning(
                    "Mooncake host staging: 슬랩 수가 예상(%d)을 넘음. "
                    "pinned 예산이 그만큼 늘어난다.",
                    self._num_slabs,
                )
        buf = torch.empty(self._slab_bytes, dtype=torch.uint8, pin_memory=True)
        ret = self._store.register_buffer(buf.data_ptr(), self._slab_bytes)
        if ret != 0:
            raise RuntimeError(f"Mooncake host staging 슬랩 등록 실패: ret={ret}")
        slab = (buf, torch.cuda.Stream())
        with self._lock:
            self._slabs.append(slab)
        self._local.slab = slab
        logger.info(
            "Mooncake host staging slab 등록: %.2f GiB (thread %s, 슬랩 %d개째)",
            self._slab_bytes / 2**30,
            threading.current_thread().name,
            len(self._slabs),
        )
        return slab

    def _gpu_view(self, addr: int, size: int) -> torch.Tensor:
        for base, nbytes, storage in self._regions:
            if base <= addr and addr + size <= base + nbytes:
                view = torch.empty(0, dtype=torch.uint8, device=storage.device)
                view.set_(storage, addr - base, (size,), (1,))
                return view
        raise KeyError(f"등록된 KV 구간 밖의 주소: {addr:#x}+{size}")

    def _batches(self, sizes: list[list[int]]) -> list[list[int]]:
        """키 인덱스를 슬랩 용량에 맞는 묶음으로 자른다."""
        cap = self._slab_bytes
        out: list[list[int]] = []
        cur: list[int] = []
        used = 0
        for i, seg in enumerate(sizes):
            need = sum(_align_up(s) for s in seg)
            if need > cap:
                raise ValueError(
                    f"키 하나({need} B)가 staging 슬랩({cap} B)보다 큼. "
                    "host_staging_gb를 키울 것"
                )
            if used + need > cap and cur:
                out.append(cur)
                cur = []
                used = 0
            cur.append(i)
            used += need
        if cur:
            out.append(cur)
        return out

    def _map_to_slab(
        self,
        buf: torch.Tensor,
        idx_group: list[int],
        sizes: list[list[int]],
    ) -> tuple[list[list[int]], list[tuple[int, int, int]]]:
        """묶음 안 각 세그먼트에 슬랩 오프셋을 배정.

        반환: (키별 host 주소 리스트, (키 인덱스, 세그먼트 인덱스, 오프셋) 목록)
        """
        base = buf.data_ptr()
        host_ptrs: list[list[int]] = []
        slots: list[tuple[int, int, int]] = []
        off = 0
        for i in idx_group:
            ptrs = []
            for j, s in enumerate(sizes[i]):
                ptrs.append(base + off)
                slots.append((i, j, off))
                off = _align_up(off + s)
            host_ptrs.append(ptrs)
        return host_ptrs, slots

    # ---- 가로채는 두 호출 ----

    def batch_put_from_multi_buffers(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        sizes: list[list[int]],
        config: Any = None,
    ) -> list[int]:
        buf, stream = self._slab()
        res: list[int] = [0] * len(keys)
        for idx_group in self._batches(sizes):
            host_ptrs, slots = self._map_to_slab(buf, idx_group, sizes)
            with torch.cuda.stream(stream):
                for i, j, off in slots:
                    size = sizes[i][j]
                    buf[off : off + size].copy_(
                        self._gpu_view(all_buffer_ptrs[i][j], size), non_blocking=True
                    )
            stream.synchronize()
            sub_keys = [keys[i] for i in idx_group]
            sub_sizes = [sizes[i] for i in idx_group]
            if config is None:
                sub = self._store.batch_put_from_multi_buffers(
                    sub_keys, host_ptrs, sub_sizes
                )
            else:
                sub = self._store.batch_put_from_multi_buffers(
                    sub_keys, host_ptrs, sub_sizes, config
                )
            for pos, i in enumerate(idx_group):
                res[i] = sub[pos]
        return res

    def batch_get_into_multi_buffers(
        self,
        keys: list[str],
        all_buffer_ptrs: list[list[int]],
        sizes: list[list[int]],
    ) -> list[int]:
        buf, stream = self._slab()
        res: list[int] = [0] * len(keys)
        for idx_group in self._batches(sizes):
            host_ptrs, slots = self._map_to_slab(buf, idx_group, sizes)
            sub_keys = [keys[i] for i in idx_group]
            sub_sizes = [sizes[i] for i in idx_group]
            sub = self._store.batch_get_into_multi_buffers(
                sub_keys, host_ptrs, sub_sizes
            )
            ok = {i for pos, i in enumerate(idx_group) if sub[pos] >= 0}
            with torch.cuda.stream(stream):
                for i, j, off in slots:
                    if i not in ok:
                        continue
                    size = sizes[i][j]
                    self._gpu_view(all_buffer_ptrs[i][j], size).copy_(
                        buf[off : off + size], non_blocking=True
                    )
            stream.synchronize()
            for pos, i in enumerate(idx_group):
                res[i] = sub[pos]
        return res

    def close(self) -> int:
        with self._lock:
            slabs, self._slabs = self._slabs, []
        for buf, _ in slabs:
            try:
                self._store.unregister_buffer(buf.data_ptr())
            except Exception:  # noqa: BLE001
                pass
        return self._store.close()
