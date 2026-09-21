# SPDX-License-Identifier: Apache-2.0
"""HybridManager: 스케줄러 쪽에서 host(pinned) 티어와 SSD(GDS) 티어를 함께 관리.

티어 구성
  host: vllm.v1.kv_offload.cpu.manager.CPUOffloadingManager (정책 lru|arc)
  ssd : vllm.v1.kv_offload.cufile_fs.spec.CuFileFsManager (파일 존재 = 적재됨)

배치 정책(hybrid_placement)
  host_first: 모든 키가 host 후보(기본, 예전 동작)
  profile   : 재사용 프로파일이 고른 키만 host 후보
  ratio     : 블록 해시 버킷이 hybrid_host_share 몫 안에 드는 키만 host 후보. 한 프리픽스의
              블록이 host와 파일 티어에 정해진 비율로 흩어져, 적중 때 두 경로에서 동시에 올라온다

조회 우선순위는 host > ssd > miss. 적재는 티어별로 나눠 하위 spec 두 개를 만들고,
HybridLoadStoreSpec이 어느 키가 어느 티어인지 위치로 알려준다. 승격(SSD→host 재배치)은
하지 않는다. SSD 적중 블록은 GDS로 GPU에 바로 올라간다.

설계에서 벗어난 점: write-back(host 축출분을 SSD로 내리는 demotion)은 구현하지 않았다.
CPUOffloadingManager.prepare_store는 축출된 블록을 같은 호출 안에서 _free_block 한 뒤
곧바로 _allocate_blocks로 재사용하므로, evicted_keys를 돌려받은 시점에는 그 슬롯이 이미
다른 키 몫이고 GPU→host 쓰기가 예약돼 있을 수 있다. 스케줄러 쪽에서 /dev/shm 공유 영역을
읽어 파일로 내리려 해도 읽는 사이에 덮일 수 있어 정확성을 보장할 수 없고, 고치려면
cpu/manager.py를 수정해야 한다(금지). 그래서 두 모드만 둔다.
  hybrid_write_through=true (기본): store 때 host와 SSD 양쪽에 모두 쓴다. host 축출은
    데이터 손실이 아니므로 evicted_keys로 보고하지 않는다.
  hybrid_write_through=false: 배치 정책이 고른 한 티어에만 쓴다(demotion 없음).
    host 축출은 진짜 손실이므로 evicted_keys로 보고한다.
"""
import json
from collections.abc import Collection, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cufile_fs.multi_root import bucket_of
from vllm.v1.kv_offload.cufile_fs.spec import CuFileFsManager
from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec

logger = init_logger(__name__)

PLACEMENTS = ("host_first", "profile", "ratio")
RATIO_SCALE = 1000  # ratio 배치의 버킷 수. 몫의 해상도 0.1 %
LAST_MANAGER = None  # in-process 계측용


class _ReuseProfile:
    """cufile_fs admission 프로파일과 같은 형식의 재사용 횟수표.

    {"hashes": {"<block hash hex>": reuse, ...}, "min_reuse": N}
    재사용 횟수가 min_reuse 이상인 키만 host 티어 후보로 본다.
    """

    def __init__(self, path: str):
        with open(path) as f:
            prof = json.load(f)
        self.hashes: dict[str, int] = {
            str(h): int(v) for h, v in (prof.get("hashes") or {}).items()
        }
        self.min_reuse: int = int(prof.get("min_reuse", 1))
        logger.info(
            "Hybrid placement=profile: %d hashes from %s (min_reuse=%d)",
            len(self.hashes),
            path,
            self.min_reuse,
        )

    def wants_host(self, key: OffloadKey) -> bool:
        return self.hashes.get(get_offload_block_hash(key).hex(), -1) >= self.min_reuse


class HybridManager(OffloadingManager):
    """host 티어와 SSD 티어를 묶은 OffloadingManager."""

    def __init__(
        self,
        host: CPUOffloadingManager,
        ssd: CuFileFsManager,
        write_through: bool = True,
        placement: str = "host_first",
        profile_path: str | None = None,
        host_share: float | None = None,
    ):
        if placement not in PLACEMENTS:
            raise ValueError(
                f"unknown hybrid_placement: {placement} (expected {PLACEMENTS})"
            )
        self.host = host
        self.ssd = ssd
        self.write_through = bool(write_through)
        self.placement = placement
        self.profile: _ReuseProfile | None = None
        if placement == "profile":
            if not profile_path:
                raise ValueError("hybrid_placement=profile requires hybrid_profile")
            self.profile = _ReuseProfile(profile_path)
        self.host_share: float | None = None
        self._host_buckets = 0
        if placement == "ratio":
            if host_share is None:
                raise ValueError("hybrid_placement=ratio requires hybrid_host_share")
            host_share = float(host_share)
            if not 0.0 <= host_share <= 1.0:
                raise ValueError(f"hybrid_host_share must be in [0, 1]: {host_share}")
            self.host_share = host_share
            self._host_buckets = round(RATIO_SCALE * host_share)
            logger.info(
                "Hybrid placement=ratio: host share %.3f (%d/%d buckets)",
                host_share,
                self._host_buckets,
                RATIO_SCALE,
            )

        # 티어 내용 거울. 축출된 키가 "정말 사라졌는지" 판정하는 데만 쓴다.
        self._host_keys: set[OffloadKey] = set()
        self._ssd_keys: set[OffloadKey] = set()
        # 진행 중인 store가 어느 티어로 갔는지(complete_store를 나눠 보내려고).
        self._store_host: set[OffloadKey] = set()
        self._store_ssd: set[OffloadKey] = set()
        # 진행 중인 load의 티어별 참조 수(complete_load를 나눠 보내려고).
        self._load_host: dict[OffloadKey, int] = {}
        self._load_ssd: dict[OffloadKey, int] = {}

        # 계측
        self.n_hit_host = 0
        self.n_hit_ssd = 0
        self.n_miss = 0
        self.n_pending_host = 0  # host에 쓰는 중이라 요청을 미룬 lookup 수
        self.n_pending_ssd = 0  # 파일 티어 쓰기를 기다리라고 한 lookup 수
        self.n_store_host = 0
        self.n_store_ssd = 0
        self.n_evicted_host = 0
        self.n_evicted_ssd = 0
        self.n_lost = 0  # 두 티어 모두에서 사라져 evicted_keys로 보고한 키 수
        self.n_host_alloc_fail = 0

        global LAST_MANAGER
        LAST_MANAGER = self
        logger.info(
            "Hybrid manager: host blocks=%d policy=%s, placement=%s, write_through=%s",
            host._num_blocks,
            type(host._policy).__name__,
            placement,
            self.write_through,
        )

    # --- 내부 도우미 ---

    def _host_ready(self, key: OffloadKey, req_context: ReqContext) -> bool:
        """host에 있고 읽을 수 있는지. CPUOffloadingManager.lookup은
        store_threshold < 2일 때 부작용이 없어 그대로 다시 불러도 된다."""
        return self.host.lookup(key, req_context) == LookupResult.HIT

    def _wants_host(self, key: OffloadKey) -> bool:
        if self.placement == "host_first":
            return True
        if self.placement == "ratio":
            # 파일 티어 루트 고르기와 같은 64비트 해시. 런이 달라도 같은 키는 같은 티어다.
            return bucket_of(key, RATIO_SCALE) < self._host_buckets
        assert self.profile is not None
        return self.profile.wants_host(key)

    # --- OffloadingManager 인터페이스 ---

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self.ssd.on_new_request(req_context)
        return self.host.on_new_request(req_context)

    def _traced(self, key: OffloadKey, req_context: ReqContext,
                res: LookupResult, tier: str) -> LookupResult:
        """파일 티어 계측이 켜져 있으면 상위 판정도 같은 파일에 남긴다(ev=lkh)."""
        tracer = getattr(self.ssd, "tracer", None)
        if tracer is not None:
            tracer.emit("lkh", key, req_context, res=res.name.lower(), tier=tier)
        return res

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        host_res = self.host.lookup(key, req_context)
        if host_res == LookupResult.HIT:
            self.n_hit_host += 1
            return self._traced(key, req_context, LookupResult.HIT, "host")
        ssd_res = self.ssd.lookup(key, req_context)
        if ssd_res == LookupResult.HIT:
            self.n_hit_ssd += 1
            return self._traced(key, req_context, LookupResult.HIT, "ssd")
        if host_res != LookupResult.MISS:
            # host에 쓰는 중(HIT_PENDING)이고 SSD에는 아직 파일이 없다.
            self.n_pending_host += 1
            return self._traced(key, req_context, host_res, "host")
        if ssd_res != LookupResult.MISS:
            # 파일 티어가 쓰기 완료를 기다리라고 했다(cufile_fs_pending_wait).
            self.n_pending_ssd += 1
            return self._traced(key, req_context, ssd_res, "ssd")
        self.n_miss += 1
        return self._traced(key, req_context, LookupResult.MISS, "none")

    def key_tiers(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> list[str]:
        """host에 읽을 준비가 된 키만 host, 나머지는 SSD."""
        return [
            "host" if self._host_ready(k, req_context) else "ssd" for k in keys
        ]

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        key_list = list(keys)
        cpu_pos: list[int] = []
        file_pos: list[int] = []
        host_keys: list[OffloadKey] = []
        ssd_keys: list[OffloadKey] = []
        for i, key in enumerate(key_list):
            if self._host_ready(key, req_context):
                cpu_pos.append(i)
                host_keys.append(key)
            else:
                file_pos.append(i)
                ssd_keys.append(key)

        cpu_spec = None
        if host_keys:
            cpu_spec = self.host.prepare_load(host_keys, req_context)
            for key in host_keys:
                self._load_host[key] = self._load_host.get(key, 0) + 1
        file_spec = None
        if ssd_keys:
            file_spec = self.ssd.prepare_load(ssd_keys, req_context)
            for key in ssd_keys:
                self._load_ssd[key] = self._load_ssd.get(key, 0) + 1

        return HybridLoadStoreSpec(
            n_keys=len(key_list),
            cpu=cpu_spec,
            file=file_spec,
            cpu_pos=cpu_pos,
            file_pos=file_pos,
        )

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        host_done: list[OffloadKey] = []
        ssd_done: list[OffloadKey] = []
        for key in keys:
            # 같은 키를 두 요청이 서로 다른 티어에서 읽는 중일 수 있다. 티어별 참조 수만
            # 맞으면 되므로 남아 있는 쪽을 하나 줄인다(host 우선).
            if self._load_host.get(key, 0) > 0:
                self._load_host[key] -= 1
                if self._load_host[key] == 0:
                    del self._load_host[key]
                host_done.append(key)
            elif self._load_ssd.get(key, 0) > 0:
                self._load_ssd[key] -= 1
                if self._load_ssd[key] == 0:
                    del self._load_ssd[key]
                ssd_done.append(key)
        if host_done:
            self.host.complete_load(host_done, req_context)
        if ssd_done:
            self.ssd.complete_load(ssd_done, req_context)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        self.host.touch(keys, req_context)
        self.ssd.touch(keys, req_context)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        key_list = list(keys)
        if not key_list:
            return PrepareStoreOutput(
                keys_to_store=[], store_spec=HybridLoadStoreSpec(), evicted_keys=[]
            )

        host_cands = [k for k in key_list if self._wants_host(k)]
        host_out = None
        if host_cands:
            host_out = self.host.prepare_store(host_cands, req_context)
            if host_out is None:
                # host에 자리를 못 만들었다. 전부 SSD로 보낸다(요청 자체를 죽이지 않는다).
                self.n_host_alloc_fail += 1

        host_stored = set(host_out.keys_to_store) if host_out is not None else set()

        if self.write_through:
            ssd_cands = list(key_list)
        else:
            ssd_cands = [k for k in key_list if k not in host_stored]
        ssd_out = self.ssd.prepare_store(ssd_cands, req_context) if ssd_cands else None
        ssd_stored = set(ssd_out.keys_to_store) if ssd_out is not None else set()

        # keys_to_store는 반드시 입력 keys 순서를 지켜야 한다. 스케줄러가 이 순서로
        # GPU 블록 목록(src_spec)을 만든다.
        keys_to_store = [k for k in key_list if k in host_stored or k in ssd_stored]
        cpu_pos = [i for i, k in enumerate(keys_to_store) if k in host_stored]
        file_pos = [i for i, k in enumerate(keys_to_store) if k in ssd_stored]

        self._store_host.update(host_stored)
        self._store_ssd.update(ssd_stored)
        self._host_keys.update(host_stored)
        self._ssd_keys.update(ssd_stored)
        self.n_store_host += len(host_stored)
        self.n_store_ssd += len(ssd_stored)

        evicted: list[OffloadKey] = []
        if host_out is not None and host_out.evicted_keys:
            self.n_evicted_host += len(host_out.evicted_keys)
            for k in host_out.evicted_keys:
                self._host_keys.discard(k)
                if k not in self._ssd_keys:
                    evicted.append(k)
        if ssd_out is not None and ssd_out.evicted_keys:
            self.n_evicted_ssd += len(ssd_out.evicted_keys)
            for k in ssd_out.evicted_keys:
                self._ssd_keys.discard(k)
                if k not in self._host_keys:
                    evicted.append(k)
        self.n_lost += len(evicted)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=HybridLoadStoreSpec(
                n_keys=len(keys_to_store),
                cpu=host_out.store_spec if host_stored else None,
                file=ssd_out.store_spec if ssd_stored else None,
                cpu_pos=cpu_pos,
                file_pos=file_pos,
            ),
            evicted_keys=evicted,
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        host_done = [k for k in keys if k in self._store_host]
        ssd_done = [k for k in keys if k in self._store_ssd]
        for k in host_done:
            self._store_host.discard(k)
            if not success:
                self._host_keys.discard(k)
        for k in ssd_done:
            self._store_ssd.discard(k)
            if not success:
                self._ssd_keys.discard(k)
        if host_done:
            self.host.complete_store(host_done, req_context, success)
        if ssd_done:
            self.ssd.complete_store(ssd_done, req_context, success)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self.host.on_request_finished(req_context)
        self.ssd.on_request_finished(req_context)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self.host.on_schedule_end(context)
        self.ssd.on_schedule_end(context)

    def has_pending_work(self) -> bool:
        return self.host.has_pending_work() or self.ssd.has_pending_work()

    def take_events(self) -> Iterable[OffloadingEvent]:
        yield from self.host.take_events()
        yield from self.ssd.take_events()

    def reset_cache(self) -> None:
        self.host.reset_cache()
        self.ssd.reset_cache()
        self._host_keys.clear()
        self._ssd_keys.clear()
        self._store_host.clear()
        self._store_ssd.clear()
        self._load_host.clear()
        self._load_ssd.clear()

    def get_stats(self):
        # Prometheus 계열 지표는 host 티어 것만 낸다(SSD 쪽은 stats()로 본다).
        return self.host.get_stats()

    def shutdown(self) -> None:
        self.host.shutdown()
        self.ssd.shutdown()

    # --- 러너용 요약 ---

    def stats(self) -> dict:
        host_total = self.host._num_blocks
        host_used = self.host._num_allocated_blocks - len(self.host._free_list)
        ssd = self.ssd.stats()
        return dict(
            placement=self.placement,
            host_share=self.host_share,
            write_through=self.write_through,
            host_blocks_used=host_used,
            host_blocks_total=host_total,
            ssd_files=ssd["files"],
            ssd_gib=ssd["total_gib"],
            hits_host=self.n_hit_host,
            hits_ssd=self.n_hit_ssd,
            misses=self.n_miss,
            pending_host=self.n_pending_host,
            pending_ssd=self.n_pending_ssd,
            stores_host=self.n_store_host,
            stores_ssd=self.n_store_ssd,
            demotions=0,  # write-back 미구현
            evictions_host=self.n_evicted_host,
            evictions_ssd=self.n_evicted_ssd,
            lost_keys=self.n_lost,
            host_alloc_fail=self.n_host_alloc_fail,
            ssd_manager=ssd,
        )
