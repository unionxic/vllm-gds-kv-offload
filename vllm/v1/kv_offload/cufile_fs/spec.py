# SPDX-License-Identifier: Apache-2.0
"""CuFileFsSpec: GPU KV 블록을 cuFile로 파일에 직접 저장·로드하는 오프로딩 스펙(in-tree).
데이터 이동은 native 확장(csrc/kv_offload/cufile_fs.cpp)이 하고, 여기서는 스케줄러 쪽 조회와
워커 쪽 job 분할만 한다. kv_connector_extra_config 키:
  spec_name: "CuFileFsSpec"
  cufile_fs_root_dir: 캐시 디렉터리(단일 루트)
  cufile_fs_root_dirs: 캐시 디렉터리 여러 개. [{"dir": str, "weight": int>0, "capacity_gb": float}, ...]
      블록 해시로 루트 하나를 결정적으로 고른다(가중치 비례). cufile_fs_root_dir 대신 씀.
      capacity_gb(0=무제한)를 준 루트가 차면 자리가 남은 루트들 사이에서 다시 고른다(spill).
      cufile_fs/multi_root.py 참고
  cufile_fs_register_tensors: KV 텐서를 cuFileBufRegister(BAR1 안에 들어갈 때만 성공). 기본 false
  cufile_fs_read_threads / cufile_fs_write_threads: 기본 4
  cufile_fs_admission: all | never | profile | seen_twice
  cufile_fs_capacity_gb: SSD 파일 총량 상한(GiB, 0=무제한), cufile_fs_policy: lru | lfu
  cufile_fs_store_window: "any"(기본, 지금까지의 동작) | "host"
      "host"면 가중치를 SSD에서 스트리밍하는 동안(가중치 오프로더의 SSD 창) KV 쓰기를 멈춘다.
  cufile_fs_store_window_max_s: 쓰기를 연속으로 멈춰 둘 수 있는 상한(초). 기본 10
  cufile_fs_admission: "all"(기본) | "never" | "profile" — 어떤 블록을 저장할지
  cufile_fs_admission_profile: admission="profile"일 때 읽을 JSON 경로
      {"hashes": {"<block hash hex>": expected_reuse, ...}, "min_reuse": 1}
  cufile_fs_trace_keys: 키 단위 store enqueue·commit·lookup 시각을 jsonl로 남김. 기본 false
  cufile_fs_trace_path: 그 jsonl 경로. 없으면 첫 루트 안의 keytrace.jsonl
  cufile_fs_pending_wait: 쓰기 중인 키를 만난 lookup이 남은 쓰기 시간 < 재계산 시간일 때
      HIT_PENDING을 돌려 스케줄러가 커밋을 기다리게 함. 기본 false(기다리지 않고 miss)
  cufile_fs_pending_wait_max_s: 한 (요청, 키)가 기다릴 수 있는 상한(초). 기본 5
  cufile_fs_recompute_s_per_token: 토큰 하나 prefill 비용(초)의 초기값. 0이면 러너가
      set_recompute_cost()로 넣어 줄 때까지 기다리지 않음
  block_size: GPU 블록(16)의 배수. blocks_per_chunk = block_size / 16
"""
import json
import os
import threading
import time
from collections.abc import Collection
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    TransferResult,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cufile_fs.multi_root import (
    MultiRootFileMapper,
    parse_root_dirs,
)
from vllm.v1.kv_offload.file_mapper import FileMapper

logger = init_logger(__name__)
ALIGN = 4096
LAST_WORKER = None  # in-process 계측용
LAST_MANAGER = None


@dataclass
class FileLoadStoreSpec(LoadStoreSpec):
    paths: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"FileLoadStoreSpec(n={len(self.paths)})"


class _AdmissionFilter:
    """어떤 블록을 SSD에 올릴지 고른다(admission).

    해시 문자열은 FileMapper.get_file_name과 같은 get_offload_block_hash(key).hex()라,
    프로파일 작성기가 파일 이름만 보고도 같은 키를 만들 수 있다.
    키마다 로그를 남기면 스텝당 수백 줄이 되므로 LOG_EVERY 건마다 한 줄로 요약한다.
    """

    LOG_EVERY = 100

    def __init__(self, policy: str, profile_path: str | None = None):
        self.policy = policy
        self.hashes: dict[str, int] = {}
        self.min_reuse = 1
        if policy == "profile":
            if not profile_path:
                raise ValueError("cufile_fs_admission=profile requires cufile_fs_admission_profile")
            with open(profile_path) as f:
                prof = json.load(f)
            self.hashes = {str(h): int(v) for h, v in (prof.get("hashes") or {}).items()}
            self.min_reuse = int(prof.get("min_reuse", 1))
            logger.info("CuFileFs admission=profile: %d hashes from %s (min_reuse=%d)",
                        len(self.hashes), profile_path, self.min_reuse)
        elif policy == "seen_twice":
            # 온라인 규칙(04 실험): 같은 블록 해시가 두 번째 miss일 때부터 저장. lookup miss 횟수를 센다.
            self.miss_count: dict[str, int] = {}
        elif policy != "never":
            raise ValueError(f"unknown cufile_fs_admission: {policy}")
        self.n_admit = 0
        self.n_reject = 0
        self._logged = 0

    def note_miss(self, key: OffloadKey) -> None:
        return  # lookup은 첫 miss에서 멈추므로 여기서 세지 않는다(filter에서 제시 횟수로 셈)

    def _admit(self, key: OffloadKey) -> bool:
        if self.policy == "never":
            return False
        if self.policy == "seen_twice":
            # 저장 후보로 제시된 횟수 = 그 블록이 GPU에서 계산된 횟수(GPU 프리픽스 캐시 적중 포함). 두 번째부터 저장.
            h = get_offload_block_hash(key).hex()
            n = self.miss_count.get(h, 0) + 1
            self.miss_count[h] = n
            return n >= 2
        return self.hashes.get(get_offload_block_hash(key).hex(), -1) >= self.min_reuse

    def filter(self, keys: list[OffloadKey]) -> list[OffloadKey]:
        out = [k for k in keys if self._admit(k)]
        self.n_admit += len(out)
        self.n_reject += len(keys) - len(out)
        total = self.n_admit + self.n_reject
        if total - self._logged >= self.LOG_EVERY:
            self._logged = total
            logger.info("CuFileFs admission(%s): %d/%d keys admitted (%.1f%%)",
                        self.policy, self.n_admit, total, 100.0 * self.n_admit / max(total, 1))
        return out


class _StoreWindow:
    """가중치가 SSD에서 스트리밍되는 동안(SSD 창) KV 전송을 멈춘다.

    kind="write"면 쓰기 풀, kind="read"면 읽기 풀을 멈춘다. 읽기 쪽은
    split-source KV의 tail 적재를 SSD 창 밖으로 미루는 용도다(가중치와 KV가
    같은 NVMe 채널을 두고 다투지 않게).

    미러링 지점이 두 개다.
      - get_finished(): 스케줄러가 매 스텝 부른다. GIL 밖 비용이 0에 가깝고 스텝 경계에서
        확실히 한 번 맞춰진다. 다만 SSD 창은 층당 0.2~0.6 s로 스텝보다 짧게 여러 번 바뀌어서
        스텝 경계만으로는 창 하나를 통째로 놓친다.
      - 데몬 스레드: 그래서 poll_s(기본 2 ms) 주기로 같은 미러링을 돌린다. 하는 일은
        Event.is_set() 한 번 + 값이 바뀔 때만 native 호출이라 사실상 idle이다.
    둘 다 sync()를 부르고 sync()는 멱등이라 순서가 섞여도 안전하다.

    안전장치: 창이 풀리지 않아 max_s보다 오래 멈춰 있으면 창과 무관하게 재개하고(한 번만 경고),
    창이 한 번 host로 돌아올 때까지 다시 멈추지 않는다.
    """

    def __init__(self, native, max_s: float, poll_s: float = 0.002, kind: str = "write"):
        from vllm.model_executor.offloader.base import get_offloader
        assert kind in ("write", "read")
        self.native = native
        self.kind = kind
        self._pause = native.pause_writes if kind == "write" else native.pause_reads
        self._resume = native.resume_writes if kind == "write" else native.resume_reads
        self._get_offloader = get_offloader  # 2 ms 폴링이라 매번 import 하지 않는다
        self.max_s = max_s
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._paused = False
        self._paused_since = 0.0
        self._forced = False  # max_s로 강제 재개한 상태(창이 풀릴 때까지 유지)
        self._warned = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cufile_fs_{kind}_window", daemon=True)
        self._thread.start()

    def _ssd_window(self) -> bool:
        return self._get_offloader().is_ssd_window()

    def sync(self) -> None:
        want = self._ssd_window()
        with self._lock:
            if not want:
                self._forced = False
            elif self._forced:
                want = False
            elif self._paused and time.monotonic() - self._paused_since > self.max_s:
                want = False
                self._forced = True
                if not self._warned:
                    self._warned = True
                    logger.warning("CuFileFs %s window: %ss paused > %.1fs, resuming anyway "
                                   "(weight SSD window stuck?)", self.kind, self.kind, self.max_s)
            if want == self._paused:
                return
            if want:
                self._pause()
                self._paused = True
                self._paused_since = time.monotonic()
            else:
                self._resume()
                self._paused = False

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.sync()
            except Exception:  # 미러링 실패로 IO 스레드를 죽이지 않는다
                logger.exception("CuFileFs %s window: sync failed", self.kind)
                return

    def close(self) -> None:
        """멈춘 전송을 반드시 풀고(안 그러면 shutdown의 join이 막힌다) 스레드를 접는다."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        with self._lock:
            if self._paused:
                self._resume()
                self._paused = False


@dataclass
class _Entry:
    size: int = 0
    last_ns: int = 0
    hits: int = 0


class _KeyTracer:
    """키 하나 단위로 store enqueue·commit과 lookup 시각을 jsonl로 남긴다.

    write-behind(쓰기가 끝나기 전에 재사용 요청이 도착) 때문에 생긴 miss를 런 뒤에
    가려내려는 계측이다. cufile_fs_trace_keys=true일 때만 만들어지고 기본은 꺼져 있다.
    시각은 time.monotonic()이라 러너(run_obs)의 submit_mono와 같은 기준이며, 한 줄이
    한 사건이다.
      {"t": mono, "ev": "enq"|"cmt"|"lk"|"lkh", "k": key hex, "r": request id, ...}
      enq  prepare_store가 저장 대상으로 잡은 시각(쓰기 작업 큐에 들어간 시각)
      cmt  complete_store가 성공 보고를 받아 파일이 읽을 수 있게 된 시각
      lk   파일 티어 lookup 결과(hit|miss|pend)
      lkh  상위 매니저(HybridManager) lookup 결과. host 티어까지 포함한 최종 판정
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._f = open(self.path, "a", buffering=1 << 16)
        self._lock = threading.Lock()
        self.n_rows = 0
        logger.info("CuFileFs key trace: %s", self.path)

    def emit(self, ev: str, key: OffloadKey, req_context: ReqContext | None, **kw) -> None:
        row = dict(t=round(time.monotonic(), 6), ev=ev, k=bytes(key).hex(),
                   r=None if req_context is None else req_context.req_id, **kw)
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with self._lock:
            self._f.write(line)
            self.n_rows += 1

    def flush(self) -> None:
        with self._lock:
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()
                self._f.close()


class CuFileFsManager(OffloadingManager):
    """파일 존재 = 적중. store는 tmp→rename이라 부분 파일이 보이지 않는다.

    용량 상한(capacity_bytes > 0)이 있으면 LMCache의 캐시 정책과 같은 규칙으로 SSD 파일을 지운다.
      lru: 마지막 접근(lookup 적중, touch, 저장)이 오래된 것부터
      lfu: 적중 횟수가 적은 것부터, 같으면 lru
    적재 중(prepare_load ~ complete_load)이거나 저장 중(pending)인 키는 지우지 않는다.
    """

    def __init__(self, mapper: FileMapper, admission: _AdmissionFilter | None = None,
                 capacity_bytes: int = 0, policy: str = "lru",
                 tracer: "_KeyTracer | None" = None,
                 pending_wait: bool = False, pending_wait_max_s: float = 5.0,
                 tokens_per_chunk: int = 0, recompute_s_per_token: float = 0.0):
        self.mapper = mapper
        self.tracer = tracer  # None이면 키 단위 계측 없음(기본)
        self._pending: set[OffloadKey] = set()
        # 루트별 용량 spill(MultiRootFileMapper.has_capacity). 키가 놓인 루트를 기억하고
        # 루트별 완료 바이트·대기 청크 수로 자리를 판단한다. 단일 루트/용량 없음이면 안 쓴다.
        self._spill = bool(getattr(mapper, "has_capacity", False))
        self._n_roots = len(getattr(mapper, "roots", [None]))
        self._root_of: dict[OffloadKey, int] = {}
        self._root_bytes: list[int] = [0] * self._n_roots
        self._root_pending: list[int] = [0] * self._n_roots
        self._root_files: list[int] = [0] * self._n_roots
        self.n_spill_refused = 0  # 모든 루트가 차서 저장을 거른 키 수
        self.admission = admission  # None = "all"(전부 저장)
        self.capacity_bytes = int(capacity_bytes)
        self.policy = policy
        if policy not in ("lru", "lfu"):
            raise ValueError(f"unknown cufile_fs_policy: {policy}")
        self._entries: dict[OffloadKey, _Entry] = {}
        self._loading: dict[OffloadKey, int] = {}
        self.chunk_bytes = 0  # 첫 완료 파일 크기로 확정. 그 전의 대기분은 예약하지 못함
        self.total_bytes = 0
        self.n_evicted = 0
        self.bytes_evicted = 0
        self.n_refused = 0
        self.n_hit = 0
        self.n_miss = 0
        # --- 쓰기 중인 키를 기다릴지 고르는 정책(pending_wait). 기본 꺼짐 ---
        self.pending_wait = bool(pending_wait)
        self.pending_wait_max_s = float(pending_wait_max_s)
        self.tokens_per_chunk = int(tokens_per_chunk)
        # 토큰 하나 재계산(prefill) 비용(초). 0이면 아직 관측값이 없다는 뜻이고 그때는 기다리지 않는다.
        # 러너가 set_recompute_cost()로 매 step 갱신한다.
        self.recompute_s_per_token = float(recompute_s_per_token)
        self._enq_t: dict[OffloadKey, float] = {}  # 키가 쓰기 큐에 들어간 시각
        self._enq_q: dict[OffloadKey, int] = {}  # 들어갈 때 앞에 있던(자기 포함) 청크 수
        # 청크 하나를 쓰는 데 드는 시간(초)의 EMA. 커밋된 키마다 (커밋−투입)/투입 당시 큐 깊이로 샘플을 만든다.
        # 쓰기 스레드가 여럿이므로 이 값은 "큐 한 칸이 줄어드는 데 걸린 시간"이고 병렬도가 이미 반영돼 있다.
        self.chunk_service_s = 0.0
        self._wait_since: dict[OffloadKey, dict[str, float]] = {}  # 키 -> 요청 -> 처음 기다리기 시작한 시각
        self._wait_spent: dict[OffloadKey, set[str]] = {}  # 상한까지 써 버린 (키, 요청). 다시 기다리지 않는다
        self.n_wait_calls = 0  # 기다리라고 답한 lookup 수
        self.n_wait_keys = 0  # 기다리기 시작한 (요청, 키) 쌍 수
        self.n_wait_timeout = 0  # 상한까지 기다렸는데 커밋이 안 온 쌍 수
        self.n_wait_resolved = 0  # 기다린 끝에 적중으로 바뀐 쌍 수
        self.wait_s_total = 0.0  # 끝난(적중·상한) 기다림의 시간 합
        self.n_wait_declined = 0  # 쓰기 중이지만 재계산이 더 싸서 miss로 답한 lookup 수
        self.n_wait_unknown = 0  # 재계산 비용을 아직 몰라 miss로 답한 lookup 수
        global LAST_MANAGER
        LAST_MANAGER = self
        if self.capacity_bytes:
            logger.info("CuFileFs manager: capacity %.1f GiB, policy %s", self.capacity_bytes / 2**30, policy)
        if self.pending_wait:
            logger.info("CuFileFs pending_wait: on (max %.1fs, tokens/chunk %d)",
                        self.pending_wait_max_s, self.tokens_per_chunk)

    def set_recompute_cost(self, s_per_token: float) -> None:
        """토큰 하나 prefill 비용(초)의 최신 추정치. 러너가 step마다 넣어 준다."""
        if s_per_token > 0:
            self.recompute_s_per_token = float(s_per_token)

    def _remaining_store_s(self, key: OffloadKey, now: float) -> float:
        """이 키의 쓰기가 끝나기까지 남은 시간의 추정(초). 모르면 -1."""
        if self.chunk_service_s <= 0:
            return -1.0
        q = self._enq_q.get(key)
        t0 = self._enq_t.get(key)
        if q is None or t0 is None:
            return -1.0
        return max(0.0, self.chunk_service_s * q - (now - t0))

    def _path(self, key: OffloadKey) -> str:
        if self._spill:
            i = self._root_of.get(key)
            if i is not None:
                return self.mapper.file_name(key, i)
        return self.mapper.get_file_name(key)

    def _root_room(self, i: int) -> bool:
        cap = self.mapper.capacities[i]
        if cap <= 0:
            return True
        return self._root_bytes[i] + (self._root_pending[i] + 1) * self.chunk_bytes <= cap

    def _place(self, keys: list[OffloadKey]) -> list[OffloadKey]:
        """spill 배치. 키마다 자리가 있는 루트를 고르고 _root_of에 적는다. 전부 찬 키는 거른다."""
        out = []
        for k in keys:
            allowed = [self._root_room(i) for i in range(self._n_roots)]
            if not any(allowed):
                self.n_spill_refused += 1
                continue
            i = self.mapper.root_index(k, allowed)
            h = self.mapper.hash_root(k)
            self.mapper.mapped[i] += 1
            if i != h:
                self.mapper.spilled[i] += 1
            self._root_of[k] = i
            self._root_pending[i] += 1
            out.append(k)
        return out

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        e = self._entries.get(key)
        if e is not None or os.path.exists(self._path(key)):
            if e is None:
                e = self._entries[key] = _Entry(size=os.path.getsize(self._path(key)))
                self.total_bytes += e.size
            e.last_ns = time.monotonic_ns(); e.hits += 1
            self.n_hit += 1
            if self._wait_since:
                self._end_wait(key, req_context, resolved=True)
            if self.tracer is not None:
                self.tracer.emit("lk", key, req_context, res="hit")
            return LookupResult.HIT
        if self.pending_wait and key in self._pending:
            res = self._pending_decision(key, req_context)
            if res is not None:
                return res
        self.n_miss += 1
        if self.admission is not None:
            self.admission.note_miss(key)
        if self.tracer is not None:
            self.tracer.emit("lk", key, req_context, res="miss",
                             pend=key in self._pending)
        return LookupResult.MISS

    def _end_wait(self, key: OffloadKey, req_context: ReqContext, resolved: bool) -> None:
        """이 (요청, 키)의 기다림을 끝내고 시간을 집계한다."""
        per_req = self._wait_since.get(key)
        if not per_req:
            return
        t0 = per_req.pop(req_context.req_id, None)
        if not per_req:
            self._wait_since.pop(key, None)
        if t0 is None:
            return
        self.wait_s_total += time.monotonic() - t0
        if resolved:
            self.n_wait_resolved += 1

    def _pending_decision(self, key: OffloadKey, req_context: ReqContext) -> LookupResult | None:
        """쓰기 중인 키를 만났을 때 기다릴지 고른다. 기다리면 HIT_PENDING, 아니면 None(=miss로 진행).

        남은 쓰기 시간 추정 < 재계산 시간 추정이면 기다린다. 남은 쓰기 시간은 투입 당시 큐 깊이와
        청크 하나의 관측 처리 시간으로, 재계산 시간은 러너가 넣어 준 토큰당 prefill 비용과
        청크 하나의 토큰 수로 잡는다. 청크 하나분만 세므로 이득을 낮게 잡는 쪽이다(그 뒤 프리픽스는
        어차피 같이 못 쓴다). 상한(pending_wait_max_s)을 넘으면 기다림을 접고 miss로 답한다.
        """
        now = time.monotonic()
        rid = req_context.req_id
        if rid in self._wait_spent.get(key, ()):  # 이미 상한까지 기다려 본 쌍
            return None
        per_req = self._wait_since.get(key)
        t0 = per_req.get(rid) if per_req else None
        if t0 is not None and now - t0 > self.pending_wait_max_s:
            self.n_wait_timeout += 1
            self._wait_spent.setdefault(key, set()).add(rid)
            self._end_wait(key, req_context, resolved=False)
            if self.tracer is not None:
                self.tracer.emit("lk", key, req_context, res="miss", pend=True, why="timeout")
            return None

        recompute_s = self.recompute_s_per_token * self.tokens_per_chunk
        if recompute_s <= 0:
            self.n_wait_unknown += 1
            return None
        remain_s = self._remaining_store_s(key, now)
        if remain_s < 0 or remain_s >= recompute_s:
            self.n_wait_declined += 1
            if self.tracer is not None:
                self.tracer.emit("lk", key, req_context, res="miss", pend=True,
                                 why="declined", remain=round(remain_s, 4), rec=round(recompute_s, 4))
            return None

        if t0 is None:
            self._wait_since.setdefault(key, {})[rid] = now
            self.n_wait_keys += 1
        self.n_wait_calls += 1
        if self.tracer is not None:
            self.tracer.emit("lk", key, req_context, res="pend",
                             remain=round(remain_s, 4), rec=round(recompute_s, 4))
        return LookupResult.HIT_PENDING

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        now = time.monotonic_ns()
        for k in keys:
            e = self._entries.get(k)
            if e is not None:
                e.last_ns = now

    def key_tiers(self, keys, req_context: ReqContext) -> list:
        return ["ssd"] * len(list(keys))

    def prepare_load(self, keys: Collection[OffloadKey], req_context: ReqContext) -> LoadStoreSpec:
        for k in keys:
            self._loading[k] = self._loading.get(k, 0) + 1
        return FileLoadStoreSpec([self._path(k) for k in keys])

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        for k in keys:
            n = self._loading.get(k, 0) - 1
            if n <= 0:
                self._loading.pop(k, None)
            else:
                self._loading[k] = n

    def _committed(self) -> int:
        return self.total_bytes + len(self._pending) * self.chunk_bytes

    def _evict(self, need: int) -> list[OffloadKey]:
        """완료분 + 대기 예약분 + need 가 상한을 넘으면 정책 순서로 파일을 지워 자리를 만든다."""
        if not self.capacity_bytes or self._committed() + need <= self.capacity_bytes:
            return []
        cands = [(k, e) for k, e in self._entries.items() if k not in self._loading and k not in self._pending]
        if self.policy == "lfu":
            cands.sort(key=lambda ke: (ke[1].hits, ke[1].last_ns))
        else:
            cands.sort(key=lambda ke: ke[1].last_ns)
        out = []
        for k, e in cands:
            if self._committed() + need <= self.capacity_bytes:
                break
            try:
                os.remove(self._path(k))
            except FileNotFoundError:
                pass
            self._entries.pop(k, None)
            self.total_bytes -= e.size
            if self._spill:
                i = self._root_of.pop(k, None)
                if i is not None:
                    self._root_bytes[i] -= e.size; self._root_files[i] -= 1
            self.n_evicted += 1; self.bytes_evicted += e.size
            out.append(k)
        if out:
            logger.info("CuFileFs evict(%s): %d files, total now %.2f GiB (evicted so far %d, %.2f GiB)",
                        self.policy, len(out), self.total_bytes / 2**30, self.n_evicted, self.bytes_evicted / 2**30)
        return out

    def prepare_store(self, keys: Collection[OffloadKey], req_context: ReqContext) -> PrepareStoreOutput | None:
        to_store = [k for k in keys if k not in self._pending and k not in self._entries and not os.path.exists(self._path(k))]
        if self.admission is not None:
            to_store = self.admission.filter(to_store)
        evicted = []
        if self.capacity_bytes and to_store:
            est = self.chunk_bytes * len(to_store)
            evicted = self._evict(est)
            if self._committed() + est > self.capacity_bytes:  # 자리를 못 만들면 들어가는 만큼만 저장
                room = self.capacity_bytes - self._committed()
                n_ok = max(0, room // self.chunk_bytes) if self.chunk_bytes else len(to_store)
                self.n_refused += len(to_store) - n_ok
                to_store = to_store[:n_ok]
        if self._spill and to_store:
            to_store = self._place(to_store)
        self._pending.update(to_store)
        if to_store and (self.pending_wait or self.tracer is not None):
            # 쓰기 큐 깊이는 "커밋을 기다리는 키 수"로 본다(워커의 청크 큐와 같은 단위).
            now_s = time.monotonic()
            depth = len(self._pending)
            for k in to_store:
                if self.pending_wait:
                    self._enq_t[k] = now_s
                    self._enq_q[k] = depth
                if self.tracer is not None:
                    self.tracer.emit("enq", k, req_context, q=depth)
        return PrepareStoreOutput(keys_to_store=to_store, store_spec=FileLoadStoreSpec([self._path(k) for k in to_store]), evicted_keys=evicted)

    def complete_store(self, keys: Collection[OffloadKey], req_context: ReqContext, success: bool = True):
        now = time.monotonic_ns()
        now_s = time.monotonic()
        for k in keys:
            self._pending.discard(k)
            if self._spill:
                ri = self._root_of.get(k)
                if ri is not None:
                    self._root_pending[ri] = max(0, self._root_pending[ri] - 1)
                    if not success:
                        self._root_of.pop(k, None)
            if self.pending_wait:
                t0 = self._enq_t.pop(k, None)
                q = self._enq_q.pop(k, None)
                if t0 is not None and q:
                    # 청크 한 칸이 줄어드는 데 걸린 시간의 EMA(alpha 0.2). 병렬 쓰기 스레드가 이미 반영된 값.
                    sample = (now_s - t0) / q
                    self.chunk_service_s = sample if self.chunk_service_s <= 0 else 0.2 * sample + 0.8 * self.chunk_service_s
                # 커밋 시점이 곧 기다림이 끝난 시점이다(다음 lookup은 적중).
                for t_wait in (self._wait_since.pop(k, None) or {}).values():
                    self.n_wait_resolved += 1
                    self.wait_s_total += now_s - t_wait
                self._wait_spent.pop(k, None)
            if self.tracer is not None:
                self.tracer.emit("cmt", k, req_context, ok=bool(success))
            if success and k not in self._entries:
                try:
                    sz = os.path.getsize(self._path(k))
                except OSError:
                    continue
                self._entries[k] = _Entry(size=sz, last_ns=now, hits=0)
                self.total_bytes += sz
                if self._spill:
                    ri = self._root_of.get(k)
                    if ri is not None:
                        self._root_bytes[ri] += sz; self._root_files[ri] += 1
                if not self.chunk_bytes:
                    self.chunk_bytes = sz

    def shutdown(self) -> None:
        if self.tracer is not None:
            self.tracer.close()

    def stats(self) -> dict:
        if self.tracer is not None:
            self.tracer.flush()  # result.json을 쓰는 시점에 트레이스가 디스크에 다 있도록
        # files는 complete_store가 처리된 파일 수. 마지막 엔진 step 뒤에 끝난 쓰기는 pending에 남아 디스크 파일 수보다 적을 수 있음.
        return dict(files=len(self._entries), pending=len(self._pending), total_gib=round(self.total_bytes / 2**30, 3), capacity_gib=round(self.capacity_bytes / 2**30, 3),
                    policy=self.policy, refused=self.n_refused, chunk_bytes=self.chunk_bytes, evicted=self.n_evicted, evicted_gib=round(self.bytes_evicted / 2**30, 3),
                    lookup_hit=self.n_hit, lookup_miss=self.n_miss,
                    trace_path=None if self.tracer is None else self.tracer.path,
                    trace_rows=None if self.tracer is None else self.tracer.n_rows,
                    pending_wait=None if not self.pending_wait else dict(
                        max_s=self.pending_wait_max_s, tokens_per_chunk=self.tokens_per_chunk,
                        waits=self.n_wait_keys, wait_calls=self.n_wait_calls,
                        resolved=self.n_wait_resolved, timeouts=self.n_wait_timeout,
                        declined=self.n_wait_declined, unknown_cost=self.n_wait_unknown,
                        wait_s_total=round(self.wait_s_total, 3),
                        wait_s_mean=round(self.wait_s_total / max(1, self.n_wait_resolved + self.n_wait_timeout), 4),
                        chunk_service_s=round(self.chunk_service_s, 5),
                        recompute_s_per_token=round(self.recompute_s_per_token, 8)),
                    admission=None if self.admission is None else dict(policy=self.admission.policy, admit=self.admission.n_admit, reject=self.admission.n_reject),
                    # roots는 다중 루트일 때만. get_file_name 호출 수라 파일 수가 아니라 매핑 횟수다.
                    roots=self._roots_stats())

    def _roots_stats(self):
        if not hasattr(self.mapper, "roots_stats"):
            return None
        st = self.mapper.roots_stats()
        if self._spill:
            for i, r in enumerate(st):
                r.update(files=self._root_files[i], bytes_gib=round(self._root_bytes[i] / 2**30, 3),
                         pending=self._root_pending[i])
            st.append(dict(spill_refused=self.n_spill_refused))
        return st

    def reset_cache(self) -> None:
        self._pending.clear()
        self._root_pending = [0] * self._n_roots
        self._enq_t.clear(); self._enq_q.clear(); self._wait_since.clear(); self._wait_spent.clear()


class CuFileFsWorker(OffloadingWorker):
    def __init__(self, kv_caches: CanonicalKVCaches, blocks_per_chunk: int, n_read: int, n_write: int, register_tensors: bool,
                 store_window: str = "any", store_window_max_s: float = 10.0, load_window: str = "any"):
        assert len(kv_caches.group_data_refs) == 1, "CuFileFs: single KV cache group only"
        from vllm.v1.kv_offload.cufile_fs.native import load
        self.bpc = blocks_per_chunk
        base, tbytes, pages = [], [], []
        for ct in kv_caches.tensors:
            page = ct.page_size_bytes
            assert page % ALIGN == 0, f"CuFileFs: page_size_bytes {page} not {ALIGN}-aligned"
            t = ct.tensor
            base.append(t.data_ptr()); tbytes.append(t.numel() * t.element_size()); pages.append(page)
        self.native = load().CuFileFs(base, tbytes, pages, self.bpc, n_read, n_write, register_tensors)
        self.chunk_bytes = self.native.chunk_bytes
        self._pending: set[int] = set()
        self._events: dict[int, torch.cuda.Event] = {}
        global LAST_WORKER
        LAST_WORKER = self
        logger.info("CuFileFs worker: %d tensors, bpc=%d, chunk=%d bytes, registered_tensors=%d%s",
                    len(base), self.bpc, self.chunk_bytes, self.native.registered,
                    "" if register_tensors and self.native.registered else f" (register_err={self.native.register_err})" if register_tensors else "")
        self.window: _StoreWindow | None = None
        if store_window == "host":
            self.window = _StoreWindow(self.native, store_window_max_s, kind="write")
            logger.info("CuFileFs store window=host: KV writes pause while weights stream from SSD "
                        "(max %.1fs)", store_window_max_s)
        elif store_window != "any":
            raise ValueError(f"unknown cufile_fs_store_window: {store_window}")
        # 읽기 창. split-source KV에서 tail 적재가 가중치 SSD 읽기와 채널을 다투지 않게 한다.
        self.load_window: _StoreWindow | None = None
        if load_window == "host":
            self.load_window = _StoreWindow(self.native, store_window_max_s, kind="read")
            logger.info("CuFileFs load window=host: KV reads pause while weights stream from SSD "
                        "(max %.1fs)", store_window_max_s)
        elif load_window != "any":
            raise ValueError(f"unknown cufile_fs_load_window: {load_window}")

    def _partition(self, gpu_spec: GPULoadStoreSpec, file_spec: FileLoadStoreSpec, is_store: bool):
        assert isinstance(gpu_spec, GPULoadStoreSpec) and isinstance(file_spec, FileLoadStoreSpec)
        assert len(gpu_spec.group_sizes) == 1, "CuFileFs: single group only"
        bids = [int(b) for b in gpu_spec.block_ids]
        n_chunks = len(file_spec.paths)
        skip = int(gpu_spec.block_indices[0]) % self.bpc
        if is_store:
            assert skip == 0, "CuFileFs: store must be chunk-aligned"
        assert len(bids) == n_chunks * self.bpc - skip, f"CuFileFs: {len(bids)} blocks vs {n_chunks} chunks (bpc={self.bpc}, skip={skip})"
        paths, blocks, j0s, pos = [], [], [], 0
        for c, path in enumerate(file_spec.paths):
            j0 = skip if c == 0 else 0
            take = self.bpc - j0
            paths.append(path); blocks.append(bids[pos:pos + take]); j0s.append(j0)
            pos += take
        return paths, blocks, j0s

    def submit_store(self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec) -> bool:
        paths, blocks, j0s = self._partition(src_spec, dst_spec, is_store=True)
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
        self._events[job_id] = ev  # native가 cudaEventSynchronize 할 때까지 살아 있어야 함
        self._pending.add(job_id)
        self.native.submit(job_id, True, paths, blocks, j0s, ev.cuda_event)
        return True

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        paths, blocks, j0s = self._partition(dst_spec, src_spec, is_store=False)
        self._pending.add(job_id)
        self.native.submit(job_id, False, paths, blocks, j0s, 0)
        return True

    def get_finished(self) -> list[TransferResult]:
        if self.window is not None:
            self.window.sync()  # 스텝 경계에서 한 번(데몬 스레드가 그 사이를 메운다)
        if self.load_window is not None:
            self.load_window.sync()
        out = []
        for job_id, ok in self.native.get_finished():
            self._pending.discard(job_id); self._events.pop(job_id, None)
            out.append(TransferResult(job_id=job_id, success=ok))
        return out

    def wait(self, job_ids: set[int]) -> None:
        self.native.wait([int(j) for j in job_ids])

    def stats(self) -> dict:
        return self.native.stats()

    def shutdown(self) -> None:
        if self.window is not None:
            self.window.close()  # join 전에 반드시 재개
            self.window = None
        if self.load_window is not None:
            self.load_window.close()
            self.load_window = None
        self.native.shutdown()


class CuFileFsSpec(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        root = self.extra_config.get("cufile_fs_root_dir")
        roots_cfg = self.extra_config.get("cufile_fs_root_dirs")
        # 루트 여러 개를 주면 그쪽이 기준이다(cufile_fs_root_dir는 무시). 하나만 주면 예전과 같다.
        self.root_dirs: list[tuple[str, int]] | None = (
            parse_root_dirs(roots_cfg) if roots_cfg else None
        )
        if self.root_dirs is None and not root:
            raise ValueError("cufile_fs_root_dir or cufile_fs_root_dirs must be set in kv_connector_extra_config")
        # root_dir은 로그·하위 스펙 호환용 대표 경로(다중 루트면 첫 루트).
        self.root_dir = root if self.root_dirs is None else self.root_dirs[0][0]
        self.register_tensors = str(self.extra_config.get("cufile_fs_register_tensors", "false")).lower() in ("1", "true", "yes")
        self.n_read = int(self.extra_config.get("cufile_fs_read_threads", 4))
        self.n_write = int(self.extra_config.get("cufile_fs_write_threads", 4))
        self.store_window = str(self.extra_config.get("cufile_fs_store_window", "any")).lower()
        self.load_window = str(self.extra_config.get("cufile_fs_load_window", "any")).lower()
        self.store_window_max_s = float(self.extra_config.get("cufile_fs_store_window_max_s", 10.0))
        self.admission = str(self.extra_config.get("cufile_fs_admission", "all")).lower()
        self.admission_profile = self.extra_config.get("cufile_fs_admission_profile")
        self.capacity_gb = float(self.extra_config.get("cufile_fs_capacity_gb", 0))  # 0 = 무제한
        self.policy = str(self.extra_config.get("cufile_fs_policy", "lru")).lower()
        # 키 단위 계측(기본 꺼짐). 경로를 주지 않으면 첫 루트 안에 둔다(런 폴더를 주는 쪽을 권장.
        # 캠페인이 런 뒤 KV 루트를 지우므로 루트 안에 두면 같이 사라짐).
        self.trace_keys = str(self.extra_config.get("cufile_fs_trace_keys", "false")).lower() in ("1", "true", "yes")
        self.trace_path = self.extra_config.get("cufile_fs_trace_path")
        # 쓰기 중인 키를 기다릴지 고르는 정책(기본 꺼짐. 켜면 예전 동작과 달라진다)
        self.pending_wait = str(self.extra_config.get("cufile_fs_pending_wait", "false")).lower() in ("1", "true", "yes")
        self.pending_wait_max_s = float(self.extra_config.get("cufile_fs_pending_wait_max_s", 5.0))
        self.recompute_s_per_token = float(self.extra_config.get("cufile_fs_recompute_s_per_token", 0.0))
        self._manager = None
        self._worker = None

    def _mapper(self) -> FileMapper | MultiRootFileMapper:
        if self.root_dirs is not None:
            return MultiRootFileMapper.from_offloading_spec(roots=self.root_dirs, offloading_spec=self,
                                                            blocks_per_file=self.blocks_per_chunk, parallel_agnostic=True)
        return FileMapper.from_offloading_spec(root_dir=self.root_dir, offloading_spec=self,
                                               blocks_per_file=self.blocks_per_chunk, parallel_agnostic=True)

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            mapper = self._mapper()
            for cfg in mapper.get_config_file_paths():  # 루트마다 하나씩
                os.makedirs(os.path.dirname(cfg), exist_ok=True)
                if not os.path.exists(cfg):
                    with open(cfg, "w") as f:
                        json.dump(mapper.get_run_config(), f, indent=2, sort_keys=True)
            adm = None if self.admission == "all" else _AdmissionFilter(self.admission, self.admission_profile)
            tracer = None
            if self.trace_keys:
                tracer = _KeyTracer(self.trace_path or os.path.join(self.root_dir, "keytrace.jsonl"))
            self._manager = CuFileFsManager(mapper, adm, capacity_bytes=int(self.capacity_gb * 2**30), policy=self.policy,
                                            tracer=tracer, pending_wait=self.pending_wait,
                                            pending_wait_max_s=self.pending_wait_max_s,
                                            tokens_per_chunk=self.tokens_per_block[0] * self.blocks_per_chunk,
                                            recompute_s_per_token=self.recompute_s_per_token)
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = CuFileFsWorker(kv_caches, self.blocks_per_chunk, self.n_read, self.n_write, self.register_tensors,
                                          self.store_window, self.store_window_max_s, self.load_window)
        return self._worker
