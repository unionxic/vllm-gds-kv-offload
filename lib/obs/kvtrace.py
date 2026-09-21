# SPDX-License-Identifier: Apache-2.0
"""KV 커넥터 요청·키 단위 계측과 저장 차단(조건 공통). 산출물은 RUN_DIR/kvtrace.jsonl.

포크(vllm) 쪽 클래스 메서드를 감싸기만 하고 동작은 바꾸지 않는다. 예외는 no_store 하나로,
켜 두는 동안 prepare_store가 빈 결과를 돌려준다(강제 적중 조건의 재생 구간).
시각 t는 time.monotonic() 원값이라 requests.jsonl의 submit_mono/first_mono와 바로 잇는다.

행(ev)
  lookup     조회 한 번. req, key, res(HIT|MISS|HIT_PENDING|RETRY)
  store_enq  prepare_store가 받아들인 키. req, host(키 목록), ssd(키 목록)
  store_cmt  complete_store. req, keys, ok
  load_prep  prepare_load. req, n_host, n_ssd(각 키 수), bytes, by_root(루트별 바이트: host 포함)
  mc_load_tier mooncake 묶음 적재의 층별 바이트(VLLM_MOONCAKE_STORE_TIER_LOG=1 필요). req, memory, disk, unknown
  load_done  complete_load. req, dur_s(load_prep부터), keys
  w_submit / w_end   SSD(cuFile) 쓰기 job 하나. job, chunks, bytes, keys / dur_s, ok
  r_submit / r_end   SSD(cuFile) 읽기 job 하나. 같은 필드
  mc_lookup  mooncake 스케줄러 조회. req, hit_tokens, dur_s
  mc_store_enq / mc_store_cmt  mooncake 저장 스레드 큐 투입 / 처리 끝. req, tokens
  mc_put / mc_get              mooncake 묶음 전송 한 번. req, keys, bytes, dur_s

한 요청의 적재 완료 시각은 load_done, 한 키의 커밋 시각은 store_cmt(파일 티어는 w_end)다.
요청별 층·루트 읽기 바이트 합은 KvTrace.req_loads[req]에 쌓이고 run_obs가 requests.jsonl(load_bytes)에 옮긴다.
"""
import json
import os
import threading
import time

_KEYHEX_LEN = 16  # 키 해시 앞 16자리만 남긴다(행 크기 절약, 충돌은 사실상 없음)


def _hexk(key) -> str:
    return bytes(key)[:-4].hex()[:_KEYHEX_LEN]


def _hexpath(path: str) -> str:
    return os.path.basename(path).split(".")[0][:_KEYHEX_LEN]


class KvTrace:
    """jsonl 기록기 + 저장 차단 스위치. 여러 스레드가 같이 쓴다."""

    def __init__(self, path: str, flush_every: int = 256):
        self.path = path
        self._f = open(path, "w", buffering=1 << 20)
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._flush_every = flush_every
        self.no_store = False
        self.phase = ""
        self.n_rows = 0
        self.n_store_blocked = 0
        self.req_loads: dict[str, dict[str, int]] = {}  # req → {층/루트: 바이트}

    def add_load(self, req, by: dict) -> None:
        if req is None:
            return
        with self._lock:
            d = self.req_loads.setdefault(str(req), {})
            for k, v in by.items():
                d[k] = d.get(k, 0) + int(v)

    def loads_of(self, rid: str) -> dict[str, int]:
        """rid와 같거나 rid- 로 시작하는(엔진이 접미사를 붙인) 요청의 적재 바이트 합."""
        out: dict[str, int] = {}
        with self._lock:
            for k, d in self.req_loads.items():
                if k == rid or k.startswith(rid + "-"):
                    for kk, v in d.items():
                        out[kk] = out.get(kk, 0) + v
        return out

    def emit(self, ev: str, **kw) -> None:
        row = dict(ev=ev, t=time.monotonic(), ph=self.phase)
        row.update(kw)
        with self._lock:
            self._buf.append(row)
            self.n_rows += 1
            if len(self._buf) >= self._flush_every:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        self._f.write("".join(json.dumps(r) + "\n" for r in self._buf))
        self._buf.clear()
        self._f.flush()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()
        self._f.close()


# ---- 파일/하이브리드 티어(OffloadingConnector 계열) ----

def _patch_offloading(transport: str, tr: KvTrace) -> None:
    from vllm.v1.kv_offload.base import PrepareStoreOutput
    from vllm.v1.kv_offload.cufile_fs.spec import (
        CuFileFsManager,
        CuFileFsWorker,
        FileLoadStoreSpec,
    )

    if transport == "hybrid":
        from vllm.v1.kv_offload.hybrid.common import HybridLoadStoreSpec
        from vllm.v1.kv_offload.hybrid.manager import HybridManager as MGR

        def _empty():
            return PrepareStoreOutput([], HybridLoadStoreSpec(), [])
    else:
        MGR = CuFileFsManager

        def _empty():
            return PrepareStoreOutput([], FileLoadStoreSpec([]), [])

    rid = lambda ctx: getattr(ctx, "req_id", None)  # noqa: E731

    _lookup = MGR.lookup
    # 스케줄러는 한 요청을 다시 볼 때마다 같은 키를 또 조회한다. 결과가 바뀐 때만 남긴다
    # (MISS→HIT 같은 전이가 봐야 할 전부이고, 전부 남기면 행이 수십 배로 는다).
    _seen: dict[tuple, str] = {}

    def lookup(self, key, req_context):
        r = _lookup(self, key, req_context)
        k = (rid(req_context), _hexk(key))
        if _seen.get(k) != r.name:
            _seen[k] = r.name
            tr.emit("lookup", req=k[0], key=k[1], res=r.name)
        return r

    _prep_store = MGR.prepare_store

    def prepare_store(self, keys, req_context):
        if tr.no_store:
            tr.n_store_blocked += 1
            return _empty()
        out = _prep_store(self, keys, req_context)
        if out is not None and out.keys_to_store:
            kl = list(out.keys_to_store)
            sp = out.store_spec
            host = [_hexk(kl[i]) for i in getattr(sp, "cpu_pos", ())]
            ssd = [_hexk(kl[i]) for i in getattr(sp, "file_pos", range(len(kl)))]
            tr.emit("store_enq", req=rid(req_context), host=host, ssd=ssd)
        return out

    _cmpl_store = MGR.complete_store

    def complete_store(self, keys, req_context, success=True):
        kl = list(keys)
        tr.emit("store_cmt", req=rid(req_context), keys=[_hexk(k) for k in kl], ok=bool(success))
        return _cmpl_store(self, kl, req_context, success)

    _prep_load = MGR.prepare_load
    _load_t0: dict[tuple, float] = {}

    def prepare_load(self, keys, req_context):
        kl = list(keys)
        spec = _prep_load(self, kl, req_context)
        n_host = len(getattr(spec, "cpu_pos", ()))
        n_ssd = len(getattr(spec, "file_pos", kl))
        chunk = _chunk_bytes(self)
        # 루트별 바이트. host 티어는 "host", 파일 티어는 그 키가 놓인 루트 디렉터리(용량 spill이면 _root_of).
        by_root: dict[str, int] = {}
        if n_host:
            by_root["host"] = n_host * chunk
        ssd = getattr(self, "ssd", self)
        mapper = getattr(ssd, "mapper", None)
        root_of = getattr(ssd, "_root_of", {})
        fpos = getattr(spec, "file_pos", None)
        ssd_keys = [kl[i] for i in fpos] if fpos is not None else kl
        for k in ssd_keys:
            if mapper is not None and hasattr(mapper, "hash_root"):
                r = mapper.roots[root_of.get(k, mapper.hash_root(k))]
            else:
                r = getattr(mapper, "root_dir", "file")
            by_root[r] = by_root.get(r, 0) + chunk
        tr.add_load(rid(req_context), by_root)
        tr.emit("load_prep", req=rid(req_context), n_host=n_host, n_ssd=n_ssd,
                bytes=(n_host + n_ssd) * chunk, by_root=by_root, keys=[_hexk(k) for k in kl])
        _load_t0.setdefault(rid(req_context), time.monotonic())
        return spec

    _cmpl_load = MGR.complete_load

    def complete_load(self, keys, req_context):
        kl = list(keys)
        t0 = _load_t0.pop(rid(req_context), None)
        tr.emit("load_done", req=rid(req_context), n_keys=len(kl),
                dur_s=None if t0 is None else round(time.monotonic() - t0, 6))
        return _cmpl_load(self, kl, req_context)

    def _chunk_bytes(mgr) -> int:
        ssd = getattr(mgr, "ssd", mgr)
        cb = getattr(ssd, "chunk_bytes", 0)
        if cb:
            return cb
        import vllm.v1.kv_offload.cufile_fs.spec as _cfs
        w = _cfs.LAST_WORKER
        return getattr(w, "chunk_bytes", 0) if w is not None else 0

    MGR.lookup = lookup
    MGR.prepare_store = prepare_store
    MGR.complete_store = complete_store
    MGR.prepare_load = prepare_load
    MGR.complete_load = complete_load

    # --- SSD(cuFile) 워커 job 단위. 파일 이름이 블록 해시라 job↔키를 그대로 잇는다 ---
    _jobs: dict[int, tuple] = {}
    _ss, _sl, _gf = CuFileFsWorker.submit_store, CuFileFsWorker.submit_load, CuFileFsWorker.get_finished

    def submit_store(self, job_id, src, dst):
        ks = [_hexpath(p) for p in getattr(dst, "paths", ())]
        _jobs[job_id] = ("w", time.monotonic(), len(ks) * self.chunk_bytes)
        tr.emit("w_submit", job=job_id, chunks=len(ks), bytes=len(ks) * self.chunk_bytes, keys=ks)
        return _ss(self, job_id, src, dst)

    def submit_load(self, job_id, src, dst):
        ks = [_hexpath(p) for p in getattr(src, "paths", ())]
        _jobs[job_id] = ("r", time.monotonic(), len(ks) * self.chunk_bytes)
        tr.emit("r_submit", job=job_id, chunks=len(ks), bytes=len(ks) * self.chunk_bytes, keys=ks)
        return _sl(self, job_id, src, dst)

    def get_finished(self):
        out = _gf(self)
        for r in out:
            op, t, nb = _jobs.pop(r.job_id, ("?", time.monotonic(), 0))
            tr.emit(f"{op}_end", job=r.job_id, ok=bool(r.success), bytes=nb,
                    dur_s=round(time.monotonic() - t, 6))
        return out

    CuFileFsWorker.submit_store = submit_store
    CuFileFsWorker.submit_load = submit_load
    CuFileFsWorker.get_finished = get_finished


# ---- Mooncake Store ----

LAST_MC_WORKER = None


def _patch_mooncake(tr: KvTrace) -> None:
    import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler as msched
    import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker as mw
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import ReqMeta

    # 저장 차단: 스케줄러가 put을 결정하는 지점이 skip_save 하나라 여기서 강제한다.
    _frt = ReqMeta.from_request_tracker

    def from_request_tracker(tracker, block_size, load_spec=None, skip_save=False,
                             block_hashes=None, is_last_chunk=None):
        if tr.no_store:
            tr.n_store_blocked += 1
            skip_save = True
        return _frt(tracker, block_size, load_spec, skip_save, block_hashes, is_last_chunk)

    ReqMeta.from_request_tracker = staticmethod(from_request_tracker)

    # 조회(스케줄러 쪽)
    _gm = msched.MooncakeStoreScheduler.get_num_new_matched_tokens

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        t0 = time.monotonic()
        r = _gm(self, request, num_computed_tokens)
        tr.emit("mc_lookup", req=getattr(request, "request_id", None),
                hit_tokens=r[0], dur_s=round(time.monotonic() - t0, 6))
        return r

    msched.MooncakeStoreScheduler.get_num_new_matched_tokens = get_num_new_matched_tokens

    # 전송 스레드. _record_operation은 묶음 단위 시간·바이트를 이미 계산해 준다.
    _tl = threading.local()
    _rec = mw.KVTransferThread._record_operation

    def _record_operation(self, operation, start_time, num_keys, *, num_bytes=0,
                          status="ok", num_failed_keys=0):
        dur = time.perf_counter() - start_time
        acc = getattr(_tl, "acc", None)
        if acc is not None:
            acc[0] += num_bytes
            acc[1] += dur
        if operation in ("save_put", "load_get"):
            tr.emit("mc_put" if operation == "save_put" else "mc_get",
                    req=getattr(_tl, "req", None), keys=num_keys, bytes=num_bytes,
                    dur_s=round(dur, 6), status=status)
        return _rec(self, operation, start_time, num_keys, num_bytes=num_bytes,
                    status=status, num_failed_keys=num_failed_keys)

    mw.KVTransferThread._record_operation = _record_operation

    _add = mw.KVTransferThread.add_request

    def add_request(self, request):
        tr.emit("mc_store_enq", req=getattr(request, "req_id", None),
                tokens=getattr(request, "token_len_chunk", None))
        return _add(self, request)

    mw.KVTransferThread.add_request = add_request

    # 적재는 add_request를 거치지 않고 get_finished가 recv_request_queue에 바로 넣는다.
    # 큐 대기까지 포함한 적재 시작 시각을 남기려면 여기를 감싸야 한다.
    _gf = mw.MooncakeStoreWorker.get_finished

    def worker_get_finished(self, finished_req_ids, meta):
        for rq in meta.requests:
            ls = rq.load_spec
            if ls is not None and ls.can_load:
                tr.emit("mc_load_enq", req=rq.req_id, tokens=ls.kvpool_cached_tokens)
        return _gf(self, finished_req_ids, meta)

    mw.MooncakeStoreWorker.get_finished = worker_get_finished

    # 묶음 적재의 층(memory/disk)별 바이트. 워커가 VLLM_MOONCAKE_STORE_TIER_LOG=1일 때만 이 함수를 부른다
    # (run_obs가 --kv-trace와 함께 환경변수를 켠다). 모듈 전역을 호출 시점에 찾으므로 여기서 바꿔치기 된다.
    _tier_log = mw._log_mooncake_load_tier_summary

    def _log_tier(req_id, batch_keys, load_results, tiers_by_key):
        by = {"memory": 0, "disk": 0, "unknown": 0}
        for i, k in enumerate(batch_keys):
            v = load_results[i] if i < len(load_results) else -1
            if v >= 0:
                t = tiers_by_key.get(k, "unknown")
                by[t if t in by else "unknown"] += int(v)
        tr.add_load(req_id, {"mc_" + k: v for k, v in by.items()})
        tr.emit("mc_load_tier", req=req_id, **by)
        return _tier_log(req_id, batch_keys, load_results, tiers_by_key)

    mw._log_mooncake_load_tier_summary = _log_tier

    def _wrap_handle(cls, ev_end):
        _h = cls._handle_request

        def _handle_request(self, req_meta):
            _tl.req = getattr(req_meta, "req_id", None)
            _tl.acc = [0, 0.0]
            t0 = time.monotonic()
            try:
                return _h(self, req_meta)
            finally:
                acc = _tl.acc
                tr.emit(ev_end, req=_tl.req, bytes=acc[0], io_s=round(acc[1], 6),
                        dur_s=round(time.monotonic() - t0, 6))
                _tl.acc = None
                _tl.req = None

        cls._handle_request = _handle_request

    _wrap_handle(mw.KVCacheStoreSendingThread, "mc_store_cmt")
    _wrap_handle(mw.KVCacheStoreRecvingThread, "mc_load_done")

    # 저장 큐 상태를 러너가 볼 수 있게 워커 인스턴스를 붙잡아 둔다.
    _init = mw.MooncakeStoreWorker.__init__

    def __init__(self, *a, **kw):
        global LAST_MC_WORKER
        _init(self, *a, **kw)
        LAST_MC_WORKER = self

    mw.MooncakeStoreWorker.__init__ = __init__


def install(transport: str, path: str) -> KvTrace | None:
    """transport에 맞는 계측을 붙이고 KvTrace를 돌려준다. 해당 없으면 None."""
    tr = KvTrace(path)
    if transport in ("hybrid", "cufile"):
        _patch_offloading(transport, tr)
    elif transport == "mooncake":
        _patch_mooncake(tr)
    else:
        tr.close()
        os.remove(path)
        return None
    return tr
