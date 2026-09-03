# Pressure-aware Prefix Value Admission — scheduler-side value scorer + worker hint.
#   in-process(VLLM_ENABLE_V1_MULTIPROCESSING=0)에서 scheduler와 worker가 메모리 공유.
#   scheduler가 store 후보 chunk(=OffloadKey→path)의 관측 이력으로 value 산출 →
#   path→(intent, priority) hint. worker._admit(value 정책)이 hint를 읽어 압박과 결합.
#
# value = P(SSD-tier reuse) × saved_prefill / write_cost   (1단계 게이트 신호)
#   P(SSD-reuse) ≈ evict 확률: 재사용 거리 > GPU/CPU 상주 → SSD가 유일 hit원.
#   seen-twice 필터: 미재사용 예상 chunk 배제. seen-twice 미충족은 DROP intent.
# 정책 변형(mode): random_skip | seen_twice | value_density | oracle
#   random_skip: hint 없음(비차단 skip과 동일). seen_twice: freq>=2만 RING.
#   value_density: value 랭킹으로 RING/CPU/DROP. oracle: 사전계산된 future-useful.
import os
from collections import defaultdict

# scheduler가 본 chunk-key 관측 이력 (bytes key)
_seen = defaultdict(int)
_last = {}
_rd_sum = defaultdict(int)
_rd_n = defaultdict(int)
_step = [0]
_path_stats = {}       # path -> dict(freq, avg_rd)  (prepare_store 시점 스냅샷)
_oracle_paths = None   # oracle 모드: 저장 가치 있는 path 집합 (사전계산)
_MODE = ["value_density"]

GPUCPU_BLOCKS = int(13.5 * 2**30 // (16 * 327_680))


def set_mode(mode):
    _MODE[0] = mode


def set_oracle(paths):
    global _oracle_paths
    _oracle_paths = set(paths)


def observe(path, key):
    """prepare_store에서 각 store 후보 chunk를 관측 (scheduler 스레드)."""
    t = _step[0]
    if key in _last:
        _rd_sum[key] += t - _last[key]
        _rd_n[key] += 1
    _last[key] = t
    _seen[key] += 1
    freq = _seen[key]
    avg_rd = _rd_sum[key] / _rd_n[key] if _rd_n[key] else 0
    _path_stats[path] = dict(freq=freq, avg_rd=avg_rd)


def tick():
    _step[0] += 1


def hint(path):
    """worker._admit이 호출. (intent, priority) 반환."""
    mode = _MODE[0]
    st = _path_stats.get(path, dict(freq=1, avg_rd=0))
    if mode == "oracle":
        if _oracle_paths is not None and path in _oracle_paths:
            return ("GDS_RING", 1)
        return ("DROP", 0)
    if mode == "random_skip":
        return ("GDS_RING", 0)          # 압박 시 비차단 drop (priority 0)
    if mode == "seen_twice":
        return ("GDS_RING", 1) if st["freq"] >= 2 else ("DROP", 0)
    # value_density
    if st["freq"] < 2:
        return ("DROP", 0)              # seen-twice 필터
    resident = max(1, GPUCPU_BLOCKS // 64)   # chunk 단위 상주 근사
    if st["avg_rd"] <= resident:
        return ("DROP", 0)             # 가까운 재사용 = GPU/CPU가 잡음
    # 고가치: 압박 시에도 CPU fallback으로 보존 (priority 1)
    return ("GDS_RING", 1)


def install(worker_transport, mode):
    """expfs worker의 FilesystemManager.prepare_store를 관측 훅으로, transport에
    hint_fn 주입. mode에 따라 hint 정책 선택."""
    set_mode(mode)
    worker_transport.set_hint_fn(hint)
    import expfs
    M = expfs.FilesystemManager
    _ps = M.prepare_store
    def prepare_store(self, keys, req_context):
        out = _ps(self, keys, req_context)
        if out is not None:
            for k, p in zip(out.keys_to_store, out.store_spec.paths):
                observe(p, bytes(k))
        tick()
        return out
    M.prepare_store = prepare_store
    return dict(seen=_seen, path_stats=_path_stats)


def reset():
    _seen.clear(); _last.clear(); _rd_sum.clear(); _rd_n.clear()
    _path_stats.clear(); _step[0] = 0
