# 1단계: 오프라인 trace 시뮬레이션 (GPU 불필요, admission 게이트).
#   질문: store admission '신호'가 동일 write-byte budget에서 arrival-order random보다
#   useful external hit을 더 많이 남기는가. 통과해야 실제 구현 진행.
#
# 설계(신호력 측정): 각 chunk의 전역 feature를 계산 → 정책별로 랭킹 → 상위 budget개를
#   prefix-closed로 선택 → 그 store-set으로 replay해 useful hit 측정. 동일 budget에서
#   정책 간 비교. random은 임의 부분집합(seed 다중), oracle은 실제 미래 유용성 랭킹.
#   (온라인 근사는 2단계 이후. 게이트는 '신호가 예측력이 있는가'만 판정.)
import json
import os
import random as rnd
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(HERE, "..", "..", "..", "experiments", "02-bailian",
                     "replay600", "trace", "qwen_coder.jsonl")
BYTES_PER_BLOCK = 16 * 327_680
GPUCPU_BLOCKS = int(13.5 * 2**30 // BYTES_PER_BLOCK)
MS_PER_TOKEN = 0.5
CAP = 127


def load(n=600):
    reqs = []
    for line in open(TRACE):
        reqs.append(json.loads(line)["hash_ids"][:CAP])
        if len(reqs) >= n:
            break
    return reqs


def gpucpu_sim(reqs):
    """정책 독립 LRU. resident_at[i] = 요청 i 도착 시 GPU/CPU 블록집합."""
    lru = OrderedDict()
    resident_at = []
    for blocks in reqs:
        resident_at.append(set(lru.keys()))
        for b in blocks:
            if b in lru:
                lru.move_to_end(b)
            else:
                lru[b] = True
                if len(lru) > GPUCPU_BLOCKS:
                    lru.popitem(last=False)
    return resident_at


def chunk_features(reqs, bpc, resident_at):
    """각 store 후보 chunk의 feature. chunk_id = (req_idx, chunk_pos).
    반환: chunks[list], 각 dict(id, req, pos, blocks, freq, avg_rd, saved_ms,
          bytes, future_useful, parent)."""
    occ = defaultdict(list)
    for i, blocks in enumerate(reqs):
        for b in blocks:
            occ[b].append(i)
    # 각 (i, b): 다음 발생에서 evict되어 SSD가 유일 hit원이 되는가 (oracle 신호)
    useful_occ = defaultdict(set)      # req_idx -> set(block) 저장 시 미래 SSD hit
    for b, idxs in occ.items():
        for k in range(len(idxs) - 1):
            i, j = idxs[k], idxs[k + 1]
            if b not in resident_at[j]:
                useful_occ[i].add(b)
    seen = defaultdict(int)
    last = {}
    rd_sum = defaultdict(int)
    rd_n = defaultdict(int)
    chunks = []
    for i, blocks in enumerate(reqs):
        n_ch = (len(blocks) + bpc - 1) // bpc
        for c in range(n_ch):
            cb = blocks[c * bpc:(c + 1) * bpc]
            freq = max((seen[b] + 1 for b in cb), default=1)  # 이번 관측 포함
            rds = [rd_sum[b] / rd_n[b] for b in cb if rd_n[b]]
            avg_rd = sum(rds) / len(rds) if rds else 0
            fut = sum(1 for b in cb if b in useful_occ.get(i, ()))
            chunks.append(dict(
                id=(i, c), req=i, pos=c, blocks=cb,
                freq=freq, avg_rd=avg_rd,
                saved_ms=len(cb) * 16 * MS_PER_TOKEN,
                bytes=bpc * BYTES_PER_BLOCK, future_useful=fut,
                parent=(i, c - 1) if c > 0 else None))
        # 관측 갱신 (chunk feature는 관측 '전' 상태 기준이므로 여기서 갱신)
        for b in blocks:
            if b in last:
                rd_sum[b] += i - last[b]
                rd_n[b] += 1
            last[b] = i
            seen[b] += 1
    return chunks


def rank_key(policy):
    if policy == "seen_twice":
        return lambda c: (1 if c["freq"] >= 2 else 0, c["freq"])
    if policy == "reuse_distance":
        return lambda c: (1 if c["freq"] >= 2 else 0, c["avg_rd"])
    if policy == "value_density":
        # value = P(SSD-tier reuse within horizon) × saved_prefill / write_cost.
        # P(SSD-reuse) ≈ evict 확률: 재사용 거리가 GPU/CPU 상주보다 커 SSD가 유일 hit원.
        # seen-twice는 필터(재사용 안 될 것 배제), 랭킹 주신호는 evict-likelihood.
        def k(c):
            if c["freq"] < 2:
                return (0, 0.0)               # seen-twice 필터 (미재사용 배제)
            resident_reqs = max(1, GPUCPU_BLOCKS // max(1, len(c["blocks"]) * 8))
            if c["avg_rd"] <= resident_reqs:
                return (0, 0.0)               # 가까운 재사용 = GPU/CPU가 잡음 → SSD 제외
            p_ssd = min(1.0, c["avg_rd"] / (resident_reqs * 4))  # evict 확률 근사
            v = p_ssd * c["saved_ms"] / (c["bytes"] / 2**20)     # value density
            return (1, v)
        return k
    if policy == "oracle":
        return lambda c: (1 if c["future_useful"] > 0 else 0, c["future_useful"])
    raise ValueError(policy)


def select(chunks, policy, budget, seed=0):
    """정책 랭킹으로 상위 budget개 chunk 선택 + prefix-closed 보정."""
    by_id = {c["id"]: c for c in chunks}
    if policy == "random_skip":
        idx = list(range(len(chunks)))
        rnd.Random(seed).shuffle(idx)
        chosen = set(chunks[i]["id"] for i in idx[:budget])
    else:
        key = rank_key(policy)
        ranked = sorted(chunks, key=key, reverse=True)
        chosen = set(c["id"] for c in ranked[:budget] if key(c)[0] > 0)
    # prefix-closed: 선택된 chunk의 모든 부모 chunk를 포함 (부모 없으면 자식 무효)
    closed = set()
    for cid in chosen:
        chain = []
        cur = by_id[cid]
        ok = True
        while cur is not None:
            chain.append(cur["id"])
            p = cur["parent"]
            cur = by_id.get(p) if p else None
            if p and p not in by_id:
                ok = False
                break
        if ok:
            closed.update(chain)
    return closed


def measure(reqs, store_ids, chunks, bpc):
    """선택된 store-set으로 replay → useful hit·wasted 측정."""
    by_req = defaultdict(dict)
    for c in chunks:
        by_req[c["req"]][c["pos"]] = c
    stored_blocks = set()
    lru = OrderedDict()
    ssd = set()
    useful = 0
    ssd_reads = 0
    written_blocks = set()
    reused = set()
    # 저장 스케줄: 각 요청 처리 후 그 요청의 선택된 chunk를 SSD에 추가
    for i, blocks in enumerate(reqs):
        matched = 0
        for b in blocks:
            if b in lru:
                matched += 1
            elif b in ssd:
                matched += 1
                useful += 16
                ssd_reads += 1
                if b in written_blocks:
                    reused.add(b)
            else:
                break
        for b in blocks:
            if b in lru:
                lru.move_to_end(b)
            else:
                lru[b] = True
                if len(lru) > GPUCPU_BLOCKS:
                    lru.popitem(last=False)
        n_ch = (len(blocks) + bpc - 1) // bpc
        for c in range(n_ch):
            if (i, c) in store_ids:
                cb = blocks[c * bpc:(c + 1) * bpc]
                for b in cb:
                    ssd.add(b)
                    written_blocks.add(b)
    n_written = len(store_ids)
    gib = n_written * bpc * BYTES_PER_BLOCK / 2**30
    wasted = len(written_blocks) - len(reused)
    return dict(useful_hit_tokens=useful, ssd_reads=ssd_reads,
                written_chunks=n_written, gib_written=round(gib, 2),
                hit_yield_per_gib=round(useful / gib, 0) if gib else 0,
                wasted_ratio=round(wasted / max(1, len(written_blocks)), 3))


def main():
    reqs = load(600)
    resident_at = gpucpu_sim(reqs)
    out = []
    policies = ["random_skip", "seen_twice", "reuse_distance", "value_density", "oracle"]
    for bpc, geo in ((16, "V1_b256"), (4, "V2_b64")):
        chunks = chunk_features(reqs, bpc, resident_at)
        demand = len(chunks)
        for frac in (0.1, 0.2, 0.4):
            budget = int(demand * frac)
            for pol in policies:
                if pol == "random_skip":
                    # 다중 seed 평균 (단일 빠른 승리 금지)
                    rs = [measure(reqs, select(chunks, pol, budget, s), chunks, bpc)
                          for s in range(3)]
                    r = {k: (sum(x[k] for x in rs) / len(rs)
                             if isinstance(rs[0][k], (int, float)) else rs[0][k])
                         for k in rs[0]}
                else:
                    r = measure(reqs, select(chunks, pol, budget), chunks, bpc)
                r.update(geo=geo, policy=pol, budget_frac=frac, demand=demand)
                out.append(r)
    json.dump(out, open(os.path.join(HERE, "..", "results", "sim_results.json"), "w"),
              indent=1)
    for geo in ("V1_b256", "V2_b64"):
        print(f"\n===== {geo} (GPU+CPU {GPUCPU_BLOCKS}블록, budget=수요 대비 비율) =====")
        print(f"{'budget':>7s} {'policy':16s} {'usefulHit':>10s} {'GiB':>7s} "
              f"{'hit/GiB':>8s} {'wasted%':>8s} {'vs random':>10s}")
        for frac in (0.1, 0.2, 0.4):
            rnd_r = next(r for r in out if r["geo"] == geo
                        and r["policy"] == "random_skip" and r["budget_frac"] == frac)
            for pol in policies:
                r = next(x for x in out if x["geo"] == geo and x["policy"] == pol
                         and x["budget_frac"] == frac)
                vs = (r["useful_hit_tokens"] - rnd_r["useful_hit_tokens"]) / max(
                    1, rnd_r["useful_hit_tokens"]) * 100
                print(f"{frac:>6.0%} {pol:16s} {r['useful_hit_tokens']:>10.0f} "
                      f"{r['gib_written']:>7.1f} {r['hit_yield_per_gib']:>8.0f} "
                      f"{r['wasted_ratio']*100:>7.0f}% {vs:>+9.0f}%")


if __name__ == "__main__":
    main()
