# 5단계용 real-text, synthetic-interleaving admission workload.
#   기존 64 LEval 문서(실텍스트)를 재료로, admission 변별을 만드는 요청 스케줄 생성.
#   4개 재사용 카테고리를 의도적으로 섞는다:
#     one_shot   : 1회만 등장 → 저장 무가치 (admission이 drop해야 이득)
#     near_reuse : 짧은 거리 재사용 → 재사용 시 GPU/CPU에 상주 (SSD 무가치, skip해도 무손실)
#     far_reuse  : 긴 거리 재사용 → 재사용 시 evict됨 (SSD가 유일 hit원, 저장해야 함)
#     repeated   : 여러 번 등장 (high frequency, seen-twice가 잘 잡음)
#   near/far 거리는 GPU+CPU 용량(문서 수 기준) 기준으로 설정.
# 산출: w2_leval/mixed_workload.json (요청 스케줄 = [(doc_idx, q_idx, category)...])
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(HERE, "workload.json")))
docs = W["docs"]
NDOC = len(docs)

# opt-2.7b: prefix 1920tok ≈ 0.586GiB KV. GPU+CPU ≈ 13.5GiB → 약 23문서 상주.
# near = 상주 창(< 15문서 간격), far = 상주 초과(> 30문서 간격).
RESIDENT_DOCS = 23
NEAR_GAP = 8
FAR_GAP = 40
SEED = W["seed"] + 101

rng = random.Random(SEED)

# 문서를 카테고리에 배정 (겹치지 않게)
idx = list(range(NDOC))
rng.shuffle(idx)
n_each = NDOC // 4
cat_docs = dict(
    one_shot=idx[:n_each],
    near_reuse=idx[n_each:2 * n_each],
    far_reuse=idx[2 * n_each:3 * n_each],
    repeated=idx[3 * n_each:],
)

# 스케줄 구성: 슬롯 배열에 각 카테고리의 재등장을 배치
schedule = []  # (doc_idx, q_idx, category, occurrence)


# 첫 등장 배치 (모든 문서의 occ=0을 한 번씩, 섞음)
first = []
for cat in ("one_shot", "near_reuse", "far_reuse", "repeated"):
    for di in cat_docs[cat]:
        first.append((di, cat, 0))
rng.shuffle(first)
final = list(first)                       # 리스트에 재등장을 삽입해 나감
pos0 = {di: p for p, (di, cat, occ) in enumerate(final)}   # 첫 등장 위치

# 재등장 삽입 (near=짧은 간격, far=긴 간격, repeated=2·3차 무작위)
inserts = []   # (target_pos, (di, cat, occ))
for di in cat_docs["near_reuse"]:
    inserts.append((pos0[di] + NEAR_GAP, (di, "near_reuse", 1)))
for di in cat_docs["far_reuse"]:
    inserts.append((pos0[di] + FAR_GAP, (di, "far_reuse", 1)))
for di in cat_docs["repeated"]:
    inserts.append((pos0[di] + rng.randint(6, 30), (di, "repeated", 1)))
    inserts.append((pos0[di] + rng.randint(31, 70), (di, "repeated", 2)))

inserts.sort(key=lambda x: x[0])
# 뒤에서부터 삽입하면 앞 인덱스가 안 밀림
for tgt, item in sorted(inserts, key=lambda x: -x[0]):
    final.insert(min(max(tgt, 0), len(final)), item)


def emit(di, cat, occ):
    q = occ % len(docs[di]["questions"])
    schedule.append(dict(doc=di, q=q, category=cat, occ=occ))


for di, cat, occ in final:
    emit(di, cat, occ)

# 카테고리별 통계
from collections import Counter
cat_count = Counter(s["category"] for s in schedule)
out = dict(
    kind="real-text synthetic-interleaving admission workload",
    seed=SEED, resident_docs=RESIDENT_DOCS, near_gap=NEAR_GAP, far_gap=FAR_GAP,
    prefix_tokens=W["prefix_tokens"], delim_tokens=W["delim_tokens"],
    schedule=schedule, n_requests=len(schedule),
    category_counts=dict(cat_count), cat_docs=cat_docs,
    doc_ref=os.path.join("w2_leval", "workload.json"))
json.dump(out, open(os.path.join(HERE, "mixed_workload.json"), "w"))
print(f"혼합 스케줄 {len(schedule)}요청: {dict(cat_count)}")
print(f"카테고리 문서수: { {k: len(v) for k, v in cat_docs.items()} }")
