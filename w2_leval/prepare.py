# W2 LEval 준비: 실제 문서+실제 질문으로 shared-prefix workload 구성.
#   real-text, synthetic-interleaving long-document workload (production 아님).
#
# 규칙:
#  - 문서당 질문 2개 이상, OPT-2.7b tokenizer 기준 문서가 충분히 긴 것만
#  - prefix = [고정 system instruction] + [문서] 를 정확히 1920 토큰으로 절단
#    (1920 = 16과 64의 공배수 → b16: 120블록, b64: 30블록)
#  - 총 입력 2032 토큰 이하: 질문 예산 = 2032 - 1920 = 112 (delimiter 포함)
#  - random padding 금지, 문서 조각 이어붙이기 금지, 짧은 문서는 제외
#  - prompt_token_ids 직접 전달 전제 → 여기서 토큰 배열까지 확정
# 산출: results/w2_leval/dataset_manifest.json, w2_leval/workload.json
import hashlib
import json
import os
import random
import time

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "w2_leval")
os.makedirs(RESULTS, exist_ok=True)

SEED = 20260901
TARGET_DOCS = 64
PREFIX_TOKENS = 1920
MAX_INPUT = 2032
SYSTEM = ("You are a careful assistant. Read the following document and then "
          "answer the question at the end.\n\nDocument:\n")
DELIM = "\n\nQuestion: "

tok = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
sys_ids = tok(SYSTEM, add_special_tokens=True)["input_ids"]  # BOS 포함
delim_ids = tok(DELIM, add_special_tokens=False)["input_ids"]
q_budget = MAX_INPUT - PREFIX_TOKENS - len(delim_ids)
assert PREFIX_TOKENS % 16 == 0 and PREFIX_TOKENS % 64 == 0
print(f"system={len(sys_ids)}tok delim={len(delim_ids)}tok q_budget={q_budget}tok")

# datasets 5.x는 스크립트형 로더를 지원하지 않으므로 hub의 jsonl을 직접 파싱
api = HfApi()
info = api.repo_info("L4NLP/LEval", repo_type="dataset")
REVISION = info.sha
files = [f for f in api.list_repo_files("L4NLP/LEval", repo_type="dataset")
         if f.endswith(".jsonl")]
print(f"LEval revision={REVISION[:12]} jsonl files={len(files)}")

docs = []
schema_note = {}
for path in sorted(files):
    cfg = os.path.basename(path).replace(".jsonl", "")
    try:
        local = hf_hub_download("L4NLP/LEval", path, repo_type="dataset",
                                revision=REVISION)
        rows = [json.loads(l) for l in open(local)]
    except Exception as e:
        schema_note[cfg] = f"load 실패: {e}"
        continue
    cols = sorted(rows[0].keys()) if rows else []
    schema_note[cfg] = f"{len(rows)} rows, cols={cols}"
    if not rows or not {"input", "instructions", "outputs"} <= set(rows[0]):
        continue
    for idx, row in enumerate(rows):
        ins, outs = row["instructions"], row["outputs"]
        if not isinstance(ins, list) or len(ins) < 2:
            continue
        doc_ids = tok(row["input"], add_special_tokens=False)["input_ids"]
        need = PREFIX_TOKENS - len(sys_ids)
        if len(doc_ids) < need:
            continue  # 짧은 문서 제외 (패딩 금지)
        prefix = sys_ids + doc_ids[:need]
        assert len(prefix) == PREFIX_TOKENS
        qs = []
        seen_q = set()
        for qi, (q, o) in enumerate(zip(ins, outs)):
            q_ids = tok(q, add_special_tokens=False)["input_ids"]
            truncated = len(q_ids) > q_budget
            q_ids = q_ids[:q_budget]  # 명시적 질문 절단
            if len(q_ids) == 0 or tuple(q_ids) in seen_q:
                continue  # 절단 후 중복 질문 제외
            seen_q.add(tuple(q_ids))
            ref_len = len(tok(o, add_special_tokens=False)["input_ids"]) if isinstance(o, str) else 0
            qs.append(dict(q_index=qi, tokens=q_ids, truncated=truncated,
                           ref_output_tokens=ref_len))
        if len(qs) < 2:
            continue
        docs.append(dict(
            subtask=cfg, row_index=idx,
            doc_sha=hashlib.sha256(row["input"].encode()).hexdigest()[:16],
            doc_tokens_full=len(doc_ids),
            prefix=prefix, questions=qs[:4],  # 문서당 최대 4질문 보존
        ))
# 절단된 prefix 기준 dedup (서로 다른 원문이라도 앞 1920토큰이 같으면 같은 prefix)
seen_pref = set()
unique_docs = []
for d in docs:
    key = hashlib.sha256(bytes(str(d["prefix"]), "utf-8")).hexdigest()
    if key in seen_pref:
        continue
    seen_pref.add(key)
    unique_docs.append(d)
dropped = len(docs) - len(unique_docs)
docs = unique_docs
print(f"적격 문서: {len(docs)} (절단-prefix 중복 {dropped}건 제외, 요구 {TARGET_DOCS})")

rng = random.Random(SEED)
rng.shuffle(docs)
# 도메인 다양성: subtask당 상한을 두고 라운드로빈 선택
by_task = {}
for d in docs:
    by_task.setdefault(d["subtask"], []).append(d)
selected = []
while len(selected) < TARGET_DOCS and any(by_task.values()):
    for k in sorted(by_task):
        if by_task[k] and len(selected) < TARGET_DOCS:
            selected.append(by_task[k].pop(0))
print(f"선정: {len(selected)}문서, subtask 분포:",
      {k: sum(1 for d in selected if d['subtask'] == k) for k in sorted({d['subtask'] for d in selected})})

# 검증
prefixes = [tuple(d["prefix"]) for d in selected]
assert len(set(prefixes)) == len(prefixes), "문서 prefix가 서로 달라야 함"
for d in selected:
    sufs = [tuple(q["tokens"]) for q in d["questions"]]
    assert len(set(sufs)) == len(sufs), "질문 suffix가 서로 달라야 함"
    for q in d["questions"]:
        total = PREFIX_TOKENS + len(delim_ids) + len(q["tokens"])
        assert total <= MAX_INPUT, total

workload = dict(
    kind="real-text synthetic-interleaving long-document workload",
    seed=SEED, prefix_tokens=PREFIX_TOKENS, max_input=MAX_INPUT,
    system_tokens=sys_ids, delim_tokens=delim_ids,
    blocks_b16=PREFIX_TOKENS // 16, blocks_b64=PREFIX_TOKENS // 64,
    docs=selected,
)
json.dump(workload, open(os.path.join(HERE, "workload.json"), "w"))

manifest = dict(
    dataset="L4NLP/LEval", download_date=time.strftime("%Y-%m-%d"), revision=REVISION,
    license="LEval 저장소 명시 라이선스 준수(원본 미커밋)",
    seed=SEED, target_docs=TARGET_DOCS, selected_docs=len(selected),
    eligible_docs=len(docs), configs_seen=schema_note,
    selection=[{k: d[k] for k in ("subtask", "row_index", "doc_sha",
                                  "doc_tokens_full")} |
               {"n_questions": len(d["questions"])} for d in selected],
)
json.dump(manifest, open(os.path.join(RESULTS, "dataset_manifest.json"), "w"), indent=1)
print("완료: workload.json + dataset_manifest.json")
print(f"b16 prefix 블록 {PREFIX_TOKENS//16}, b64 prefix 블록 {PREFIX_TOKENS//64}")
