# 최신 main 회귀 검증 1: TieringOffloadingSpec 종료 시점 _req_state race.
#   우리 pinned base(568afb3a13, #49671 미포함)에서는 LEval 64문서 C 런이
#   가드 없이는 3/3 크래시(KeyError '3x-...'). #49671 병합 후 main에서 사라지는지 확인.
# 가드 없음 — 재현이 목적. 통과 기준: 128요청 완주 + KeyError 0.
import json
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

W = json.load(open(os.path.join(HERE, "..", "w2_leval", "workload.json")))
docs = W["docs"][:64]
delim = W["delim_tokens"]

kvroot = os.path.join(HERE, "kvroot-race")
import shutil
shutil.rmtree(kvroot, ignore_errors=True)

llm = LLM(model="facebook/opt-2.7b",
          kv_transfer_config=KVTransferConfig(
              kv_connector="OffloadingConnector", kv_role="kv_both",
              kv_connector_extra_config={
                  "spec_name": "TieringOffloadingSpec",
                  "cpu_bytes_to_use": 8 << 30, "block_size": 16,
                  "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]}),
          gpu_memory_utilization=0.7, max_model_len=2048, enforce_eager=True)
sp = SamplingParams(max_tokens=1, temperature=0.0)

import random
rng = random.Random(W["seed"] + 1)
order2 = list(range(len(docs)))
rng.shuffle(order2)

n = 0
t0 = time.perf_counter()
for rnd, order, qidx in ((1, list(range(len(docs))), 0), (2, order2, 1)):
    for di in order:
        d = docs[di]
        q = d["questions"][min(qidx, len(d["questions"]) - 1)]
        llm.generate([{"prompt_token_ids": d["prefix"] + delim + q["tokens"]}],
                     sp, use_tqdm=False)
        n += 1
print(f"RACE-REPRO PASS: {n} requests, {time.perf_counter()-t0:.0f}s, KeyError 0")
try:
    llm.llm_engine.engine_core.shutdown()
except Exception:
    pass
shutil.rmtree(kvroot, ignore_errors=True)
