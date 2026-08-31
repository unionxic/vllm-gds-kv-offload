# Phase 0 보완: prefix hit이 "CPU primary hit"이 아니라 "FS tier hit → promotion"임을 분리 증명.
#   store 모드: 새 엔진으로 프롬프트 실행 → chunk 파일 생성, 출력 토큰 저장
#   load 모드:  완전히 새 프로세스/엔진(CPU primary 비어 있음)에서 같은 프롬프트 실행.
#               fs.manager.load_block 호출을 계수(in-process 엔진 + 몽키패치)해
#               promotion 발생을 직접 관측하고, 출력 토큰 일치로 KV 정합성 검증.
import json
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODE = sys.argv[1]  # store | load
HERE = os.path.dirname(os.path.abspath(__file__))
KVROOT = os.path.join(HERE, "kvroot-fshit")
TOKENS_JSON = os.path.join(HERE, "fs_hit_tokens.json")

# --- load 모드: 엔진 생성 전에 fs 티어 I/O 함수를 계수 래퍼로 교체 ---
counters = {"load": 0, "store": 0}
if MODE == "load":
    import vllm.v1.kv_offload.tiering.fs.manager as fsm

    orig_load, orig_store = fsm.load_block, fsm.store_block

    def counting_load(*a, **kw):
        counters["load"] += 1
        return orig_load(*a, **kw)

    def counting_store(*a, **kw):
        counters["store"] += 1
        return orig_store(*a, **kw)

    fsm.load_block, fsm.store_block = counting_load, counting_store

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

llm = LLM(
    model="facebook/opt-125m",
    kv_transfer_config=KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": 1 << 30,
            "block_size": 16,
            "secondary_tiers": [{"type": "fs", "root_dir": KVROOT}],
        },
    ),
    gpu_memory_utilization=0.5,
    max_model_len=2048,
    enforce_eager=True,
)

prompt = "The history of computing begins with " + " and".join(
    f" chapter {i} about machines" for i in range(120)
)
sp = SamplingParams(max_tokens=32, temperature=0.0)

t0 = time.perf_counter()
out = llm.generate([prompt], sp)
dt = time.perf_counter() - t0
tokens = list(out[0].outputs[0].token_ids)
print(f"[{MODE}] gen {dt:.3f}s, {len(tokens)} tokens")

if MODE == "store":
    time.sleep(3)  # 비동기 store 대기
    nbin = sum(len([f for f in fs if f.endswith(".bin")]) for _, _, fs in os.walk(KVROOT))
    json.dump(tokens, open(TOKENS_JSON, "w"))
    print(f"[store] chunks on disk: {nbin}")
    sys.exit(0 if nbin > 0 else 1)
else:
    ref = json.load(open(TOKENS_JSON))
    match = tokens == ref
    print(f"[load] load_block calls (FS->CPU promotion): {counters['load']}")
    print(f"[load] store_block calls: {counters['store']}")
    print(f"[load] output tokens match store run: {match}")
    sys.exit(0 if (counters["load"] > 0 and match) else 1)
