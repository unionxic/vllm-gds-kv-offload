# Phase 0 smoke: TieringOffloadingSpec + fs tier가 opt-125m에서 실제로 도는지 확인.
# 성공 기준: kvroot에 .bin chunk 파일 생성(store) + 2차 실행에서 프로모션(load) 경로 통과.
import os
import sys
import time

KVROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kvroot")

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
out1 = llm.generate([prompt], sp)
t1 = time.perf_counter()
print(f"[run1] {t1 - t0:.3f}s  tokens_out={len(out1[0].outputs[0].token_ids)}")

# store는 비동기 — 파일 생성 대기
time.sleep(3)
nbin = sum(len([f for f in fs if f.endswith(".bin")]) for _, _, fs in os.walk(KVROOT))
print(f"[store] .bin files in kvroot: {nbin}")

t2 = time.perf_counter()
out2 = llm.generate([prompt], sp)
t3 = time.perf_counter()
print(f"[run2] {t3 - t2:.3f}s (prefix hit expected)")
sys.exit(0 if nbin > 0 else 1)
