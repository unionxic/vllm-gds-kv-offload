# Phase 2 smoke: ExperimentalFilesystemSpec store→(새 프로세스)load 왕복 + 토큰 일치.
# usage: python smoke.py {store|load} {cufile|posix}
import json
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODE, TRANSPORT = sys.argv[1], sys.argv[2]
BLOCK = int(sys.argv[3]) if len(sys.argv) > 3 else 16
HERE = os.path.dirname(os.path.abspath(__file__))
KVROOT = os.path.join(HERE, f"kvroot-{TRANSPORT}-b{BLOCK}")
TOKENS_JSON = os.path.join(HERE, f"smoke_tokens_{TRANSPORT}_b{BLOCK}.json")

# transport 호출 계수 (엔진 생성 전 패치) — 경로 사용의 직접 증거
import expfs

calls = {"write": 0, "read": 0}
for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport):
    _w, _r = cls.write_chunk, cls.read_chunk

    def wrap(fn, key):
        def inner(self, *a, **kw):
            calls[key] += 1
            return fn(self, *a, **kw)
        return inner

    cls.write_chunk, cls.read_chunk = wrap(_w, "write"), wrap(_r, "read")

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

llm = LLM(
    model="facebook/opt-125m",
    kv_transfer_config=KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "ExperimentalFilesystemSpec",
            "spec_module_path": "expfs",
            "expfs_root_dir": KVROOT,
            "expfs_transport": TRANSPORT,
            "block_size": BLOCK,
        },
    ),
    gpu_memory_utilization=0.5,
    max_model_len=2048,
    enforce_eager=True,
)

prompt = "The history of computing begins with " + " and".join(
    f" chapter {i} about machines" for i in range(120)
)
sp = SamplingParams(max_tokens=24, temperature=0.0)

t0 = time.perf_counter()
out = llm.generate([prompt], sp)
dt = time.perf_counter() - t0
tokens = list(out[0].outputs[0].token_ids)
print(f"[{MODE}/{TRANSPORT}] gen {dt:.3f}s, {len(tokens)} tokens")

if MODE == "store":
    time.sleep(3)
    nbin = sum(len([f for f in fs if f.endswith(".bin")]) for _, _, fs in os.walk(KVROOT))
    ntmp = sum(len([f for f in fs if f.endswith(".tmp")]) for _, _, fs in os.walk(KVROOT))
    json.dump(tokens, open(TOKENS_JSON, "w"))
    print(f"[store] .bin files: {nbin}, stray .tmp: {ntmp}")
    sys.exit(0 if (nbin > 0 and ntmp == 0) else 1)
else:
    ref = json.load(open(TOKENS_JSON))
    match = tokens == ref
    print(f"[load] transport read_chunk calls: {calls['read']}, "
          f"write_chunk calls: {calls['write']}")
    print(f"[load] tokens match store run: {match}")
    sys.exit(0 if (match and calls["read"] > 0) else 1)
