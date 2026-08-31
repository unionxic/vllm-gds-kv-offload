# cross-layer uniform KV layout이 왜 선택 안 됐는지 게이트별로 판정.
# VLLM_ENABLE_V1_MULTIPROCESSING=0으로 엔진을 in-process로 띄워 내부 접근.
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM
from vllm.config import KVTransferConfig

KVROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kvroot-layout")

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

# 엔진 내부로 진입 (in-process라 가능)
core = llm.llm_engine.engine_core
inner = getattr(core, "engine_core", core)
executor = inner.model_executor
worker = executor.driver_worker
mr = worker.model_runner if hasattr(worker, "model_runner") else worker.worker.model_runner

from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

print("has_kv_transfer_group:", has_kv_transfer_group())
if has_kv_transfer_group():
    print("prefer_cross_layer_blocks:", get_kv_transfer_group().prefer_cross_layer_blocks)

ag = mr.attn_groups
print("attn_groups:", len(ag), [len(g) for g in ag])
spec = ag[0][0].kv_cache_spec
print("spec type:", type(spec).__name__)
print("indexes_kv_by_block_stride:", getattr(spec, "indexes_kv_by_block_stride", "N/A"))
backend = ag[0][0].backend
print("backend:", backend.__name__)
try:
    print("stride_order(no layers):", backend.get_kv_cache_stride_order(include_num_layers_dimension=False))
    print("stride_order(layers):", backend.get_kv_cache_stride_order(include_num_layers_dimension=True))
except Exception as e:
    print("stride_order raised:", type(e).__name__, e)
print("runner module:", type(mr).__module__)
if hasattr(mr, "use_uniform_kv_cache"):
    print("use_uniform_kv_cache:", mr.use_uniform_kv_cache(ag))
else:
    print("use_uniform_kv_cache: N/A (V2 runner, cross-layer TODO)")

# GPU KV 텐서 배치: 같은 storage 공유 여부 + data_ptr 간격
kv = mr.kv_caches
ptrs = sorted(t.data_ptr() for t in kv)
print("n kv tensors:", len(kv), "shape0:", tuple(kv[0].shape), "stride0:", kv[0].stride())
print("ptr deltas:", [ptrs[i + 1] - ptrs[i] for i in range(min(3, len(ptrs) - 1))])
print("storage sizes:", [t.untyped_storage().nbytes() for t in kv[:2]])
