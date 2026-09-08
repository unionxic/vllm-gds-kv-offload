"""07 스모크(opt-2.7b): 가중치 3단 오프로드(prefetch+SSD cufile) + KV→SSD(expfs) 한 엔진 동거.
   usage: python smoke_combo.py baseline|combo [kv_transport]
   combo: 프리픽스 공유 프롬프트 6개를 2라운드 → 2라운드는 SSD KV 적중. 토큰은 baseline과 비교."""
import json, os, re, sys, time
mode = sys.argv[1]; kvt = sys.argv[2] if len(sys.argv) > 2 else "cufile"
kv_gib = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # combo: GPU KV 상한(GiB), 0=기본
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
os.environ.setdefault("CUFILE_ENV_PATH_JSON", os.path.join(ROOT, "experiments", "01-feasibility", "cufile-microbench", "cufile_trace.json"))
sys.path.insert(0, os.path.join(ROOT, "lib"))
out = os.path.join(ROOT, "results", "combined", "smoke"); os.makedirs(out, exist_ok=True); os.chdir(out)
tag = mode if mode == "baseline" else f"combo-{kvt}" + (f"-kv{kv_gib}" if kv_gib else "")
if os.path.exists("cufile.log"): os.remove("cufile.log")
import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def nvfs():
    for line in open("/proc/driver/nvidia-fs/stats"):
        if line.startswith("Reads"):
            d = dict(kv.split("=") for kv in line.split(":")[1].split() if "=" in kv); return int(d.get("n", 0))
    return 0
n0 = nvfs()
kw = {}
if mode == "combo":
    import shutil
    wroot = os.path.join(out, f"wssd-{kvt}"); kvroot = os.path.join(out, f"kv-{kvt}")
    shutil.rmtree(wroot, ignore_errors=True); shutil.rmtree(kvroot, ignore_errors=True)
    kw = dict(offload_backend="prefetch", offload_group_size=4, offload_num_in_group=3, offload_prefetch_step=1,
              offload_ssd_path=wroot, offload_host_fraction=0.005, offload_ssd_transport="cufile", offload_ssd_io_threads=4,
              kv_transfer_config=KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                  kv_connector_extra_config={"spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
                      "expfs_root_dir": kvroot, "expfs_transport": kvt, "expfs_read_threads": 4, "expfs_write_threads": 4,
                      "block_size": 64}))
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
    cnt = {"matched": 0}
    _gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
    def _gmw(self, request, n):
        r = _gm(self, request, n); cnt["matched"] += r[0] or 0; return r
    osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw
    import expfs
    cnt.update(kv_read_n=0, kv_read_b=0, kv_write_n=0, kv_write_b=0)
    for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport):
        _r, _w = cls.read_chunk, cls.write_chunk
        def mk(fn, kn, kb):
            def inner(self, path, spans, cb):
                cnt[kn] += 1; cnt[kb] += sum(sp_[2] for sp_ in spans); return fn(self, path, spans, cb)
            return inner
        cls.read_chunk = mk(_r, "kv_read_n", "kv_read_b"); cls.write_chunk = mk(_w, "kv_write_n", "kv_write_b")
    if kv_gib: kw["kv_cache_memory_bytes"] = int(kv_gib * 2**30)
llm = LLM(model="facebook/opt-2.7b", gpu_memory_utilization=0.5, max_model_len=1200 if kv_gib else 2048, enforce_eager=True, **kw)
tok = llm.get_tokenizer(); vocab = tok.vocab_size
import random
rng = random.Random(7)
# 프롬프트마다 서로 다른 1,088토큰(공유 프리픽스면 GPU 프리픽스 캐시가 먼저 맞아 SSD까지 안 감). 2라운드 = 같은 프롬프트 재방문.
prompts = [{"prompt_token_ids": [rng.randrange(1000, vocab - 1000) for _ in range(1088)]} for _ in range(6)]
sp = SamplingParams(max_tokens=16, temperature=0)
res = {"mode": tag}
for rnd in (1, 2):
    if mode == "combo":
        for k in cnt: cnt[k] = 0
    t = time.time(); outs = llm.generate(prompts, sp, use_tqdm=False); dt = time.time() - t
    res[f"round{rnd}_s"] = round(dt, 2); res[f"round{rnd}_ids"] = [o.outputs[0].token_ids for o in outs]
    if mode == "combo": res[f"round{rnd}_kv"] = dict(cnt)
res["nvfs_reads"] = nvfs() - n0
text = open("cufile.log", errors="replace").read() if os.path.exists("cufile.log") else ""
px = len(re.findall(r"cufio-px.*(?:read|write)", text, re.I)) - text.count("px-pool")
res["cufile_cls"] = "COMPAT-POSIX" if px > 0 else ("INTERNAL-BOUNCE" if "read_through_bounce_buffer completed" in text else "DIRECT")
if mode == "combo":
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(kvroot) for f in fs]
    res["kv_files"] = len(files); res["kv_bytes"] = sum(os.path.getsize(f) for f in files)
    res["round_ids_equal"] = res["round1_ids"] == res["round2_ids"]
    b = os.path.join(out, "baseline.json")
    if os.path.exists(b):
        base = json.load(open(b)); res["match_baseline"] = base["round1_ids"] == res["round1_ids"] and base["round1_ids"] == res["round2_ids"]
json.dump(res, open(f"{tag}.json", "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if not k.endswith("_ids")}), flush=True)
try: llm.llm_engine.engine_core.shutdown()
except Exception as e: print("shutdown:", e)
