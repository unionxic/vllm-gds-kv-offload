#!/usr/bin/env python
"""QA harness for the prefetch weight-offload SSD tier (opt-2.7b, V1 runner, eager).

Driver mode (default):
    python qa_ssd.py [--nsys] [--keep] [arm ...]      arms: baseline cpu ssd-posix ssd-cufile
Worker mode (one arm in its own process; the driver spawns this):
    python qa_ssd.py --worker ARM --out ARM.json --ssd-path DIR ...

Each arm: build LLM -> greedy generate 4 prompts (token ids) -> timed 4x32 decode batch.
Records load time, GPU max alloc, VmRSS/VmLck/VmPin, torch pinned-host stats,
/proc/driver/nvidia-fs/stats deltas, files under the SSD dir. The driver aggregates,
compares token ids with baseline, and prints PASS/FAIL. Exit code != 0 if any FAIL.

Environment (run_qa.sh does this): source env.sh, VLLM_USE_V2_MODEL_RUNNER=0,
VLLM_ENABLE_V1_MULTIPROCESSING=0 (engine in-process, so /proc/self is the engine).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results", "weight-offload", "qa-ssd")
MODEL = "facebook/opt-2.7b"
NVFS_STATS = "/proc/driver/nvidia-fs/stats"
NSYS = "/usr/local/cuda/bin/nsys"
GDSCHECK = "/usr/local/cuda/gds/tools/gdscheck"
ARMS = ["baseline", "cpu", "ssd-posix", "ssd-cufile"]
SSD_ARMS = ("ssd-posix", "ssd-cufile")

PROMPTS = [
    "The capital of France is",
    "GPUDirect Storage lets",
    "In a distant galaxy, a small robot named",
    "The recipe for a perfect omelette starts with",
]
DECODE_PROMPTS = [
    "Hello, my name is",
    "The weather today is",
    "Once upon a time",
    "Deep learning models are",
]
DECODE_TOKENS = 32
# cuFile TRACE config (logging.level=TRACE); cufile.log then lands in the worker's CWD.
CUFILE_TRACE_JSON = os.path.join(ROOT, "experiments", "01-feasibility", "cufile-microbench",
                                 "cufile_trace.json")
COMPAT_RE = re.compile(r"compat mode|not registered with nvidia-fs|GDS not supported", re.I)
TIERS_RE = re.compile(r"host (\d+) layers.*?ssd (\d+) layers.*?in (\d+) files")


# ----------------------------------------------------------------------------- common
def arm_kwargs(arm, a, ssd_path):
    if arm == "baseline":
        return {}
    kw = dict(offload_backend="prefetch", offload_group_size=a.group_size,
              offload_num_in_group=a.num_in_group, offload_prefetch_step=a.prefetch_step)
    if arm in SSD_ARMS:
        kw.update(offload_ssd_path=ssd_path, offload_host_fraction=a.host_fraction,
                  offload_ssd_transport=arm.split("-", 1)[1],
                  offload_ssd_io_threads=a.io_threads)
    return kw


def nvfs_snapshot():
    """Parse /proc/driver/nvidia-fs/stats into {section: {key: value}}."""
    out = {}
    try:
        text = open(NVFS_STATS).read()
    except OSError as e:
        return {"_error": {"msg": str(e)}}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        d = {}
        for k, v in re.findall(r"([\w\-]+)=(\S+)", rest):
            d[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
        if not d:
            v = rest.strip()
            d["value"] = int(v) if re.fullmatch(r"-?\d+", v) else v
        out[name] = d
    return out


def nvfs_delta(before, after):
    d = {}
    for sec, kv in after.items():
        for k, v in kv.items():
            b = before.get(sec, {}).get(k)
            if isinstance(v, int) and isinstance(b, int) and v != b:
                d[f"{sec}.{k}"] = v - b
    return d


def nvfs_read_metric(before, after):
    """Counter used as the 'native GDS read happened' witness.
    Reads.n exists only when nvidia_fs rw_stats_enabled=1; otherwise Bar1-map.n
    (GPU BAR1 pinning done by nvidia-fs for every native cuFile DMA path)."""
    if "n" in after.get("Reads", {}):
        return "Reads.n", after["Reads"]["n"] - before.get("Reads", {}).get("n", 0)
    b = before.get("Bar1-map", {}).get("n", 0)
    return "Bar1-map.n", after.get("Bar1-map", {}).get("n", 0) - b


def classify_cufile_log(path, bar1_delta):
    """Classify the kernel path from a TRACE-level cufile.log (lib/path_classify.py rules):
    any cufio-px IO line            -> COMPAT-POSIX   (userspace posix pool; not GDS)
    read_through_bounce_buffer done -> INTERNAL-BOUNCE (nvidia-fs native path via GPU bounce)
    neither + Bar1-map.n delta > 0  -> DIRECT         (registered buffer DMA)
    neither + no Bar1 activity      -> NONE           (no cuFile IO evidence)"""
    c = {"path": path, "exists": os.path.exists(path), "bytes": 0, "lines": 0,
         "px_io": 0, "px_read": 0, "px_write": 0, "bounce_read": 0, "bounce_write": 0,
         "compat_notices": 0, "bar1_delta": bar1_delta, "class": "ABSENT"}
    if not c["exists"]:
        return c
    c["bytes"] = os.path.getsize(path)
    if c["bytes"] == 0:
        c["class"] = "EMPTY"
        return c
    with open(path, errors="replace") as f:
        for line in f:
            c["lines"] += 1
            if "cufio-px" in line:
                if re.search(r"read|write", line, re.I) and "px-pool" not in line:
                    c["px_io"] += 1
                if "cufile_posix_read" in line:
                    c["px_read"] += 1
                elif "cufile_posix_write" in line:
                    c["px_write"] += 1
            elif "read_through_bounce_buffer completed" in line:
                c["bounce_read"] += 1
            elif "write_through_bounce_buffer completed" in line:
                c["bounce_write"] += 1
            if COMPAT_RE.search(line):
                c["compat_notices"] += 1
    if c["px_io"] > 0:
        c["class"] = "COMPAT-POSIX"
    elif c["bounce_read"] > 0 or c["bounce_write"] > 0:
        c["class"] = "INTERNAL-BOUNCE"
    elif bar1_delta and bar1_delta > 0:
        c["class"] = "DIRECT"
    else:
        c["class"] = "NONE"
    return c


def proc_status():
    d = {}
    for line in open("/proc/self/status"):
        m = re.match(r"(Vm\w+|Threads):\s+(\d+)(\s*kB)?", line)
        if m:
            v = int(m.group(2))
            d[m.group(1)] = v * 1024 if m.group(3) else v
    return d


def scan_files(path):
    n, total, entries = 0, 0, []
    if path and os.path.isdir(path):
        for dp, _, fns in os.walk(path):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    sz = os.stat(p).st_size
                except OSError:
                    continue
                n += 1
                total += sz
                if len(entries) < 64:
                    entries.append([os.path.relpath(p, path), sz])
    return {"count": n, "bytes": total, "entries": sorted(entries)}


def meminfo_total():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return 0


def gib(b):
    return f"{b / 2**30:.2f}GiB"


# ----------------------------------------------------------------------------- worker
class _ListHandler:
    """Minimal logging.Handler that keeps offloader-related records."""

    def __new__(cls, sink):
        import logging

        class H(logging.Handler):
            def emit(self, rec):
                try:
                    msg = rec.getMessage()
                except Exception:
                    return
                if re.search(r"offload", msg, re.I) or re.search(r"offload", rec.name, re.I):
                    sink.append(f"{rec.name}: {msg}")

        return H()


def _find_model(llm):
    """In-process V1 engine: LLM -> LLMEngine -> InprocClient -> EngineCore -> executor -> worker -> runner.model.
    (gc.get_objects() cannot see it: vLLM gc.freeze()s after model load.)"""
    import torch.nn as nn
    obj = llm
    for attr in ("llm_engine", "engine_core", "engine_core", "model_executor", "driver_worker",
                 "worker", "model_runner", "model"):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    for _ in range(4):  # unwrap CUDAGraphWrapper/UBatchWrapper-style wrappers
        if isinstance(obj, nn.Module) and type(obj).__name__.endswith("ForCausalLM"):
            return obj
        inner = getattr(obj, "model", None) or getattr(obj, "runnable", None)
        if inner is None:
            break
        obj = inner
    return obj if isinstance(obj, nn.Module) else None


def model_layer_info(llm, hf_config=None):
    """Best effort: locate the loaded nn.Module and size one decoder layer."""
    import torch.nn as nn
    err = "model not found"
    try:
        obj = _find_model(llm)
        if obj is not None:
            for name, m in obj.named_modules():
                if isinstance(m, nn.ModuleList) and name.endswith("layers") and len(m) > 1:
                    b = sum(p.numel() * p.element_size() for p in m[0].parameters())
                    return {"source": "engine", "model_cls": type(obj).__name__,
                            "layers_attr": name, "num_layers": len(m), "layer_bytes": b}
            err = f"no 'layers' ModuleList in {type(obj).__name__}"
    except Exception as e:  # noqa
        err = str(e)
    if hf_config is not None and getattr(hf_config, "model_type", "") == "opt":
        h, f, L = hf_config.hidden_size, hf_config.ffn_dim, hf_config.num_hidden_layers
        params = 4 * h * h + 4 * h + 2 * h * f + h + f + 4 * h
        return {"source": "opt-formula", "num_layers": L, "layer_bytes": params * 2, "note": err}
    return {"source": "none", "note": err}


def offloader_attrs():
    """Best effort: scalar attributes of the active offloader (for debugging)."""
    out = {}
    try:
        from vllm.model_executor.offloader.base import get_offloader
        obj = get_offloader()
        d = {}
        for k, v in vars(obj).items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                d[k] = v
            elif isinstance(v, (list, tuple, dict, set)):
                d[k] = f"<{type(v).__name__} len={len(v)}>"
            else:
                d[k] = f"<{type(v).__name__}>"
        out[type(obj).__name__] = d
    except Exception as e:  # noqa
        out["_error"] = str(e)
    return out


def run_worker(a):
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    ssd_path = a.ssd_path
    if a.worker in SSD_ARMS:
        os.makedirs(ssd_path, exist_ok=True)
    kw = arm_kwargs(a.worker, a, ssd_path)
    res = {"arm": a.worker, "ok": False, "stage": "init", "kwargs": kw, "pid": os.getpid(),
           "env": {k: os.environ.get(k) for k in
                   ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING", "QA_NSYS")}}

    def dump():
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, a.out)

    dump()
    import logging
    import torch
    from vllm import LLM, SamplingParams

    logs = []
    logging.getLogger("vllm").addHandler(_ListHandler(logs))
    res["offloader_log"] = logs

    res["proc_before"] = proc_status()
    res["nvfs_before"] = nvfs_snapshot()
    res["stage"] = "load"
    dump()
    t0 = time.perf_counter()
    llm = LLM(model=MODEL, gpu_memory_utilization=0.5, max_model_len=512,
              enforce_eager=True, **kw)
    torch.cuda.synchronize()
    res["load_s"] = time.perf_counter() - t0
    res["gpu_alloc_after_load"] = torch.cuda.memory_allocated()
    res["gpu_max_after_load"] = torch.cuda.max_memory_allocated()
    res["proc_after_load"] = proc_status()
    res["ssd_files_after_load"] = scan_files(ssd_path)
    try:
        res["model_info"] = model_layer_info(llm, llm.llm_engine.model_config.hf_config)
    except Exception as e:  # noqa
        res["model_info"] = {"source": "none", "note": str(e)}
    res["offloader_attrs"] = offloader_attrs()
    res["stage"] = "generate"
    dump()

    sp = SamplingParams(temperature=0, max_tokens=a.max_tokens)
    outs = llm.generate(PROMPTS, sp)
    res["token_ids"] = [list(o.outputs[0].token_ids) for o in outs]
    res["texts"] = [o.outputs[0].text for o in outs]
    res["stage"] = "decode"
    dump()

    sp2 = SamplingParams(temperature=0, max_tokens=DECODE_TOKENS, ignore_eos=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    outs2 = llm.generate(DECODE_PROMPTS, sp2)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t1
    ntok = sum(len(o.outputs[0].token_ids) for o in outs2)
    res["decode"] = {"batch": len(DECODE_PROMPTS), "max_tokens": DECODE_TOKENS,
                     "wall_s": dt, "gen_tokens": ntok, "tok_s": ntok / dt if dt else None}

    res["gpu_max_alloc_bytes"] = torch.cuda.max_memory_allocated()
    res["gpu_alloc_bytes"] = torch.cuda.memory_allocated()
    try:
        hs = torch.cuda.host_memory_stats()  # CachingHostAllocator = pin_memory=True tensors
        res["torch_pinned_host"] = {k: v for k, v in hs.items() if "bytes" in k}
        res["pinned_est_bytes"] = next((hs[k] for k in ("allocated_bytes.all.current",
                                                       "allocated_bytes.allocated",
                                                       "allocated_bytes.current") if k in hs), None)
    except Exception as e:  # noqa
        res["torch_pinned_host"] = {"error": str(e)}
    res["proc_after"] = proc_status()
    res["nvfs_after"] = nvfs_snapshot()
    res["nvfs_delta"] = nvfs_delta(res["nvfs_before"], res["nvfs_after"])
    m, v = nvfs_read_metric(res["nvfs_before"], res["nvfs_after"])
    res["nvfs_read_metric"] = {"name": m, "delta": v}
    nb, na = res["nvfs_before"], res["nvfs_after"]
    res["nvfs_bar1_map_n_delta"] = na.get("Bar1-map", {}).get("n", 0) - nb.get("Bar1-map", {}).get("n", 0)
    res["nvfs_mmap_n_delta"] = na.get("Mmap", {}).get("n", 0) - nb.get("Mmap", {}).get("n", 0)
    res["tiers_reported"] = None
    for ln in logs:
        mt = TIERS_RE.search(ln)
        if mt:
            res["tiers_reported"] = {"host_layers": int(mt.group(1)), "ssd_layers": int(mt.group(2)),
                                     "files": int(mt.group(3))}
    res["ssd_files"] = scan_files(ssd_path)
    res["ok"] = True
    res["stage"] = "done"
    dump()
    print(f"[worker {a.worker}] done: load {res['load_s']:.1f}s decode {dt:.2f}s "
          f"nvfs Bar1-map.n +{res['nvfs_bar1_map_n_delta']} Mmap.n +{res['nvfs_mmap_n_delta']} "
          f"files {res['ssd_files']['count']}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.environ.get("QA_NSYS") != "1":
        # Skip vLLM/NCCL teardown (can hang or spam); results are already on disk.
        os._exit(0)


# ----------------------------------------------------------------------------- driver
def collect_env_info():
    info = {"MemTotal": meminfo_total(), "nvfs_stats_head": []}
    try:
        info["rw_stats_enabled"] = open(
            "/sys/module/nvidia_fs/parameters/rw_stats_enabled").read().strip()
    except OSError:
        info["rw_stats_enabled"] = None
    try:
        info["nvfs_stats_head"] = open(NVFS_STATS).read().splitlines()[:6]
    except OSError:
        pass
    try:
        info["nvme_module"] = subprocess.run(["modinfo", "-n", "nvme"], capture_output=True,
                                             text=True).stdout.strip()
    except OSError:
        info["nvme_module"] = None
    try:
        gc = subprocess.run([GDSCHECK, "-p"], capture_output=True, text=True, timeout=60).stdout
        info["gdscheck"] = [ln.strip() for ln in gc.splitlines()
                            if re.search(r"NVMe\s+:|compat_mode|GDS release|nvidia_fs version|supports GDS",
                                         ln)]
        m = re.search(r"^\s*NVMe\s+:\s*(\w+)", gc, re.M)
        info["gds_nvme_supported"] = (m.group(1) == "Supported") if m else None
    except (OSError, subprocess.TimeoutExpired):
        info["gdscheck"] = []
        info["gds_nvme_supported"] = None
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                      "--format=csv,noheader"], capture_output=True,
                                     text=True).stdout.strip()
    except OSError:
        info["gpu"] = None
    return info


def gpu_mem():
    """(used_bytes, free_bytes, total_bytes) from nvidia-smi, or None."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        u, f, t = [int(x) * 2**20 for x in q.stdout.strip().split(",")]
        return u, f, t
    except Exception:  # noqa
        return None


def wait_gpu_free(max_wait_s):
    """The GPU is shared: block until >=0.5*total (+0.5GiB slack) is free, else warn."""
    t0 = time.time()
    warned = False
    while True:
        m = gpu_mem()
        if m is None:
            return
        used, free, total = m
        need = int(total * 0.5) + (512 << 20)
        if free >= need:
            if warned:
                print(f"   gpu free now ({gib(free)}), continuing", flush=True)
            return
        if time.time() - t0 > max_wait_s:
            print(f"   WARNING: only {gib(free)} GPU free (need ~{gib(need)}); starting anyway", flush=True)
            return
        if not warned:
            print(f"   waiting for GPU: {gib(used)} used by another process, need {gib(need)} free "
                  f"(up to {max_wait_s}s)", flush=True)
            warned = True
        time.sleep(10)


def tail_lines(path, n=30):
    try:
        with open(path, errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def parse_nsys_csv(text):
    """Extract H2D memcpy volume/count and cudaMemcpyAsync/cuFile-ish API rows."""
    out = {"h2d_mb": None, "h2d_count": None, "api": {}}
    section = None
    header = None
    for line in text.splitlines():
        if not line.strip():
            continue
        # nsys 2024.x: sections start with "Processing [x.sqlite] with [.../reports/<name>.py]..."
        m = re.search(r"reports/(\w+)\.py", line)
        if line.startswith("** ") or m:
            section = m.group(1) if m else line
            header = None
            continue
        if line.startswith("Processing") or line.startswith("SKIPPED") or line.startswith("Generating"):
            continue
        cells = [c.strip().strip('"') for c in line.split(",")]
        if header is None:
            header = cells
            continue
        row = dict(zip(header, cells))
        name = cells[-1]
        if section and "mem_size_sum" in section and "Host-to-Device" in name:
            try:
                out["h2d_mb"] = float(row.get("Total (MB)", "nan"))
                out["h2d_count"] = int(row.get("Count", "0"))
            except ValueError:
                pass
        if section and "cuda_api_sum" in section and re.search(r"Memcpy|cuFile|Memset", name):
            try:
                out["api"][name] = {"count": int(row.get("Num Calls", "0")),
                                    "total_ms": float(row.get("Total Time (ns)", "0")) / 1e6}
            except ValueError:
                pass
    return out


def run_driver(a):
    os.makedirs(RESULTS, exist_ok=True)
    env_info = collect_env_info()
    with open(os.path.join(RESULTS, "env.json"), "w") as f:
        json.dump(env_info, f, indent=1)
    memtotal = env_info["MemTotal"]
    print(f"== qa_ssd: arms={a.arms} nsys={a.nsys} keep={a.keep} results={RESULTS}")
    print(f"   host_fraction={a.host_fraction} (budget {gib(a.host_fraction * memtotal)} of "
          f"MemTotal {gib(memtotal)}), group_size={a.group_size}, num_in_group={a.num_in_group}, "
          f"prefetch_step={a.prefetch_step}, io_threads={a.io_threads}")
    print(f"   native-GDS proof = cuFile TRACE log classification ({CUFILE_TRACE_JSON}); "
          f"nvidia_fs rw_stats_enabled={env_info['rw_stats_enabled']} (Reads.n/Ops counters are info only)")
    if env_info.get("gds_nvme_supported") is False:
        print(f"   WARNING: gdscheck reports 'NVMe : Unsupported' (nvme module = {env_info.get('nvme_module')}). "
              f"cuFile will run in compat mode; the ssd-cufile native-GDS check cannot pass on this boot.")
    if not os.path.exists(CUFILE_TRACE_JSON):
        print(f"   WARNING: {CUFILE_TRACE_JSON} missing; cufile.log classification will be NONE")
    cufile_log = os.path.join(RESULTS, "cufile.log")  # cuFile writes its log into the worker CWD

    keep_env = a.keep or os.environ.get("KEEP") == "1"
    recs = {}
    for arm in a.arms:
        ssd_dir = os.path.join(RESULTS, arm)
        shutil.rmtree(ssd_dir, ignore_errors=True)
        os.makedirs(ssd_dir, exist_ok=True)
        out_json = os.path.join(RESULTS, f"{arm}.json")
        err_path = os.path.join(RESULTS, f"{arm}.err")
        for p in (out_json, err_path):
            if os.path.exists(p):
                os.remove(p)
        use_nsys = a.nsys and arm in SSD_ARMS
        cmd = ["timeout", "-k", "30", str(a.timeout)]
        if use_nsys:
            cmd += [NSYS, "profile", "-t", "cuda,nvtx,osrt", "--cuda-memory-usage=false",
                    "-o", os.path.join(RESULTS, arm), "--force-overwrite", "true"]
        cmd += [sys.executable, os.path.abspath(__file__), "--worker", arm, "--out", out_json,
                "--ssd-path", ssd_dir, "--host-fraction", str(a.host_fraction),
                "--group-size", str(a.group_size), "--num-in-group", str(a.num_in_group),
                "--prefetch-step", str(a.prefetch_step), "--io-threads", str(a.io_threads),
                "--max-tokens", str(a.max_tokens)]
        env = dict(os.environ, VLLM_USE_V2_MODEL_RUNNER="0", VLLM_ENABLE_V1_MULTIPROCESSING="0",
                   QA_NSYS="1" if use_nsys else "0")
        if arm in SSD_ARMS and os.path.exists(CUFILE_TRACE_JSON):
            env["CUFILE_ENV_PATH_JSON"] = CUFILE_TRACE_JSON
        arm_cufile_log = os.path.join(RESULTS, f"{arm}-cufile.log")
        for pth in (cufile_log, arm_cufile_log):
            if os.path.exists(pth):
                os.remove(pth)
        log_path = os.path.join(RESULTS, f"{arm}.log")
        stderr_path = os.path.join(RESULTS, f"{arm}.stderr.log")
        wait_gpu_free(a.gpu_wait)
        print(f"-- [{arm}] start{' (nsys)' if use_nsys else ''}: {' '.join(cmd[4:6] if use_nsys else cmd[4:6])} ...",
              flush=True)
        t0 = time.time()
        with open(log_path, "w") as lo, open(stderr_path, "w") as le:
            p = subprocess.run(cmd, cwd=RESULTS, env=env, stdout=lo, stderr=le)
        wall = time.time() - t0

        data = None
        try:
            with open(out_json) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = None
        rec = dict(data or {"arm": arm, "ok": False, "stage": "no-json"})
        rec.update({"exit_code": p.returncode, "timed_out": p.returncode in (124, 137),
                    "driver_wall_s": wall, "cmd": cmd, "nsys": None})
        crashed = p.returncode != 0 or not rec.get("ok")
        rec["crashed"] = crashed
        if crashed:
            lines = tail_lines(stderr_path, 30)
            with open(err_path, "w") as f:
                f.write(f"# arm={arm} exit_code={p.returncode} stage={rec.get('stage')} "
                        f"wall={wall:.0f}s{' TIMEOUT' if rec['timed_out'] else ''}\n")
                f.writelines(lines)
            rec["stderr_tail"] = [ln.rstrip("\n") for ln in lines]

        # cufile.log written during this arm (TRACE level for ssd arms) -> classify kernel path.
        if os.path.exists(cufile_log):
            os.replace(cufile_log, arm_cufile_log)
        rec["cufile_log"] = classify_cufile_log(arm_cufile_log, rec.get("nvfs_bar1_map_n_delta"))

        if use_nsys:
            rep = os.path.join(RESULTS, f"{arm}.nsys-rep")
            csv_path = os.path.join(RESULTS, f"{arm}-nsys.csv")
            if os.path.exists(rep):
                st = subprocess.run([NSYS, "stats", "--report",
                                     "cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum",
                                     "--format", "csv", rep], capture_output=True, text=True)
                with open(csv_path, "w") as f:
                    f.write(st.stdout)
                    if st.returncode != 0:
                        f.write("\n# nsys stats stderr:\n" + st.stderr)
                rec["nsys"] = dict(parse_nsys_csv(st.stdout), report=rep, csv=csv_path,
                                   stats_rc=st.returncode)
            else:
                rec["nsys"] = {"error": "no .nsys-rep produced", "report": rep}

        with open(out_json, "w") as f:
            json.dump(rec, f, indent=1)
        recs[arm] = rec

        if crashed:
            last = rec.get("stderr_tail", [""])[-1] if rec.get("stderr_tail") else ""
            print(f"   [{arm}] FAIL crashed exit={p.returncode} stage={rec.get('stage')} "
                  f"wall={wall:.0f}s -> {err_path}\n      last stderr: {last[:200]}", flush=True)
        else:
            pa, pb = rec["proc_after"], rec["proc_before"]
            pin = rec.get("pinned_est_bytes")
            fl = rec["ssd_files"]
            dec = rec["decode"]
            cl = rec["cufile_log"]
            print(f"   [{arm}] ok load={rec['load_s']:.1f}s gpu_max={gib(rec['gpu_max_alloc_bytes'])} "
                  f"rss={gib(pa['VmRSS'])}(+{gib(pa['VmRSS'] - pb['VmRSS'])}) lck={gib(pa.get('VmLck', 0))} "
                  f"torch_pinned={gib(pin) if isinstance(pin, int) else 'n/a'} "
                  f"nvfs[Bar1-map.n]=+{rec['nvfs_bar1_map_n_delta']} nvfs[Mmap.n]=+{rec['nvfs_mmap_n_delta']} "
                  f"files={fl['count']}/{gib(fl['bytes'])} "
                  f"decode4x{DECODE_TOKENS}={dec['wall_s']:.2f}s({dec['tok_s']:.0f}tok/s) "
                  f"cufile.log={cl['class']}(px_io={cl['px_io']} bounce_r={cl['bounce_read']} "
                  f"compat_notices={cl['compat_notices']} lines={cl['lines']})", flush=True)
            for ln in rec.get("offloader_log", []):
                if "PrefetchOffloader" in ln:
                    print(f"      log: {ln.split(': ', 1)[-1][:220]}")
        if rec.get("nsys"):
            ns = rec["nsys"]
            print(f"      nsys: H2D {ns.get('h2d_mb')} MB / {ns.get('h2d_count')} copies; "
                  f"api={ {k: v['count'] for k, v in ns.get('api', {}).items()} } -> {ns.get('csv')}")
        if not keep_env:
            shutil.rmtree(ssd_dir, ignore_errors=True)

    # ------------------------------------------------------------------ checks
    print("== checks")
    fails = 0

    def verdict(name, ok, detail=""):
        nonlocal fails
        fails += 0 if ok else 1
        print(f"   {'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")

    base = recs.get("baseline")
    if base is None:
        try:
            with open(os.path.join(RESULTS, "baseline.json")) as f:
                base = json.load(f)
            print(f"   (baseline token ids loaded from previous run: {os.path.join(RESULTS, 'baseline.json')})")
        except (OSError, ValueError):
            base = None
    base_ids = base.get("token_ids") if base and not base.get("crashed") else None
    if base_ids is None:
        print("   baseline token ids unavailable -> every token comparison FAILs")

    for arm in a.arms:
        r = recs[arm]
        if r.get("crashed"):
            verdict(f"{arm}: token ids == baseline", False, f"arm crashed (see {arm}.err)")
            continue
        if base_ids is None:
            verdict(f"{arm}: token ids == baseline", False, "no baseline")
            continue
        ids = r.get("token_ids")
        ok = ids == base_ids
        detail = ""
        if not ok:
            bad = [i for i in range(len(PROMPTS)) if ids[i] != base_ids[i]]
            detail = f"mismatch prompts {bad}: {ids[bad[0]][:8]} vs {base_ids[bad[0]][:8]}"
        verdict(f"{arm}: token ids == baseline", ok, detail)

    if "ssd-cufile" in recs:
        r = recs["ssd-cufile"]
        name = "ssd-cufile: native GDS path (cufile.log class INTERNAL-BOUNCE or DIRECT)"
        if r.get("crashed"):
            verdict(name, False, "arm crashed")
        else:
            cl = r["cufile_log"]
            detail = (f"class={cl['class']} px_io={cl['px_io']} (read={cl['px_read']} write={cl['px_write']}) "
                      f"bounce_read={cl['bounce_read']} bounce_write={cl['bounce_write']} "
                      f"Bar1-map.n=+{r['nvfs_bar1_map_n_delta']} Mmap.n=+{r['nvfs_mmap_n_delta']} "
                      f"compat_notices={cl['compat_notices']} log={os.path.basename(cl['path'])} "
                      f"({cl['lines']} lines)")
            if cl["class"] == "COMPAT-POSIX" and env_info.get("gds_nvme_supported") is False:
                detail += " [gdscheck: NVMe Unsupported on this boot -> compat mode expected]"
            verdict(name, cl["class"] in ("INTERNAL-BOUNCE", "DIRECT"), detail)
    if "ssd-posix" in recs:
        r = recs["ssd-posix"]
        name = "ssd-posix: no cuFile use (cufile.log absent/empty)"
        if r.get("crashed"):
            verdict(name, False, "arm crashed")
        else:
            cl = r["cufile_log"]
            verdict(name, cl["class"] in ("ABSENT", "EMPTY"),
                    f"class={cl['class']} lines={cl['lines']} px_io={cl['px_io']} bounce_read={cl['bounce_read']} "
                    f"Bar1-map.n=+{r['nvfs_bar1_map_n_delta']} Mmap.n=+{r['nvfs_mmap_n_delta']}")
    for arm in SSD_ARMS:
        if arm not in recs:
            continue
        r = recs[arm]
        if r.get("crashed"):
            verdict(f"{arm}: files created >= offloaded - cpu layers", False, "arm crashed")
            continue
        mi = r.get("model_info") or {}
        L, lb = mi.get("num_layers"), mi.get("layer_bytes")
        if not L or not lb:
            verdict(f"{arm}: files created >= offloaded - cpu layers", False,
                    f"layer size unknown ({mi.get('note')})")
            continue
        offloaded = (L // a.group_size) * a.num_in_group
        cpu_layers = min(offloaded, int(math.floor(a.host_fraction * memtotal / lb)))
        need = offloaded - cpu_layers
        fl = r["ssd_files"]
        verdict(f"{arm}: files created >= offloaded - cpu layers", fl["count"] >= need and need > 0,
                f"files={fl['count']} ({gib(fl['bytes'])}) need>={need} "
                f"[layers={L}, layer={lb / 2**20:.0f}MiB, offloaded={offloaded}, cpu_budget={cpu_layers}; "
                f"offloader reported {r.get('tiers_reported')}]")

    summary = {"arms": a.arms, "fails": fails, "nsys": a.nsys, "env": env_info,
               "checks_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"== {'ALL PASS' if fails == 0 else f'{fails} FAIL'}  (results in {RESULTS})")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", default=ARMS, help=f"subset of {ARMS} (default: all)")
    ap.add_argument("--nsys", action="store_true", help="profile ssd-* arms under nsys")
    ap.add_argument("--keep", action="store_true", help="keep SSD files (also KEEP=1)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per arm")
    ap.add_argument("--gpu-wait", type=int, default=600,
                    help="seconds to wait for >=50%% GPU memory free before each arm (GPU is shared)")
    ap.add_argument("--host-fraction", type=float, default=0.005)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--num-in-group", type=int, default=3)
    ap.add_argument("--prefetch-step", type=int, default=1)
    ap.add_argument("--io-threads", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--worker", metavar="ARM", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--ssd-path", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        assert a.worker in ARMS and a.out and a.ssd_path
        run_worker(a)
        return 0
    bad = [x for x in a.arms if x not in ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; choose from {ARMS}")
    return run_driver(a)


if __name__ == "__main__":
    sys.exit(main())
