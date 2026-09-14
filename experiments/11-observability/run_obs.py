"""관측 체계(lib/obs)를 붙인 3단계 워크로드 러너.
   단계: cold_fill(문서 N개, 질문 0) → settle → reverse_retrieve(역순, 질문 1) → final_settle
   가중치는 prefetch 오프로더(CPU/SSD 티어), KV는 in-tree CuFileFsSpec(native cuFile)으로 SSD. 외부 파이썬 전송 코드 없음. 엔진 step을 직접 돌려 step 단위 기록.
   산출물(RUN_DIR): environment.txt, capacity.json, events.jsonl(KV IO와 phase 마커), requests.jsonl(요청별 시각),
     steps.jsonl, tier_samples.jsonl(nvidia-fs·프로세스·캐시 파일 1초), hostmon 파일들, result.json, summary.csv
   usage: python run_obs.py --run-dir DIR --model facebook/opt-13b --n-docs 32 --kv-batch 6 ..."""
import argparse, json, os, subprocess, sys, threading, time
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--model", default="facebook/opt-13b")
ap.add_argument("--n-docs", type=int, default=32)
ap.add_argument("--leval-workload", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03-leval", "workload.json"))
ap.add_argument("--kv-transport", default="cufile", choices=["cufile", "none", "lmcache"],
                help="cufile: in-tree CuFileFsSpec(native). none: 재계산. lmcache: LMCache MP 서버의 GDS L1(GPU↔NVMe 직접)")
ap.add_argument("--lmcache-l1-gb", type=float, default=40.0, help="lmcache: GDS L1 슬랩 크기(GB)")
ap.add_argument("--lmcache-port", type=int, default=5555)
ap.add_argument("--register-tensors", action="store_true", help="KV 텐서를 cuFileBufRegister(BAR1 안에 들어갈 때만)")
ap.add_argument("--kv-batch", type=int, default=4, help="GPU KV 예산 = 요청 N개분 × 1.15")
ap.add_argument("--kv-threads", type=int, default=4)
ap.add_argument("--kv-block", type=int, default=64)
ap.add_argument("--host-weight-fraction", type=float, default=None, help="오프로드 가중치 대비 CPU 비율(환산). 미지정이면 --host-ram-fraction")
ap.add_argument("--host-ram-fraction", type=float, default=None, help="host memory(RAM 전체) 대비 비율을 오프로더에 그대로 전달. 둘 다 없으면 오프로더 기본값 0.3")
ap.add_argument("--pure", action="store_true", help="GPU KV 예산과 block_size를 vLLM 기본에 맡김(--kv-batch, --kv-block 무시)")
ap.add_argument("--prefetch-step", type=int, default=1)
ap.add_argument("--io-threads", type=int, default=4)
ap.add_argument("--gpu-util", type=float, default=0.9)
ap.add_argument("--max-model-len", type=int, default=2048)
ap.add_argument("--decode-tokens", type=int, default=8)
ap.add_argument("--settle-sec", type=float, default=15.0)
ap.add_argument("--final-settle-sec", type=float, default=15.0)
ap.add_argument("--poll-sleep-ms", type=float, default=1.0)
ap.add_argument("--ssd-root", required=True); ap.add_argument("--kv-root", required=True)
ap.add_argument("--no-monitors", action="store_true")
args = ap.parse_args()
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
R = os.path.abspath(args.run_dir)
if os.path.exists(os.path.join(R, "result.json")) or os.path.exists(os.path.join(R, "workload.exitcode")):
    sys.exit(f"RUN_DIR에 이미 런이 있음: {R}")
os.makedirs(R, exist_ok=True)
if os.path.isdir(args.kv_root) and os.listdir(args.kv_root):
    sys.exit(f"kv-root가 비어 있지 않음: {args.kv_root}")

# ---- 환경, 용량 사전 검사 ----
subprocess.run(["bash", os.path.join(ROOT, "lib", "obs", "envinfo.sh"), R, args.leval_workload], check=False)
from transformers import AutoConfig
hc = AutoConfig.from_pretrained(args.model); d_model, n_layer = int(hc.hidden_size), int(hc.num_hidden_layers)
kv_req = 2 * n_layer * d_model * 2 * args.max_model_len
kv_gib = round(args.kv_batch * kv_req * 1.15 / 2**30, 2)
w_off = 12 * d_model * d_model * 2 * n_layer
mem_total = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) * 1024
if args.host_weight_fraction is not None:
    host_fraction = round(args.host_weight_fraction * w_off * (1.03 if args.host_weight_fraction >= 1.0 else 1.0) / mem_total, 4)
elif args.host_ram_fraction is not None:
    host_fraction = args.host_ram_fraction
else:
    host_fraction = 0.3
if args.pure:
    kv_gib = None
plan = subprocess.run([sys.executable, os.path.join(ROOT, "lib", "obs", "plan.py"), "--model", args.model, "--tokens-per-request", str(args.max_model_len),
                       "--requests", str(args.n_docs), "--gpu-kv-gib", str(kv_gib if kv_gib else 0), "--cache-dir", args.kv_root,
                       "--host-gib", str(host_fraction * mem_total / 2**30), "--out", os.path.join(R, "capacity.json")], capture_output=True, text=True)
print(plan.stdout, plan.stderr, flush=True)

from obs.events import Events
EV = Events(os.path.join(R, "events.jsonl"))
EV.emit("run_start", args=vars(args), kv_gib=kv_gib, host_fraction=host_fraction, pid=os.getpid())

# ---- 외부 모니터 ----
mons = []
if not args.no_monitors:
    mons.append(subprocess.Popen(["bash", os.path.join(ROOT, "lib", "obs", "hostmon.sh")], env=dict(os.environ, RUN_DIR=R, STOP_FILE=os.path.join(R, "hostmon.stop"))))
    mons.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "lib", "obs", "observe.py"), "--run-dir", R, "--pid", str(os.getpid()), "--cache-dir", args.kv_root]))
def stop_monitors():
    open(os.path.join(R, "hostmon.stop"), "w").close()
    for m in mons:
        try: m.terminate(); m.wait(timeout=10)
        except Exception: pass

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
kw = dict(offload_backend="prefetch", offload_group_size=n_layer, offload_num_in_group=n_layer, offload_prefetch_step=args.prefetch_step,
          offload_ssd_path=args.ssd_root, offload_host_fraction=host_fraction, offload_ssd_transport="cufile",
          offload_ssd_io_threads=args.io_threads, offload_ssd_ring_mb=0)
matched = [0]
LMC = None
if args.kv_transport == "lmcache":
    # LMCache MP 서버를 별도 프로세스로. --gds-l1-path 가 있으면 DRAM 층이 꺼지고 cuFile로 GPU↔NVMe 직접
    import socket
    os.makedirs(args.kv_root, exist_ok=True)
    lmc_cmd = ["lmcache", "server", "--host", "127.0.0.1", "--port", str(args.lmcache_port), "--chunk-size", "256",
               "--l1-size-gb", str(args.lmcache_l1_gb), "--gds-l1-path", args.kv_root, "--gds-l1-backend", "cufile",
               "--gds-l1-use-direct-io", "--max-workers", str(args.kv_threads), "--eviction-policy", "LRU"]
    open(os.path.join(R, "lmcache_command.txt"), "w").write(" ".join(lmc_cmd) + "\n")
    LMC = subprocess.Popen(lmc_cmd, stdout=open(os.path.join(R, "lmcache.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(600):
        if LMC.poll() is not None: sys.exit("LMCache 서버가 종료됨. lmcache.log 확인")
        try:
            socket.create_connection(("127.0.0.1", args.lmcache_port), timeout=0.5).close(); break
        except OSError: time.sleep(0.5)
    else: sys.exit("LMCache 서버 포트 대기 시간 초과")
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both",
        kv_connector_extra_config={"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": args.lmcache_port})
    import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector as lmcc
    for _n in dir(lmcc):
        _c = getattr(lmcc, _n)
        if isinstance(_c, type) and hasattr(_c, "get_num_new_matched_tokens") and "LMCache" in _n:
            _orig = _c.get_num_new_matched_tokens
            def _mk(_o):
                def _w(self, request, n):
                    r = _o(self, request, n); matched[0] += (r[0] or 0) if isinstance(r, tuple) else (r or 0); return r
                return _w
            _c.get_num_new_matched_tokens = _mk(_orig)
elif args.kv_transport != "none":
    extra = {"spec_name": "CuFileFsSpec", "cufile_fs_root_dir": args.kv_root, "cufile_fs_register_tensors": str(args.register_tensors),
             "cufile_fs_read_threads": args.kv_threads, "cufile_fs_write_threads": args.kv_threads}
    if not args.pure: extra["block_size"] = args.kv_block
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both", kv_connector_extra_config=extra)
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
    _gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
    def _gmw(self, request, n):
        r = _gm(self, request, n); matched[0] += r[0] or 0; return r
    osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

try:
    t0 = time.time(); EV.phase("model_load_begin")
    if kv_gib: kw["kv_cache_memory_bytes"] = int(kv_gib * 2**30)
    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              enforce_eager=True, **kw)
    EV.phase("model_load_end", load_s=round(time.time() - t0, 1))
    from vllm.model_executor.offloader.base import get_offloader
    off = get_offloader()
    tiers = dict(n_modules=len(off.module_offloaders), n_ssd=sum(1 for m in off.module_offloaders if m.mode == "ssd"),
                 host_tier_gib=round(off.host_tier_bytes / 2**30, 2), ssd_tier_gib=round(off.ssd_tier_bytes / 2**30, 2))
    EV.emit("tiers", **tiers)
    KVW = None
    if args.kv_transport == "cufile":
        import vllm.v1.kv_offload.cufile_fs.spec as cfs
        KVW = cfs.LAST_WORKER
        EV.emit("kv_worker", registered_tensors=KVW.native.registered, register_err=KVW.native.register_err, chunk_bytes=KVW.chunk_bytes)
        _ss, _sl, _gf = KVW.submit_store, KVW.submit_load, KVW.get_finished
        _jobs = {}
        def submit_store(job_id, src, dst):
            _jobs[job_id] = ("w", time.monotonic(), len(dst.paths) * KVW.chunk_bytes)
            EV.emit("kv_w_submit", job=job_id, chunks=len(dst.paths), bytes=len(dst.paths) * KVW.chunk_bytes); return _ss(job_id, src, dst)
        def submit_load(job_id, src, dst):
            _jobs[job_id] = ("r", time.monotonic(), len(src.paths) * KVW.chunk_bytes)
            EV.emit("kv_r_submit", job=job_id, chunks=len(src.paths), bytes=len(src.paths) * KVW.chunk_bytes); return _sl(job_id, src, dst)
        def get_finished():
            out = _gf()
            for r in out:
                op, t, nb = _jobs.pop(r.job_id, ("?", time.monotonic(), 0))
                EV.emit(f"kv_{op}_end", job=r.job_id, ok=r.success, bytes=nb, dur_ms=round((time.monotonic() - t) * 1e3, 1))
            return out
        KVW.submit_store, KVW.submit_load, KVW.get_finished = submit_store, submit_load, get_finished
    def kvstat():
        return KVW.stats() if KVW is not None else {}
    W = json.load(open(args.leval_workload)); docs = W["docs"][:args.n_docs]
    def prompt(i, q):
        d = docs[i]; return d["prefix"] + W["delim_tokens"] + d["questions"][q % len(d["questions"])]["tokens"]
    sp = SamplingParams(max_tokens=args.decode_tokens, temperature=0, ignore_eos=True)
    eng = llm.llm_engine
    steps_f = open(os.path.join(R, "steps.jsonl"), "a", buffering=1); reqs_f = open(os.path.join(R, "requests.jsonl"), "a", buffering=1)
    def wstat():
        s = getattr(off, "ssd_tier", None); return (s.stats["reads"], s.stats["bytes"]) if s is not None else (0, 0)
    def drain():
        t = time.monotonic()
        while KVW is not None and time.monotonic() - t < 600:
            KVW.get_finished()
            st_ = KVW.stats()
            if not KVW._pending and st_["outstanding_writes"] == 0 and st_["outstanding_reads"] == 0: break
            time.sleep(0.005)
        return round(time.monotonic() - t, 3)
    def run_phase(name, order, q):
        matched[0] = 0
        EV.phase(name, requests=len(order))
        st = {}
        for i in order:
            rid = f"{name}-{i}"; toks = prompt(i, q)
            eng.add_request(rid, {"prompt_token_ids": toks}, sp)
            st[rid] = dict(doc=i, tokens=len(toks), submit_mono=time.monotonic(), submit_wall=time.time(), first_mono=None, finish_mono=None, ntok=0)
        tS = time.monotonic()
        while any(v["finish_mono"] is None for v in st.values()):
            if time.monotonic() - tS > 3 * 3600: EV.emit("warn", msg=f"{name} 3시간 초과"); break
            pre_w = wstat(); pre_k = kvstat(); s0 = time.monotonic(); w0 = time.time()
            torch.cuda.nvtx.range_push("step"); outs = eng.step(); torch.cuda.nvtx.range_pop()
            s1 = time.monotonic(); post_w = wstat(); post_k = kvstat()
            got_first = False
            for o in outs:
                v = st.get(o.request_id)
                if v is None: continue
                if v["first_mono"] is None and o.outputs and o.outputs[0].token_ids: v["first_mono"] = s1; v["first_wall"] = time.time(); got_first = True
                if o.outputs: v["ntok"] = len(o.outputs[0].token_ids); v["ids"] = list(o.outputs[0].token_ids)
                if o.finished:
                    v["finish_mono"] = s1; v["finish_wall"] = time.time()
                    reqs_f.write(json.dumps(dict(phase=name, rid=o.request_id, **v)) + "\n")
            n_out = len(outs); n_tok = sum(len(o.outputs[0].token_ids) for o in outs if o.outputs)
            if s1 - s0 > 0.3 or n_out:
                kd = {k: post_k[k] - pre_k[k] for k in ("reads", "writes", "read_bytes", "write_bytes")} if post_k else {}
                steps_f.write(json.dumps(dict(phase=name, mono0=round(s0, 4), mono1=round(s1, 4), wall0=w0, dur=round(s1 - s0, 4),
                    kind="prefill" if (got_first or (n_out == 0 and any(v["first_mono"] is None for v in st.values()))) else "decode",
                    n_out=n_out, n_tok=n_tok, w_reads=post_w[0] - pre_w[0], w_bytes=post_w[1] - pre_w[1],
                    kv_out_r=post_k.get("outstanding_reads", 0), kv_out_w=post_k.get("outstanding_writes", 0), **{"kv_" + k: v for k, v in kd.items()})) + "\n")
            elif args.poll_sleep_ms and not outs:
                time.sleep(args.poll_sleep_ms / 1000.0)
        d = drain()
        EV.phase(name + "_end", wall_s=round(time.monotonic() - tS, 3), matched_tokens=matched[0], drain_s=d)
        return dict(wall_s=round(time.monotonic() - tS, 3), matched=matched[0], drain_s=d,
                    ttft=[round(v["first_mono"] - v["submit_mono"], 3) for v in st.values() if v["first_mono"]],
                    e2e=[round(v["finish_mono"] - v["submit_mono"], 3) for v in st.values() if v["finish_mono"]])
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        kv_alloc_gib = round(cc.num_gpu_blocks * cc.block_size * (2 * n_layer * d_model * 2) / 2**30, 2)
    except Exception: kv_alloc_gib = kv_gib
    res = dict(args=vars(args), kv_gib=kv_gib, kv_alloc_gib=kv_alloc_gib, host_fraction=host_fraction, tiers=tiers, phases={})
    N = len(docs)
    res["phases"]["cold_fill"] = run_phase("cold_fill", list(range(N)), 0)
    EV.phase("settle"); time.sleep(args.settle_sec)
    res["phases"]["reverse_retrieve"] = run_phase("reverse_retrieve", list(reversed(range(N))), 1)
    EV.phase("final_settle"); time.sleep(args.final_settle_sec)
    ks = kvstat()
    res["kv_io"] = dict(read_n=ks.get("reads", 0), read_gib=round(ks.get("read_bytes", 0) / 2**30, 2), write_n=ks.get("writes", 0),
                        write_gib=round(ks.get("write_bytes", 0) / 2**30, 2), read_busy_s=round(ks.get("read_busy_ns", 0) / 1e9, 1),
                        write_busy_s=round(ks.get("write_busy_ns", 0) / 1e9, 1), errors=ks.get("errors", 0), registered_tensors=ks.get("registered_tensors", 0))
    res["gpu_max_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["weight_ssd_reads"], res["weight_ssd_gib"] = wstat()[0], round(wstat()[1] / 2**30, 2)
    if args.kv_transport != "none" and os.path.isdir(args.kv_root):
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(args.kv_root) for f in fs]
        res["kv_files"], res["kv_bytes_gib"] = len(files), round(sum(os.path.getsize(f) for f in files) / 2**30, 3)
    json.dump(res, open(os.path.join(R, "result.json"), "w"), indent=1)
    EV.phase("complete")
    open(os.path.join(R, "workload.exitcode"), "w").write("0\n")
    print("RESULT", json.dumps({k: res[k] for k in ("tiers", "kv_io", "gpu_max_gib")}), flush=True)
    for ph, v in res["phases"].items(): print(f"  {ph}: wall {v['wall_s']} s, ttft median {sorted(v['ttft'])[len(v['ttft'])//2] if v['ttft'] else None}, matched {v['matched']}", flush=True)
    try: llm.llm_engine.engine_core.shutdown()
    except Exception as e: print("shutdown:", e)
except BaseException as e:
    EV.emit("error", err=repr(e)); open(os.path.join(R, "workload.exitcode"), "w").write("1\n"); raise
finally:
    stop_monitors()
    if LMC is not None:
        try: LMC.terminate(); LMC.wait(timeout=30)
        except Exception: LMC.kill()
    subprocess.run([sys.executable, os.path.join(ROOT, "lib", "obs", "summarize.py"), R], stdout=open(os.path.join(R, "summary.csv"), "w"), check=False)
