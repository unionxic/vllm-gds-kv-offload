"""09-phase: KV 읽기 비용을 구간별로 분리 측정.
   기존 러너는 generate 두 번의 벽시계 차이로 decode 시간을 구했다. 그 방식은 두 번째 호출 안의 prefill이
   decode 몫으로 새어 들어갈 수 있어, "decode가 느려졌다"를 KV 읽기 탓으로 돌릴 근거가 되지 못한다.
   여기서는 엔진 step을 직접 돌려 다음을 측정한다.
     구간1 요청 제출 → KV 로드 완료(읽기 outstanding 이 0으로 복귀)
     구간2 KV 로드 완료 → 첫 토큰
     구간3 첫 토큰 → 마지막 토큰
   각 step 을 prefill 단계(아직 첫 토큰이 안 나온 요청이 있음)와 decode 단계로 분류하고,
   step 마다 가중치 SSD 읽기와 KV 읽기/쓰기의 바이트·건수·최대 outstanding 을 기록한다.
   decode 단계에 KV 읽기 바이트가 0 이면 "decode 손해"는 경합이 아니라 계산 방식의 산물이다.
   라운드 2 시작 전에 라운드 1 의 store 가 전부 끝났는지 확인하고 배출에 걸린 시간을 따로 남긴다.
   NVTX 구간을 붙여 nsys 로 중첩을 눈으로 확인할 수 있다.
   usage: python run_phase_66b.py --kv-transport cufile|none --tag ..."""
import argparse, json, os, random, resource, sys, threading, time
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="facebook/opt-66b")
ap.add_argument("--weight-transport", default="cufile", choices=["cufile", "posix"])
ap.add_argument("--kv-transport", default="cufile", choices=["cufile", "posix", "none", "cufile_staged", "posix_staged", "cufile_q8"])
ap.add_argument("--staging-policy", default="block", choices=["block", "skip", "cpu_fallback", "value"])
ap.add_argument("--staging-slots", type=int, default=6)
ap.add_argument("--staging-writers", type=int, default=4)
ap.add_argument("--cpu-fallback-slots", type=int, default=8)
ap.add_argument("--value-mode", default="seen_twice", choices=["random_skip", "seen_twice", "value_density", "oracle"])
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--hot-prompts", type=int, default=None,
                help="라운드 간 동일하게 반복되는 프롬프트 수. 나머지는 라운드마다 새 토큰열(1회성). 기본=전부 반복")
ap.add_argument("--host-fraction", type=float, default=0.85)
ap.add_argument("--group-size", type=int, default=64)
ap.add_argument("--num-in-group", type=int, default=62)
ap.add_argument("--prefetch-step", type=int, default=1)
ap.add_argument("--io-threads", type=int, default=4)
ap.add_argument("--ring-mb", type=int, default=0)
ap.add_argument("--gpu-util", type=float, default=0.9)
ap.add_argument("--kv-cache-gib", type=float, default=5.5)
ap.add_argument("--max-model-len", type=int, default=640)
ap.add_argument("--n-prompts", type=int, default=16)
ap.add_argument("--prefix-tokens", type=int, default=448)
ap.add_argument("--tail-tokens", type=int, default=32)
ap.add_argument("--decode-tokens", type=int, default=8)
ap.add_argument("--kv-threads", type=int, default=4)
ap.add_argument("--kv-block", type=int, default=64)
ap.add_argument("--ssd-root", default=os.path.expanduser("~/experiments/vllm-gds-kv/results/weight-offload/ssd-66b"))
ap.add_argument("--kv-root", default=os.path.expanduser("~/experiments/vllm-gds-kv/results/combined/kv-66b"))
ap.add_argument("--out-dir", default=None)
ap.add_argument("--tag", default="run")
ap.add_argument("--host-weight-fraction", type=float, default=None,
                help="오프로드되는 가중치 중 CPU에 둘 비율. 지정하면 --host-fraction(RAM 전체 대비)을 모델 크기로부터 환산")
ap.add_argument("--kv-batch", type=int, default=0,
                help="GPU KV 예산을 '요청 N개분(max_model_len 토큰) × 1.15'로 모델 크기에서 계산. 0이면 --kv-cache-gib 사용")
ap.add_argument("--poll-sleep-ms", type=float, default=0.0,
                help="step이 아무 출력도 내지 않은 폴링이면 이만큼 잠들어 GIL을 KV 로드 스레드에 양보(검증용)")
ap.add_argument("--prompt-source", default="random", choices=["random", "leval"],
                help="leval: 03-leval/workload.json의 실제 문서 프리픽스 + 라운드별 다른 질문")
ap.add_argument("--leval-workload", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03-leval", "workload.json"))
args = ap.parse_args()
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "lib"))
OUT = args.out_dir or os.path.join(ROOT, "results", "kv-policy"); os.makedirs(OUT, exist_ok=True)

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def sysmem():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        if k in ("MemTotal", "MemAvailable", "AnonPages", "Cached"): d[k] = int(v.split()[0]) * 1024
    return d

# ---- KV IO 계측: 모든 read_chunk/write_chunk 에 시작·종료 시각과 outstanding 을 붙인다 ----
IO = []                     # (t0, t1, bytes, "r"|"w")
LK = threading.Lock()
OUTS = {"r": 0, "w": 0, "r_max": 0, "w_max": 0}
def _wrap(cls, name, op):
    fn = getattr(cls, name)
    def inner(self, path, spans, chunk_bytes):
        nb = sum(s[2] for s in spans)
        with LK:
            OUTS[op] += 1; OUTS[op + "_max"] = max(OUTS[op + "_max"], OUTS[op])
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(f"kv_{op}")
        try:
            return fn(self, path, spans, chunk_bytes)
        finally:
            torch.cuda.nvtx.range_pop()
            t1 = time.perf_counter()
            with LK:
                OUTS[op] -= 1; IO.append((t0, t1, nb, op))
    setattr(cls, name, inner)

derived = {}
if args.host_weight_fraction is not None or args.kv_batch:
    from transformers import AutoConfig
    hc = AutoConfig.from_pretrained(args.model)
    d_model, n_layer = int(hc.hidden_size), int(hc.num_hidden_layers)
    # 디코더 층 가중치 = (qkv+out 4d² + fc1+fc2 8d²) × 2바이트. opt-2.7b 실측 4.69 GiB, opt-66b 62층 117.7 GiB와 일치
    w_layer = 12 * d_model * d_model * 2
    w_off = w_layer * n_layer * args.num_in_group / args.group_size
    if args.host_weight_fraction is not None:
        mem_total = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) * 1024
        # 층 바이트 추정이 실제보다 살짝 작을 수 있어 1.0(전부 CPU)이 한 층을 SSD로 흘리지 않도록 3% 여유
        margin = 1.03 if args.host_weight_fraction >= 1.0 else 1.0
        args.host_fraction = round(args.host_weight_fraction * w_off * margin / mem_total, 4)
        derived["host_fraction_from_weight"] = args.host_fraction
    if args.kv_batch:
        kv_req = 2 * n_layer * d_model * 2 * args.max_model_len
        args.kv_cache_gib = round(args.kv_batch * kv_req * 1.15 / 2**30, 2)
        derived["kv_per_request_gib"] = round(kv_req / 2**30, 3)
    derived.update(offloaded_weight_gib=round(w_off / 2**30, 2), d_model=d_model, n_layer=n_layer)
    print("derived:", derived, flush=True)
kw = dict(offload_backend="prefetch", offload_group_size=args.group_size, offload_num_in_group=args.num_in_group,
          offload_prefetch_step=args.prefetch_step, offload_ssd_path=args.ssd_root, offload_host_fraction=args.host_fraction,
          offload_ssd_transport=args.weight_transport, offload_ssd_io_threads=args.io_threads, offload_ssd_ring_mb=args.ring_mb)
matched = [0]
if args.kv_transport != "none":
    import shutil; shutil.rmtree(args.kv_root, ignore_errors=True)
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
        kv_connector_extra_config={"spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
            "expfs_root_dir": args.kv_root, "expfs_transport": args.kv_transport,
            "expfs_read_threads": args.kv_threads, "expfs_write_threads": args.kv_threads, "block_size": args.kv_block})
    if args.kv_transport.endswith("_staged"):
        kw["kv_transfer_config"].kv_connector_extra_config.update({
            "expfs_staging_policy": args.staging_policy, "expfs_staging_slots": args.staging_slots,
            "expfs_staging_writers": args.staging_writers, "expfs_cpu_fallback_slots": args.cpu_fallback_slots})
    import expfs
    for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport, expfs.StagedCuFileTransport, expfs.CuFileQ8Transport):
        for nm, op in (("read_chunk", "r"), ("write_chunk", "w")):
            if hasattr(cls, nm): _wrap(cls, nm, op)
    # staged 의 ring 경유 쓰기는 write_slot 으로 나간다
    if hasattr(expfs.StagedCuFileTransport, "write_slot"):
        _ws0 = expfs.StagedCuFileTransport.write_slot
        def _ws(self, slot, path, kind="ring"):
            with LK:
                OUTS["w"] += 1; OUTS["w_max"] = max(OUTS["w_max"], OUTS["w"])
            t0 = time.perf_counter()
            try: return _ws0(self, slot, path, kind)
            finally:
                t1 = time.perf_counter()
                with LK: OUTS["w"] -= 1; IO.append((t0, t1, self.chunk_bytes, "w"))
        expfs.StagedCuFileTransport.write_slot = _ws
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
    _gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
    def _gmw(self, request, n):
        r = _gm(self, request, n); matched[0] += r[0] or 0; return r
    osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

res = dict(args=vars(args), derived=derived, sysmem_start=sysmem())
t0 = time.time()
llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
          kv_cache_memory_bytes=int(args.kv_cache_gib * 2**30), enforce_eager=True, **kw)
res["load_s"] = round(time.time() - t0, 1)
if args.kv_transport != "none" and args.staging_policy == "value":
    import value_admission
    value_admission.install(sys.modules["expfs"].LAST_WORKER.transport, args.value_mode)
from vllm.model_executor.offloader.base import get_offloader
off = get_offloader()
res["tiers"] = dict(n_modules=len(off.module_offloaders), n_ssd=sum(1 for m in off.module_offloaders if m.mode == "ssd"),
                    host_tier_gib=round(off.host_tier_bytes / 2**30, 2), ssd_tier_gib=round(off.ssd_tier_bytes / 2**30, 2),
                    gpu_resident_layers=args.group_size - args.num_in_group)
def wstat():
    s = getattr(off, "ssd_tier", None)
    return (s.stats["reads"], s.stats["bytes"]) if s is not None else (0, 0)

vocab_hi = 50000
HOT = args.n_prompts if args.hot_prompts is None else args.hot_prompts
LEVAL = json.load(open(args.leval_workload)) if args.prompt_source == "leval" else None
def make_prompts(rnd):
    # leval: 문서 i의 프리픽스 1920토큰 + 구분자 + 질문(rnd-1). 라운드가 바뀌면 질문만 바뀌어 프리픽스가 적중
    if LEVAL is not None:
        out = []
        for i in range(args.n_prompts):
            d = LEVAL["docs"][i]
            q = d["questions"][(rnd - 1) % len(d["questions"])]
            out.append(d["prefix"] + LEVAL["delim_tokens"] + q["tokens"])
        return out
    # 앞 HOT 개는 라운드가 바뀌어도 같은 토큰열(재사용), 나머지는 라운드마다 새 토큰열(1회성)
    out = []
    for i in range(args.n_prompts):
        seed = 500 + i if i < HOT else 10_000 * rnd + i
        out.append([2] + [random.Random(seed).randrange(4, vocab_hi) for _ in range(args.prefix_tokens + args.tail_tokens - 1)])
    return out
sp = SamplingParams(max_tokens=args.decode_tokens, temperature=0, ignore_eos=True)
eng = llm.llm_engine

def kv_snapshot():
    with LK:
        n = len(IO)
        rb = sum(x[2] for x in IO if x[3] == "r"); wb = sum(x[2] for x in IO if x[3] == "w")
        rn = sum(1 for x in IO if x[3] == "r"); wn = sum(1 for x in IO if x[3] == "w")
        return n, rn, rb, wn, wb, OUTS["r"], OUTS["w"]

def drain_stores(limit=300.0):
    """라운드 경계에서 남은 store 가 전부 끝날 때까지 대기하고 걸린 시간과 잔여 건수를 돌려준다."""
    w = getattr(sys.modules.get("expfs"), "LAST_WORKER", None) if args.kv_transport != "none" else None
    t = time.perf_counter()
    while time.perf_counter() - t < limit:
        with LK: out_w = OUTS["w"]
        pend = len(getattr(w, "_pending", ())) if w is not None else 0
        if out_w == 0 and pend == 0: break
        if w is not None:
            try: w.get_finished()
            except Exception: pass
        time.sleep(0.002)
    with LK: out_w = OUTS["w"]
    pend = len(getattr(w, "_pending", ())) if w is not None else 0
    return round(time.perf_counter() - t, 3), out_w, pend

rounds = []
for rnd in range(1, args.rounds + 1):
    prompts = make_prompts(rnd)
    matched[0] = 0
    _stg0 = dict(getattr(getattr(sys.modules.get("expfs"), "LAST_WORKER", None), "transport", None).stats) \
        if args.kv_transport.endswith("_staged") else None
    with LK: IO.clear(); OUTS["r_max"] = OUTS["w_max"] = 0
    steps = []
    st = {}                      # rid -> dict(submit, first, finish)
    c0 = resource.getrusage(resource.RUSAGE_SELF)
    torch.cuda.nvtx.range_push(f"round{rnd}")
    tR0 = time.perf_counter()
    for i, toks in enumerate(prompts):
        rid = f"r{rnd}-{i}"
        eng.add_request(rid, {"prompt_token_ids": toks}, sp)
        st[rid] = dict(submit=time.perf_counter(), first=None, finish=None, ntok=0)
    kv_load_done = None
    guard = 0
    while any(v["finish"] is None for v in st.values()) and guard < 100000:
        guard += 1
        pre_w = wstat(); pre_kv = kv_snapshot()
        s0 = time.perf_counter()
        torch.cuda.nvtx.range_push("step")
        outs = eng.step()
        if args.poll_sleep_ms and not outs: time.sleep(args.poll_sleep_ms / 1000.0)
        torch.cuda.nvtx.range_pop()
        s1 = time.perf_counter()
        post_w = wstat(); post_kv = kv_snapshot()
        pending_first = sum(1 for v in st.values() if v["first"] is None and v["finish"] is None)
        n_out = len(outs)
        n_tok = sum(len(o.outputs[0].token_ids) for o in outs if o.outputs)
        for o in outs:
            v = st.get(o.request_id)
            if v is None: continue
            if v["first"] is None and o.outputs and o.outputs[0].token_ids:
                v["first"] = s1
            if o.outputs: v["ntok"] = len(o.outputs[0].token_ids)
            if o.finished: v["finish"] = s1
        steps.append(dict(t0=round(s0 - tR0, 4), t1=round(s1 - tR0, 4),
                          phase="prefill" if pending_first > 0 else "decode",
                          w_reads=post_w[0] - pre_w[0], w_bytes=post_w[1] - pre_w[1],
                          kv_rn=post_kv[1] - pre_kv[1], kv_rb=post_kv[2] - pre_kv[2],
                          kv_wn=post_kv[3] - pre_kv[3], kv_wb=post_kv[4] - pre_kv[4],
                          kv_out_r=post_kv[5], kv_out_w=post_kv[6],
                          n_out=n_out, n_tok=n_tok, pending_first=pending_first))
        # KV 읽기 outstanding 이 처음으로 0 으로 돌아온 시점 = 배치 전체의 KV 로드 완료
        if kv_load_done is None and post_kv[1] > 0 and post_kv[5] == 0:
            kv_load_done = s1
    torch.cuda.nvtx.range_pop()
    tR1 = time.perf_counter()
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    drain_s, drain_out, drain_pend = drain_stores()
    first = [v["first"] - tR0 for v in st.values() if v["first"]]
    fin = [v["finish"] - tR0 for v in st.values() if v["finish"]]
    with LK:
        io_r = [x for x in IO if x[3] == "r"]; io_w = [x for x in IO if x[3] == "w"]
        r_max, w_max = OUTS["r_max"], OUTS["w_max"]
    dec_steps = [s for s in steps if s["phase"] == "decode"]
    pre_steps = [s for s in steps if s["phase"] == "prefill"]
    rounds.append(dict(
        round=rnd, wall_s=round(tR1 - tR0, 3), cpu_s=round((c1.ru_utime - c0.ru_utime) + (c1.ru_stime - c0.ru_stime), 1),
        n_steps=len(steps), n_prefill_steps=len(pre_steps), n_decode_steps=len(dec_steps),
        seg1_submit_to_kvload_s=round(kv_load_done - tR0, 3) if kv_load_done else None,
        seg2_kvload_to_first_token_s=round(min(first) - (kv_load_done - tR0), 3) if (kv_load_done and first) else None,
        first_token_s=dict(min=round(min(first), 3), max=round(max(first), 3)) if first else None,
        seg3_first_to_last_token_s=round(max(fin) - min(first), 3) if first and fin else None,
        last_token_s=round(max(fin), 3) if fin else None,
        prefill_steps=dict(wall_s=round(sum(s["t1"] - s["t0"] for s in pre_steps), 3),
                           w_bytes_gib=round(sum(s["w_bytes"] for s in pre_steps) / 2**30, 2),
                           kv_read_gib=round(sum(s["kv_rb"] for s in pre_steps) / 2**30, 2),
                           kv_read_n=sum(s["kv_rn"] for s in pre_steps),
                           kv_write_gib=round(sum(s["kv_wb"] for s in pre_steps) / 2**30, 2),
                           kv_write_n=sum(s["kv_wn"] for s in pre_steps)),
        decode_steps=dict(wall_s=round(sum(s["t1"] - s["t0"] for s in dec_steps), 3),
                          w_bytes_gib=round(sum(s["w_bytes"] for s in dec_steps) / 2**30, 2),
                          kv_read_gib=round(sum(s["kv_rb"] for s in dec_steps) / 2**30, 2),
                          kv_read_n=sum(s["kv_rn"] for s in dec_steps),
                          kv_write_gib=round(sum(s["kv_wb"] for s in dec_steps) / 2**30, 2),
                          kv_write_n=sum(s["kv_wn"] for s in dec_steps)),
        kv_io=dict(read_n=len(io_r), read_gib=round(sum(x[2] for x in io_r) / 2**30, 2),
                   write_n=len(io_w), write_gib=round(sum(x[2] for x in io_w) / 2**30, 2),
                   read_busy_s=round(sum(x[1] - x[0] for x in io_r), 2),
                   write_busy_s=round(sum(x[1] - x[0] for x in io_w), 2),
                   read_span_s=round(max((x[1] for x in io_r), default=0) - min((x[0] for x in io_r), default=0), 3),
                   max_outstanding_read=r_max, max_outstanding_write=w_max),
        store_drain=dict(after_round_s=drain_s, leftover_writes=drain_out, leftover_jobs=drain_pend),
        steps=steps,
        kv_io_timeline=[(round(a - tR0, 4), round(b - tR0, 4), nb, op) for a, b, nb, op in (io_r + io_w)],
        matched=matched[0],
        staged_delta=({k: sys.modules["expfs"].LAST_WORKER.transport.stats.get(k, 0) - _stg0.get(k, 0) for k in
                       sys.modules["expfs"].LAST_WORKER.transport.stats} if _stg0 is not None else None),
        first_token_hot_s=[round(st[f"r{rnd}-{i}"]["first"] - tR0, 3) for i in range(min(HOT, args.n_prompts)) if st[f"r{rnd}-{i}"]["first"]],
        first_token_cold_s=[round(st[f"r{rnd}-{i}"]["first"] - tR0, 3) for i in range(HOT, args.n_prompts) if st[f"r{rnd}-{i}"]["first"]],
        ids=[st[f"r{rnd}-{i}"]["ntok"] for i in range(args.n_prompts)],
        prompt_tokens=[len(t) for t in prompts]))
    # 요구사항: 라운드 2 시작 전에 라운드 1 store 가 전부 끝났는지 확인.
    # 여기서 즉시 중단하면 20분치 측정이 날아가므로 기록만 하고 json 을 쓴 뒤 종료 코드로 알린다.
    rounds[-1]["store_fully_drained"] = (drain_out == 0 and drain_pend == 0)
    if not rounds[-1]["store_fully_drained"]:
        print(f"WARN round{rnd} 잔여 store: out={drain_out} pend={drain_pend}", flush=True)
    # decode 단계와 KV 읽기가 실제로 겹쳤는지: 겹친 시간과 바이트
    ov_s = ov_b = 0.0
    for a, b, nb, op in (io_r if args.kv_transport != "none" else []):
        for s in dec_steps:
            lo = max(a - tR0, s["t0"]); hi = min(b - tR0, s["t1"])
            if hi > lo: ov_s += hi - lo; ov_b += nb * (hi - lo) / max(b - a, 1e-9)
    rounds[-1]["decode_kvread_overlap"] = dict(seconds=round(ov_s, 3), bytes_gib=round(ov_b / 2**30, 3))

res["rounds"] = rounds
res["ids_equal_rounds"] = all(r["ids"][:HOT] == rounds[0]["ids"][:HOT] for r in rounds[1:])   # 반복 프롬프트만 비교
res["gpu_max_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
res["weight_ssd_stats"] = dict(reads=wstat()[0], read_gib=round(wstat()[1] / 2**30, 2))
res["sysmem_end"] = sysmem()
if args.kv_transport != "none":
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(args.kv_root) for f in fs]
    res["kv_files"] = len(files); res["kv_bytes_gib"] = round(sum(os.path.getsize(f) for f in files) / 2**30, 3)
json.dump(res, open(os.path.join(OUT, f"{args.tag}.json"), "w"), indent=1)
summ = {k: res[k] for k in ("load_s", "tiers", "ids_equal_rounds", "gpu_max_gib", "kv_files", "kv_bytes_gib") if k in res}
summ["rounds"] = [{k: r[k] for k in ("round", "wall_s", "n_steps", "seg1_submit_to_kvload_s", "seg2_kvload_to_first_token_s",
                                     "seg3_first_to_last_token_s", "prefill_steps", "decode_steps",
                                     "decode_kvread_overlap", "store_drain", "matched", "staged_delta")} for r in rounds]
print("RESULT", args.tag, json.dumps(summ), flush=True)
try: llm.llm_engine.engine_core.shutdown()
except Exception as e: print("shutdown:", e)
if not all(r["store_fully_drained"] for r in rounds):
    sys.exit("라운드 경계에서 store 가 남았다 — 구간 분리가 무효")
