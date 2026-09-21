"""관측 체계(lib/obs)를 붙인 워크로드 러너.
   --mode phases(기본): cold_fill(문서 N개, 질문 0) → settle → reverse_retrieve(역순, 질문 1) → final_settle
   --mode forced_hit: cold_fill(질문 0) → 저장 커밋 확인 → settle → replay(같은 순서, 같은 질문 0 = 프롬프트가 cold_fill과 글자 그대로 같음)
     replay 구간은 --no-store-phase replay로 저장을 막고 --reset-gpu-cache-before replay로 GPU 프리픽스 캐시를 비워,
     모든 요청이 티어에서 적재하도록 강제한다(적중률·쓰기 지연·캐시 정책을 뺀 전송 경로만의 비교).
   --mode stream: trace 순서 한 줄 스트림. 도착 시각 = (timestamp - timestamp0) x --time-scale, open loop(도착하면 제출).
     --time-scale 0은 closed loop(전부 즉시 제출, trace 순서), --max-concurrency K는 미완료 요청 K개로 제한(대기시간은 queue_s)
   가중치는 prefetch 오프로더(CPU/SSD 티어), KV는 in-tree CuFileFsSpec(native cuFile)으로 SSD. 외부 파이썬 전송 코드 없음. 엔진 step을 직접 돌려 step 단위 기록.
   산출물(RUN_DIR): environment.txt, capacity.json, events.jsonl(KV IO와 phase 마커), requests.jsonl(요청별 시각),
     steps.jsonl, tier_samples.jsonl(nvidia-fs·프로세스·캐시 파일 1초), hostmon 파일들, result.json, summary.csv
   프롬프트 소스: --prompt-source leval(OPT 토큰열, 구 실험 재현용) | longbench(문서 텍스트 → 모델 토크나이저) | bailian(02-bailian trace 프리픽스 구조)
   --profile-out PATH를 주면 재사용 프로파일(요청별 doc·phase·프롬프트 토큰·적중 토큰, doc별 재사용 횟수)을 따로 남김
   usage: python run_obs.py --run-dir DIR --model facebook/opt-13b --n-docs 32 --kv-batch 6 ..."""
import argparse, json, os, subprocess, sys, threading, time
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--model", default="facebook/opt-13b")
ap.add_argument("--n-docs", type=int, default=32)
ap.add_argument("--leval-workload", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "leval-opt-workload.json"))
ap.add_argument("--prompt-source", default="leval", choices=["leval", "longbench", "bailian"],
                help="leval: OPT 토큰열(OPT 토크나이저 전용). longbench: data/longbench-v2-10k-32.jsonl 텍스트를 모델 토크나이저로. "
                     "bailian: Bailian trace(data/bailian-qwen_coder.jsonl)의 hash_ids 프리픽스 구조만 합성 토큰으로 재현")
ap.add_argument("--longbench-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "longbench-v2-10k-32.jsonl"))
ap.add_argument("--bailian-trace", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bailian-qwen_coder.jsonl"),
                help="bailian trace jsonl(chat_id, turn, input_length, hash_ids). Alibaba Bailian 공개 트레이스")
ap.add_argument("--bailian-offset", type=int, default=0, help="bailian: trace 앞에서 건너뛸 행 수(재사용 비율이 평균에 가까운 창을 고를 때)")
ap.add_argument("--bailian-block", type=int, default=16, help="bailian: hash_id 하나가 나타내는 토큰 수(trace 생성 시 블록 크기)")
ap.add_argument("--prompt-cap", type=int, default=0, help="longbench/bailian: 프리픽스 토큰 상한(0이면 max_model_len - decode - 64)")
ap.add_argument("--profile-out", default=None, help="재사용 프로파일 json 경로. 요청별(doc, phase, 프롬프트 토큰 수, 적중 토큰 수)와 doc별 재사용 횟수")
ap.add_argument("--kv-transport", default="cufile", choices=["cufile", "none", "lmcache", "cpu", "hybrid", "tiering", "mooncake"],
                help="cufile: in-tree CuFileFsSpec(native). none: 재계산. lmcache: LMCache MP 서버의 GDS L1(GPU↔NVMe 직접). cpu: vLLM in-tree CPUOffloadingSpec(pinned host KV 층, LRU/ARC, SSD 없음). hybrid: 포크 HybridSpec(host 층 + GDS SSD 층, write-through). "
                     "tiering: vLLM in-tree TieringOffloadingSpec(CPU 1차 티어 + fs 2차 티어, 전송이 모두 host DRAM 경유)")
ap.add_argument("--kv-host-gb", type=float, default=8.0, help="cpu/tiering: host KV 층 크기(GB, cpu_bytes_to_use). hybrid: hybrid_host_gb")
ap.add_argument("--kv-roots", default=None, help="cufile/hybrid: 파일 티어 루트 여러 개. \"DIR:WEIGHT[:CAPGB],DIR:WEIGHT\". "
                "CAPGB(GiB)를 준 루트는 그만큼 차면 자리가 남은 루트로 넘긴다(spill). "
                "주면 cufile_fs_root_dirs로 넘어가고 cufile_fs_root_dir은 쓰지 않음. 블록 해시로 루트 하나를 가중치 비례로 고름. "
                "--kv-root는 그대로 필요하며(모니터·산출물 기준 경로) 여기 나열된 디렉터리 중 하나여야 함. 비어 있어야 하는 검사는 나열된 전부에 적용")
ap.add_argument("--kv-placement", default=None, choices=["host_first", "profile", "ratio"],
                help="hybrid: 배치 정책(hybrid_placement). 미지정이면 스펙 기본값 host_first")
ap.add_argument("--kv-host-share", type=float, default=None, help="hybrid --kv-placement ratio: host 티어로 보낼 블록 몫(0..1, hybrid_host_share)")
ap.add_argument("--lmcache-l1-gb", type=float, default=40.0, help="lmcache: GDS L1 슬랩 크기(GB)")
ap.add_argument("--lmcache-port", type=int, default=5555)
ap.add_argument("--lmcache-chunk", type=int, default=64, help="lmcache: 토큰 chunk. GDS staging 버퍼 = chunk KV × 4가 BAR1 안이어야 함")
ap.add_argument("--mooncake-master", default="30.0.0.4:50051", help="mooncake: master_server_address(원격 호스트의 RDMA 링크 IP:포트)")
ap.add_argument("--mooncake-metadata", default="http://30.0.0.4:8080/metadata", help="mooncake: metadata_server. master가 --enable_http_metadata_server로 같이 띄운 것")
ap.add_argument("--mooncake-device", default="mlx5_1", help="mooncake: 이 호스트에서 쓸 HCA 이름")
ap.add_argument("--mooncake-local-ip", default="30.0.0.3", help="mooncake: transfer engine이 쓸 이 호스트의 RDMA 링크 IP(MOONCAKE_REQUESTER_LOCAL_HOSTNAME)")
ap.add_argument("--mooncake-local-buffer-gb", type=float, default=1.0, help="mooncake: store가 등록하는 로컬 스테이징 버퍼(local_buffer_size, GB)")
ap.add_argument("--mooncake-staging-gb", type=float, default=8.0, help="mooncake: KV 텐서를 RDMA에 등록하지 못하는 GPU(BAR1 부족)에서 쓸 pinned host 경유 예산(GB). 0이면 경유 없음")
ap.add_argument("--mooncake-keep-store", action="store_true", help="mooncake: 런 시작 때 store를 비우지 않음(기본은 remove_all로 콜드 스타트)")
ap.add_argument("--register-tensors", action="store_true", help="KV 텐서를 cuFileBufRegister(BAR1 안에 들어갈 때만)")
ap.add_argument("--kv-batch", type=float, default=4, help="GPU KV 예산 = 요청 N개분 × 1.15 (소수 허용: 기준 런의 자동 예산을 그대로 맞출 때)")
ap.add_argument("--kv-threads", type=int, default=4)
ap.add_argument("--kv-block", type=int, default=64)
ap.add_argument("--host-weight-fraction", type=float, default=None, help="오프로드 가중치 대비 CPU 비율(환산). 미지정이면 --host-ram-fraction")
ap.add_argument("--host-ram-fraction", type=float, default=None, help="host memory(RAM 전체) 대비 비율을 오프로더에 그대로 전달. 둘 다 없으면 오프로더 기본값 0.3")
ap.add_argument("--pure", action="store_true", help="GPU KV 예산과 block_size를 vLLM 기본에 맡김(--kv-batch, --kv-block 무시)")
ap.add_argument("--prefetch-step", type=int, default=1)
ap.add_argument("--max-num-batched-tokens", type=int, default=0, help="chunked prefill chunk 상한(0=vLLM 기본). prefetch 깊이 2로 정적 버퍼가 늘어 첫 prefill이 OOM일 때 줄임")
ap.add_argument("--io-threads", type=int, default=4)
ap.add_argument("--gpu-util", type=float, default=0.9)
ap.add_argument("--max-model-len", type=int, default=2048)
ap.add_argument("--decode-tokens", type=int, default=8)
ap.add_argument("--mode", default="phases", choices=["phases", "forced_hit", "stream"],
                help="phases: cold_fill/reverse_retrieve 2단계(기본). forced_hit: cold_fill/replay 2단계(프롬프트가 같음). "
                     "stream: trace 순서 한 줄 스트림 + open loop 도착")
ap.add_argument("--no-store-phase", default="", help="이 phase(쉼표 구분) 동안 오프로드 매니저가 저장을 거부한다. "
                "hybrid/cufile은 prepare_store가 빈 결과, mooncake는 skip_save 강제. --kv-trace가 필요")
ap.add_argument("--reset-gpu-cache-before", default="", help="이 phase(쉼표 구분) 시작에 엔진 GPU 프리픽스 캐시를 비운다"
                "(reset_prefix_cache, 커넥터 캐시는 건드리지 않음)")
ap.add_argument("--kv-trace", action="store_true", help="KV 커넥터 요청·키 단위 계측을 kvtrace.jsonl에 남김(lib/obs/kvtrace.py)")
ap.add_argument("--commit-check-sec", type=float, default=0.0, help="phase 뒤 저장이 전부 커밋될 때까지 기다리는 상한(초, 0=안 함). "
                "hybrid/cufile은 매니저 pending과 워커 outstanding이 0이고 디스크 파일 수가 매니저 항목 수와 같을 때까지, "
                "mooncake는 저장 큐가 비고 master_key_count가 멈출 때까지")
ap.add_argument("--time-scale", type=float, default=1.0,
                help="stream: 도착 간격 배율. 20이면 trace를 20배 느리게. 0이면 closed loop(전부 즉시 제출)")
ap.add_argument("--max-concurrency", type=int, default=0,
                help="stream: 제출됐지만 끝나지 않은 요청의 상한(0=무제한). 초과 도착은 FIFO로 대기, 대기 시간은 queue_s")
ap.add_argument("--arrival-gap-s", type=float, default=0.0,
                help="stream + longbench/leval: timestamp가 없어 합성 도착 간격(초, 0=전부 즉시)")
ap.add_argument("--settle-sec", type=float, default=15.0)
ap.add_argument("--final-settle-sec", type=float, default=15.0)
ap.add_argument("--poll-sleep-ms", type=float, default=1.0)
ap.add_argument("--ssd-root", required=True); ap.add_argument("--kv-root", required=True)
ap.add_argument("--no-monitors", action="store_true")
ap.add_argument("--kv-split", default="off", help="split-source KV 모드(VLLM_KV_SPLIT): off | fixed:<frac> | model. "
                "적중 프리픽스의 앞 k청크는 GPU 재계산, 뒤 H-k청크는 티어에서 동시 적재")
ap.add_argument("--kv-split-rate-toks", type=float, default=None, help="split model 모드의 prefill 처리율(tok/s, VLLM_KV_SPLIT_RATE_TOKS)")
ap.add_argument("--kv-extra", default=None, help="cufile: kv_connector_extra_config에 덧붙일 JSON. 예: '{\"cufile_fs_store_window\": \"host\"}'")
ap.add_argument("--no-weight-offload", action="store_true", help="가중치를 전부 GPU에(오프로더 끔). 작은 모델 전용")
ap.add_argument("--kv-load-failure-policy", default="fail", choices=["fail", "recompute"])
ap.add_argument("--nsys-phase", default="", help="nsys 캡처 구간으로 삼을 phase 이름(쉼표 구분). 그 phase 시작에 cudaProfilerStart, 끝(또는 --nsys-steps 뒤)에 Stop. "
                     "lib/obs/run_nsys.sh를 NSYS_CAPTURE=cudaProfilerApi 로 감쌌을 때만 효과")
ap.add_argument("--nsys-steps", type=int, default=0, help="캡처 phase에서 이 수만큼의 엔진 step(0.3 s 이상인 forward 기준) 뒤 캡처 종료. 0이면 phase 끝까지")
args = ap.parse_args()
NSYS_PHASES = {x.strip() for x in args.nsys_phase.split(",") if x.strip()}
NO_STORE_PHASES = {x.strip() for x in args.no_store_phase.split(",") if x.strip()}
RESET_GPU_PHASES = {x.strip() for x in args.reset_gpu_cache_before.split(",") if x.strip()}
if NO_STORE_PHASES and not args.kv_trace:
    args.kv_trace = True  # 저장 차단 스위치가 kvtrace 안에 있다
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["VLLM_KV_SPLIT"] = args.kv_split
if args.kv_split_rate_toks is not None:
    os.environ["VLLM_KV_SPLIT_RATE_TOKS"] = str(args.kv_split_rate_toks)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "lib"))
R = os.path.abspath(args.run_dir)
if os.path.exists(os.path.join(R, "result.json")) or os.path.exists(os.path.join(R, "workload.exitcode")):
    sys.exit(f"RUN_DIR에 이미 런이 있음: {R}")
os.makedirs(R, exist_ok=True)
# 파일 티어 루트 목록. --kv-roots가 없으면 [(kv_root, 1)] 하나로 예전과 같다.
def _parse_kv_roots(spec):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item: continue
        parts = item.split(":")
        # DIR:WEIGHT 또는 DIR:WEIGHT:CAPGB. 디렉터리에 콜론은 없다고 본다.
        if len(parts) == 2: d, w, cap = parts[0], parts[1], "0"
        elif len(parts) == 3: d, w, cap = parts
        else: sys.exit(f"--kv-roots 항목은 DIR:WEIGHT[:CAPGB] 형식이어야 함: {item}")
        try: capf = float(cap)
        except ValueError: capf = -1
        if not d or not w.isdigit() or int(w) <= 0 or capf < 0:
            sys.exit(f"--kv-roots 항목은 DIR:WEIGHT(양의 정수)[:CAPGB(0 이상)] 형식이어야 함: {item}")
        out.append((os.path.abspath(d), int(w), capf))
    if not out: sys.exit("--kv-roots가 비어 있음")
    return out

KV_ROOTS = [(os.path.abspath(args.kv_root), 1, 0.0)]
if args.kv_roots:
    if args.kv_transport not in ("cufile", "hybrid"):
        sys.exit(f"--kv-roots는 --kv-transport cufile/hybrid에서만 씀(지금 {args.kv_transport})")
    KV_ROOTS = _parse_kv_roots(args.kv_roots)
    if os.path.abspath(args.kv_root) not in [r[0] for r in KV_ROOTS]:
        sys.exit(f"--kv-root는 --kv-roots에 나열된 디렉터리 중 하나여야 함: {args.kv_root}")
for _d, *_ in KV_ROOTS:
    if os.path.isdir(_d) and os.listdir(_d):
        sys.exit(f"kv-root가 비어 있지 않음: {_d}")

# ---- 환경, 용량 사전 검사 ----
INPUT_FILE = {"longbench": args.longbench_file, "bailian": args.bailian_trace}.get(args.prompt_source, args.leval_workload)
subprocess.run(["bash", os.path.join(ROOT, "lib", "obs", "envinfo.sh"), R, INPUT_FILE], check=False)
from transformers import AutoConfig
hc = AutoConfig.from_pretrained(args.model); d_model, n_layer = int(hc.hidden_size), int(hc.num_hidden_layers)
_kvh = int(getattr(hc, 'num_key_value_heads', None) or hc.num_attention_heads); _hd = d_model // int(hc.num_attention_heads)
kv_req = 2 * n_layer * _kvh * _hd * 2 * args.max_model_len  # K+V × layer × kv_heads × head_dim × fp16(GQA 반영)
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
kw = {} if args.no_weight_offload else dict(offload_backend="prefetch", offload_group_size=n_layer, offload_num_in_group=n_layer, offload_prefetch_step=args.prefetch_step,
          offload_ssd_path=args.ssd_root, offload_host_fraction=host_fraction, offload_ssd_transport="cufile",
          offload_ssd_io_threads=args.io_threads, offload_ssd_ring_mb=0)
matched = [0]; matched_req = {}
KV_EXTRA_CONFIG = None  # kv_connector_extra_config(커넥터를 쓰는 transport에서만)
LMC = None
if args.kv_transport == "lmcache":
    # LMCache MP 서버를 별도 프로세스로. --gds-l1-path 가 있으면 DRAM 층이 꺼지고 cuFile로 GPU↔NVMe 직접
    import socket
    os.makedirs(args.kv_root, exist_ok=True)
    lmc_cmd = ["lmcache", "server", "--host", "127.0.0.1", "--port", str(args.lmcache_port), "--chunk-size", str(args.lmcache_chunk),
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
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both", kv_load_failure_policy=args.kv_load_failure_policy,
        kv_connector_extra_config={"lmcache.mp.host": "tcp://127.0.0.1", "lmcache.mp.port": args.lmcache_port})
    import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector as lmcc
    for _n in dir(lmcc):
        _c = getattr(lmcc, _n)
        if isinstance(_c, type) and hasattr(_c, "get_num_new_matched_tokens") and "LMCache" in _n:
            _orig = _c.get_num_new_matched_tokens
            def _mk(_o):
                def _w(self, request, n):
                    r = _o(self, request, n); m = (r[0] or 0) if isinstance(r, tuple) else (r or 0)
                    matched[0] += m; matched_req.setdefault(getattr(request, "request_id", None), []).append(m)
                    return r
                return _w
            _c.get_num_new_matched_tokens = _mk(_orig)
elif args.kv_transport == "mooncake":
    # 논문 기준선(Mooncake Store + Transfer Engine). KV 풀은 원격 호스트 DRAM이고
    # 이 프로세스는 global_segment_size 0으로 붙는다(standalone-store). 풀 세그먼트를
    # 내놓는 쪽은 원격의 mooncake_client 프로세스뿐이라 모든 객체가 원격에 놓인다.
    # 접속 정보는 파일로만 받는 커넥터라 런 폴더에 쓰고 MOONCAKE_CONFIG_PATH로 가리킨다.
    mc_cfg = dict(metadata_server=args.mooncake_metadata, master_server_address=args.mooncake_master,
                  protocol="rdma", device_name=args.mooncake_device, mode="standalone-store",
                  global_segment_size=0, local_buffer_size=int(args.mooncake_local_buffer_gb * 2**30),
                  enable_offload=False)
    mc_path = os.path.join(R, "mooncake.json")
    json.dump(mc_cfg, open(mc_path, "w"), indent=1)
    os.environ["MOONCAKE_CONFIG_PATH"] = mc_path
    # vLLM의 get_ip()는 관리망 주소를 집으므로 RDMA 링크 주소를 명시한다.
    os.environ["MOONCAKE_REQUESTER_LOCAL_HOSTNAME"] = args.mooncake_local_ip
    if not args.mooncake_keep_store:
        from mooncake.store import MooncakeDistributedStore as _MDS
        _mc = _MDS()
        if _mc.setup(args.mooncake_local_ip, mc_cfg["metadata_server"], 0, 1 << 26, "rdma",
                     args.mooncake_device, mc_cfg["master_server_address"]) != 0:
            sys.exit("mooncake store에 붙지 못함(master/metadata 확인)")
        print("mooncake remove_all ->", _mc.remove_all(), flush=True); _mc.close(); del _mc
    extra = {"host_staging_gb": args.mooncake_staging_gb}
    if args.kv_extra: extra.update(json.loads(args.kv_extra))
    KV_EXTRA_CONFIG = dict(extra, _mooncake=dict(mc_cfg))  # result.json에 접속 구성까지 남김
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="MooncakeStoreConnector", kv_role="kv_both",
        kv_load_failure_policy=args.kv_load_failure_policy, kv_connector_extra_config=extra)
    # store의 객체 단위 = vLLM 블록 크기라 --kv-block을 엔진 블록 크기로 넘겨
    # 파일 티어 조건의 오프로드 블록 크기와 적중 단위를 맞춘다.
    if not args.pure: kw["block_size"] = args.kv_block
    import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler as mcsched
    _mg = mcsched.MooncakeStoreScheduler.get_num_new_matched_tokens
    def _mgw(self, request, n):
        r = _mg(self, request, n); m = (r[0] or 0)
        matched[0] += m; matched_req.setdefault(getattr(request, "request_id", None), []).append(m)
        return r
    mcsched.MooncakeStoreScheduler.get_num_new_matched_tokens = _mgw
elif args.kv_transport != "none":
    if args.kv_transport == "cpu":
        extra = {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": int(args.kv_host_gb * 1e9)}
    elif args.kv_transport == "tiering":
        # in-tree 다층 오프로드. 1차 티어는 pinned host DRAM(cpu_bytes_to_use), 2차 티어는
        # tiering/fs/manager.py의 FileSystemTierManager(type "fs", 파이썬 os.write/os.readv).
        # 2차 티어는 GPU에 직접 닿지 못하고 1차 티어를 거치므로 전 구간이 host DRAM 경유다.
        # 스레드 수는 cufile 경로(--kv-threads)와 같은 값을 줘 티어 병렬도를 맞춘다.
        extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": int(args.kv_host_gb * 1e9),
                 "secondary_tiers": [{"type": "fs", "root_dir": os.path.abspath(args.kv_root),
                                      "n_read_threads": args.kv_threads, "n_write_threads": args.kv_threads}]}
    elif args.kv_transport == "hybrid":
        extra = {"spec_name": "HybridSpec", "hybrid_host_gb": args.kv_host_gb,
                 "cufile_fs_register_tensors": str(args.register_tensors), "cufile_fs_read_threads": args.kv_threads, "cufile_fs_write_threads": args.kv_threads}
        if args.kv_placement: extra["hybrid_placement"] = args.kv_placement
        if args.kv_host_share is not None: extra["hybrid_host_share"] = args.kv_host_share
    else:
        extra = {"spec_name": "CuFileFsSpec", "cufile_fs_register_tensors": str(args.register_tensors),
                 "cufile_fs_read_threads": args.kv_threads, "cufile_fs_write_threads": args.kv_threads}
    # 루트가 여럿이면 cufile_fs_root_dirs만, 하나면 예전대로 cufile_fs_root_dir만 넘긴다.
    # tiering은 루트를 secondary_tiers[].root_dir로 이미 넘겼으므로 건너뛴다.
    if args.kv_transport != "tiering":
        if args.kv_roots:
            extra["cufile_fs_root_dirs"] = [{"dir": d, "weight": w, "capacity_gb": c} for d, w, c in KV_ROOTS]
        else:
            extra["cufile_fs_root_dir"] = args.kv_root
    if not args.pure: extra["block_size"] = args.kv_block
    if args.kv_extra: extra.update(json.loads(args.kv_extra))
    # 키 단위 계측을 켰는데 경로가 없으면 런 폴더에 둔다(KV 루트는 캠페인이 런 뒤 지움).
    if str(extra.get("cufile_fs_trace_keys", "false")).lower() in ("1", "true", "yes") and not extra.get("cufile_fs_trace_path"):
        extra["cufile_fs_trace_path"] = os.path.join(R, "keytrace.jsonl")
    KV_EXTRA_CONFIG = dict(extra)  # result.json에 그대로 남길 티어 구성
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both", kv_connector_extra_config=extra)
    import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
    _gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
    def _gmw(self, request, n):
        r = _gm(self, request, n); m = r[0] or 0
        matched[0] += m; matched_req.setdefault(getattr(request, "request_id", None), []).append(m)
        return r
    osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

try:
    t0 = time.time(); EV.phase("model_load_begin")
    if kv_gib: kw["kv_cache_memory_bytes"] = int(kv_gib * 2**30)
    if args.max_num_batched_tokens: kw["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(model=args.model, dtype="float16", gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len,
              enforce_eager=True, **kw)
    EV.phase("model_load_end", load_s=round(time.time() - t0, 1))
    if args.no_weight_offload:
        off = None; tiers = dict(n_modules=0, n_ssd=0, host_tier_gib=0.0, ssd_tier_gib=0.0, gpu_resident="all")
    else:
        from vllm.model_executor.offloader.base import get_offloader
        off = get_offloader()
        tiers = dict(n_modules=len(off.module_offloaders), n_ssd=sum(1 for m in off.module_offloaders if m.mode == "ssd"),
                     host_tier_gib=round(off.host_tier_bytes / 2**30, 2), ssd_tier_gib=round(off.ssd_tier_bytes / 2**30, 2))
    EV.emit("tiers", **tiers)
    # KV 커넥터 계측(조건 공통). 저장 차단 스위치도 여기 있다. cufile 인스턴스 래핑보다 먼저 붙인다.
    TRACE = None
    if args.kv_trace:
        from obs import kvtrace
        TRACE = kvtrace.install(args.kv_transport, os.path.join(R, "kvtrace.jsonl"))
        EV.emit("kv_trace", installed=TRACE is not None, transport=args.kv_transport)
    KVW = None
    if args.kv_transport == "hybrid":
        import vllm.v1.kv_offload.hybrid.spec as hspec
        KVW = hspec.LAST_WORKER.ssd if hspec.LAST_WORKER is not None else None  # SSD(GDS) 쪽 native 통계·이벤트
    if args.kv_transport == "cufile":
        import vllm.v1.kv_offload.cufile_fs.spec as cfs
        KVW = cfs.LAST_WORKER
        EV.emit("kv_worker", registered_tensors=KVW.native.registered, register_err=KVW.native.register_err, chunk_bytes=KVW.chunk_bytes)
        _ss, _sl, _gf = KVW.submit_store, KVW.submit_load, KVW.get_finished
        _jobs = {}
        def submit_store(job_id, src, dst):
            torch.cuda.nvtx.mark(f"kv_submit_store job={job_id} chunks={len(dst.paths) if hasattr(dst, 'paths') else '?'}"); _jobs[job_id] = ("w", time.monotonic(), len(dst.paths) * KVW.chunk_bytes)
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
    # cufile_fs_pending_wait 정책이 쓰는 "토큰 하나 재계산(prefill) 비용".
    # 엔진 step의 (걸린 시간 / 그 step에 예약된 토큰 수)를 EMA로 갱신해 파일 티어 매니저에 넣는다.
    # 예약 토큰이 PW_MIN_TOK 미만인 step은 decode라 보고 건너뛴다.
    PWMGR = None; PW_SCHED_TOK = [0]; PW_COST = [0.0]; PW_MIN_TOK = 256
    if KV_EXTRA_CONFIG and str(KV_EXTRA_CONFIG.get("cufile_fs_pending_wait", "false")).lower() in ("1", "true", "yes"):
        import vllm.v1.core.sched.scheduler as _vsched
        import vllm.v1.kv_offload.cufile_fs.spec as _cfs3
        import vllm.v1.kv_offload.hybrid.spec as _hs3
        PWMGR = (_hs3.LAST_MANAGER.ssd if args.kv_transport == "hybrid" and _hs3.LAST_MANAGER is not None
                 else _cfs3.LAST_MANAGER)
        _osched = _vsched.Scheduler.schedule
        def _sched_wrap(self):
            out = _osched(self)
            PW_SCHED_TOK[0] = getattr(out, "total_num_scheduled_tokens", 0)
            return out
        _vsched.Scheduler.schedule = _sched_wrap
        EV.emit("pending_wait", manager=type(PWMGR).__name__ if PWMGR is not None else None, min_tokens=PW_MIN_TOK)
    doc_meta = []  # bailian일 때만 채움(chat_id/turn/input_length/timestamp)
    if args.prompt_source == "longbench":
        # 실제 문서 텍스트를 모델 토크나이저로. 프리픽스 = 문서 토큰(상한까지), 단계별 꼬리 = 다른 질문 문장 → 프리픽스 적중
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        rows = [json.loads(l) for l in open(args.longbench_file) if l.strip()][:args.n_docs]
        cap = args.prompt_cap or (args.max_model_len - args.decode_tokens - 64)
        TAILS = ["\n\nDescribe the generated KV working set.", "\n\nExplain where this cached prefix was recovered from.",
                 "\n\nSummarize the document in one sentence.", "\n\nList three named entities from the text."]
        docs = [tok.encode(r["prompt"], add_special_tokens=True)[:cap] for r in rows]
        tails = [tok.encode(t, add_special_tokens=False) for t in TAILS]
        def prompt(i, q):
            return docs[i] + tails[q % len(tails)]
    elif args.prompt_source == "bailian":
        # trace의 hash_ids만 사용. hash_id 하나 = 결정적 16토큰 블록(시드=hash) → 같은 hash = 같은 토큰열이므로
        # trace의 프리픽스 공유 구조(hit/miss 패턴)가 그대로 재현됨. 
        import random
        _vocab = int(getattr(llm.get_tokenizer(), "vocab_size", 0) or hc.vocab_size)
        _lo, _hi = 1000, _vocab - 1000
        rows = []; _skipped = 0
        with open(os.path.abspath(args.bailian_trace)) as f:
            for line in f:
                if not line.strip(): continue
                if _skipped < args.bailian_offset: _skipped += 1; continue
                rows.append(json.loads(line))
                if len(rows) >= args.n_docs: break
        cap = args.prompt_cap or (args.max_model_len - args.decode_tokens - 64)
        nblk = max(1, cap // args.bailian_block)
        _bcache = {}
        def _block(h):
            if h not in _bcache:
                rng = random.Random(0xB10C0000 + h)
                _bcache[h] = [rng.randrange(_lo, _hi) for _ in range(args.bailian_block)]
            return _bcache[h]
        docs = []
        for r in rows:
            toks = []
            for h in r["hash_ids"][:nblk]: toks.extend(_block(h))
            docs.append(toks[:cap]); doc_meta.append(dict(chat_id=r.get("chat_id"), turn=r.get("turn"), input_length=r.get("input_length"),
                                 timestamp=r.get("timestamp"), n_hash=len(r["hash_ids"])))
        EV.emit("bailian_trace", offset=args.bailian_offset, rows=len(docs), block=args.bailian_block, blocks_kept=nblk, vocab=_vocab,
                unique_hashes=len(_bcache), total_hash_refs=sum(m["n_hash"] for m in doc_meta))
        _tail = {q: [random.Random(0x7A11 + q).randrange(_lo, _hi) for _ in range(8)] for q in range(4)}
        def prompt(i, q):
            return docs[i] + _tail[q % 4]
    else:
        W = json.load(open(args.leval_workload)); docs = W["docs"][:args.n_docs]
        def prompt(i, q):
            d = docs[i]; return d["prefix"] + W["delim_tokens"] + d["questions"][q % len(d["questions"])]["tokens"]
    sp = SamplingParams(max_tokens=args.decode_tokens, temperature=0, ignore_eos=True)
    eng = llm.llm_engine
    steps_f = open(os.path.join(R, "steps.jsonl"), "a", buffering=1); reqs_f = open(os.path.join(R, "requests.jsonl"), "a", buffering=1)
    def wstat():
        s = getattr(off, "ssd_tier", None) if off is not None else None; return (s.stats["reads"], s.stats["bytes"]) if s is not None else (0, 0)
    def drain():
        t = time.monotonic()
        while KVW is not None and time.monotonic() - t < 600:
            KVW.get_finished()
            st_ = KVW.stats()
            if not KVW._pending and st_["outstanding_writes"] == 0 and st_["outstanding_reads"] == 0: break
            time.sleep(0.005)
        return round(time.monotonic() - t, 3)
    def commit_check(tag):
        """저장이 전부 커밋됐는지 확인한다(강제 적중 조건의 전제).
           hybrid/cufile: 매니저 pending 0, 워커 outstanding 0, 루트의 실제 파일 수 == 매니저 항목 수.
           mooncake: 저장 큐 비움, 진행 중 저장 요청 0, master_key_count가 2초 동안 그대로.
           반환은 events.jsonl에 남길 요약 dict."""
        if args.commit_check_sec <= 0: return None
        t0 = time.monotonic(); info = dict(tag=tag, ok=False)
        if args.kv_transport in ("hybrid", "cufile"):
            import vllm.v1.kv_offload.hybrid.spec as _hs2
            import vllm.v1.kv_offload.cufile_fs.spec as _cfs2
            mgr = (_hs2.LAST_MANAGER.ssd if args.kv_transport == "hybrid" else _cfs2.LAST_MANAGER)
            while time.monotonic() - t0 < args.commit_check_sec:
                if KVW is not None: KVW.get_finished()
                st_ = KVW.stats() if KVW is not None else {}
                # 청크 파일만 센다(루트마다 있는 설정 json과 쓰는 중인 .tmp는 제외)
                n_disk = sum(1 for d, *_ in KV_ROOTS for _dp, _dn, fs in os.walk(d) for f in fs if f.endswith(".bin"))
                info.update(pending=len(mgr._pending), entries=len(mgr._entries), files_on_disk=n_disk,
                            outstanding_w=st_.get("outstanding_writes", 0), outstanding_r=st_.get("outstanding_reads", 0))
                # pending은 스케줄러 step에서 complete_store가 처리돼야 줄어드는데 phase 끝에는 step이 더 없어
                # 남아 있을 수 있다(디스크에는 파일이 다 있음). 그래서 완료 판정은 워커 미완료 0 + 디스크 파일 수가
                # 매니저 항목 + pending 합과 같을 때로 한다.
                if (st_.get("outstanding_writes", 0) == 0 and st_.get("outstanding_reads", 0) == 0
                        and n_disk == len(mgr._entries) + len(mgr._pending)):
                    info["ok"] = True; break
                time.sleep(0.05)
        elif args.kv_transport == "mooncake":
            import urllib.request
            from obs import kvtrace as _kt
            w = _kt.LAST_MC_WORKER
            host = args.mooncake_master.split(":")[0]
            def key_count():
                try:
                    for ln in urllib.request.urlopen(f"http://{host}:9003/metrics", timeout=5).read().decode().splitlines():
                        if ln.startswith("master_key_count "): return int(float(ln.split()[1]))
                except Exception: pass
                return -1
            last, stable_since = None, None
            while time.monotonic() - t0 < args.commit_check_sec:
                q = w.kv_send_thread.request_queue if w is not None else None
                n_q = q.unfinished_tasks if q is not None else 0
                n_live = len(w.kv_send_thread.stored_requests) if w is not None else 0
                kc = key_count()
                info.update(queue=n_q, live_store_reqs=n_live, master_key_count=kc)
                if n_q == 0 and n_live == 0:
                    if kc == last and stable_since is not None and time.monotonic() - stable_since > 2.0:
                        info["ok"] = True; break
                    if kc != last: last, stable_since = kc, time.monotonic()
                time.sleep(0.2)
        info["wait_s"] = round(time.monotonic() - t0, 2)
        EV.emit("commit_check", **info)
        print("commit_check", json.dumps(info), flush=True)
        return info

    prof = []
    def matched_of(rid):
        # 엔진이 내부 request_id에 접미사를 붙이므로(예: cold_fill-3-9feca30f) 접두 일치까지 본다.
        # 커넥터 lookup은 한 요청에 여러 번 불릴 수 있어 적중 토큰은 최댓값, 호출 수는 따로 센다.
        v = matched_req.get(rid)
        if v is None:
            v = [m for k, ms in matched_req.items() if k.startswith(rid + "-") for m in ms]
        return (max(v) if v else 0), len(v)
    def _engine_step(name, st, cap, on_finish):
        """엔진 step 1회 + steps.jsonl 기록. cap은 nsys 캡처 상태(on/n_fwd/n_step). 반환 (n_out, got_first)."""
        pre_w = wstat(); pre_k = kvstat(); s0 = time.monotonic(); w0 = time.time()
        torch.cuda.nvtx.range_push(f"step:{name} #{cap['n_step']} running={sum(1 for v in st.values() if v['first_mono'] is None or v['finish_mono'] is None)}")
        outs = eng.step(); torch.cuda.nvtx.range_pop(); cap["n_step"] += 1
        s1 = time.monotonic(); post_w = wstat(); post_k = kvstat()
        if PWMGR is not None and PW_SCHED_TOK[0] >= PW_MIN_TOK:
            c = (s1 - s0) / PW_SCHED_TOK[0]
            PW_COST[0] = c if PW_COST[0] <= 0 else 0.2 * c + 0.8 * PW_COST[0]
            PWMGR.set_recompute_cost(PW_COST[0])
        if cap["on"] and s1 - s0 > 0.3:
            cap["n_fwd"] += 1
            if args.nsys_steps and cap["n_fwd"] >= args.nsys_steps:
                torch.cuda.profiler.stop(); cap["on"] = False; EV.emit("nsys", msg=f"capture stop {name} after {cap['n_fwd']} forwards")
        got_first = False
        for o in outs:
            v = st.get(o.request_id)
            if v is None: continue
            if v["first_mono"] is None and o.outputs and o.outputs[0].token_ids: v["first_mono"] = s1; v["first_wall"] = time.time(); got_first = True; torch.cuda.nvtx.mark(f"req_first_token {o.request_id}")
            if o.outputs: v["ntok"] = len(o.outputs[0].token_ids); v["ids"] = list(o.outputs[0].token_ids)
            # GPU prefix cache에서 맞은 토큰(vLLM 집계). SSD hit(matched_of)과 분리해 재계산 몫을 가르기 위해 기록
            if getattr(o, "num_cached_tokens", None) is not None: v["gpu_cached"] = o.num_cached_tokens
            if o.finished:
                v["finish_mono"] = s1; v["finish_wall"] = time.time(); torch.cuda.nvtx.mark(f"req_finish {o.request_id}")
                on_finish(o.request_id, v)
        n_out = len(outs); n_tok = sum(len(o.outputs[0].token_ids) for o in outs if o.outputs)
        if s1 - s0 > 0.3 or n_out:
            kd = {k: post_k[k] - pre_k[k] for k in ("reads", "writes", "read_bytes", "write_bytes")} if post_k else {}
            steps_f.write(json.dumps(dict(phase=name, mono0=round(s0, 4), mono1=round(s1, 4), wall0=w0, dur=round(s1 - s0, 4),
                kind="prefill" if (got_first or (n_out == 0 and any(v["first_mono"] is None for v in st.values()))) else "decode",
                n_out=n_out, n_tok=n_tok, w_reads=post_w[0] - pre_w[0], w_bytes=post_w[1] - pre_w[1],
                kv_out_r=post_k.get("outstanding_reads", 0), kv_out_w=post_k.get("outstanding_writes", 0), **{"kv_" + k: v for k, v in kd.items()})) + "\n")
        elif args.poll_sleep_ms and not outs:
            time.sleep(args.poll_sleep_ms / 1000.0)
        return n_out, got_first
    def phase_begin(name):
        """phase 시작 직전 훅: 저장 차단 스위치와 GPU 프리픽스 캐시 비우기."""
        if TRACE is not None:
            TRACE.phase = name; TRACE.no_store = name in NO_STORE_PHASES
        if name in RESET_GPU_PHASES:
            # 커넥터 캐시는 그대로 둔다(reset_connector 기본 False). 직전 phase의 전송이 아직
            # 보고되지 않았으면 블록이 안 풀려 실패하므로, 엔진을 돌려 가며 될 때까지 다시 부른다.
            t0 = time.monotonic(); ok = False
            while time.monotonic() - t0 < 180.0:
                ok = bool(llm.llm_engine.reset_prefix_cache())
                if ok: break
                eng.step(); time.sleep(0.02)
            EV.emit("reset_gpu_prefix_cache", phase=name, ok=ok, wait_s=round(time.monotonic() - t0, 2))
            print(f"reset_prefix_cache before {name}: {ok} ({time.monotonic()-t0:.1f} s)", flush=True)
        EV.emit("phase_gate", phase=name, no_store=bool(TRACE is not None and TRACE.no_store),
                reset_gpu_cache=name in RESET_GPU_PHASES)

    def run_phase(name, order, q):
        matched[0] = 0; matched_req.clear()
        phase_begin(name)
        EV.phase(name, requests=len(order))
        cap = dict(on=name in NSYS_PHASES, n_fwd=0, n_step=0)
        if cap["on"]: torch.cuda.profiler.start(); EV.emit("nsys", msg=f"capture start {name}")
        torch.cuda.nvtx.range_push(f"phase:{name}")
        st = {}
        def on_finish(rid, v):
            _m, _n = matched_of(rid)
            v["matched_of"] = _m; v["lookups"] = _n  # 요청별 KV 적중 토큰(lookup 최댓값)과 lookup 호출 수
            reqs_f.write(json.dumps(dict(phase=name, rid=rid, **v)) + "\n")
            prof.append(dict(phase=name, rid=rid, doc=v["doc"], q=q, tokens=v["tokens"], matched=_m, lookups=_n))
        for i in order:
            rid = f"{name}-{i}"; toks = prompt(i, q)
            eng.add_request(rid, {"prompt_token_ids": toks}, sp); torch.cuda.nvtx.mark(f"req_submit {rid} {len(toks)}tok")
            st[rid] = dict(doc=i, tokens=len(toks), submit_mono=time.monotonic(), submit_wall=time.time(), first_mono=None, finish_mono=None, ntok=0)
        tS = time.monotonic()
        while any(v["finish_mono"] is None for v in st.values()):
            if time.monotonic() - tS > 3 * 3600: EV.emit("warn", msg=f"{name} 3시간 초과"); break
            _engine_step(name, st, cap, on_finish)
        d = drain()
        torch.cuda.nvtx.range_pop()
        if cap["on"]: torch.cuda.profiler.stop(); EV.emit("nsys", msg=f"capture stop {name} at phase end")
        EV.phase(name + "_end", wall_s=round(time.monotonic() - tS, 3), matched_tokens=matched[0], drain_s=d)
        # matched는 lookup 호출마다 누적된 합(호출 수에 따라 부풀음). 요청 단위는 matched_req_sum·hit_requests를 쓸 것.
        return dict(wall_s=round(time.monotonic() - tS, 3), matched=matched[0], drain_s=d,
                    matched_req_sum=sum(v.get("matched_of", 0) for v in st.values()),
                    hit_requests=sum(1 for v in st.values() if v.get("matched_of", 0) > 0),
                    ttft=[round(v["first_mono"] - v["submit_mono"], 3) for v in st.values() if v["first_mono"]],
                    e2e=[round(v["finish_mono"] - v["submit_mono"], 3) for v in st.values() if v["finish_mono"]])
    def arrival_times(n):
        """도착 시각(초, 첫 요청 기준 0). bailian은 trace timestamp x --time-scale, 그 외는 --arrival-gap-s 균등."""
        if args.prompt_source == "bailian" and doc_meta and doc_meta[0].get("timestamp") is not None:
            ts = [float(m["timestamp"]) for m in doc_meta[:n]]
            return [max(0.0, (t - ts[0]) * args.time_scale) for t in ts]
        return [i * max(0.0, args.arrival_gap_s) for i in range(n)]
    def run_stream(name="stream"):
        """trace 순서 한 줄 스트림. 도착하면 제출(open loop), --max-concurrency로 동시 미완료 수 제한."""
        from collections import deque
        matched[0] = 0; matched_req.clear()
        phase_begin(name)
        n = len(docs); arr = arrival_times(n)
        gaps = sorted(arr[i + 1] - arr[i] for i in range(n - 1))
        med_gap = gaps[len(gaps) // 2] if gaps else 0.0
        EV.emit("stream_plan", n_requests=n, time_scale=args.time_scale, max_concurrency=args.max_concurrency,
                arrival_span_s=round(arr[-1], 3) if arr else 0.0, median_gap_s=round(med_gap, 4),
                arrival_gap_s=args.arrival_gap_s, prompt_source=args.prompt_source, bailian_offset=args.bailian_offset)
        print(f"stream plan: {n} req, span {arr[-1] if arr else 0:.1f} s, median gap {med_gap:.3f} s, "
              f"time_scale {args.time_scale}, max_concurrency {args.max_concurrency}", flush=True)
        EV.phase(name, requests=n)
        cap = dict(on=name in NSYS_PHASES, n_fwd=0, n_step=0)
        if cap["on"]: torch.cuda.profiler.start(); EV.emit("nsys", msg=f"capture start {name}")
        torch.cuda.nvtx.range_push(f"phase:{name}")
        st = {}; live = [0]
        def on_finish(rid, v):
            live[0] -= 1
            _m, _n = matched_of(rid)
            v["matched_of"] = _m; v["lookups"] = _n
            reqs_f.write(json.dumps(dict(phase=name, rid=rid, **v)) + "\n")
            prof.append(dict(phase=name, rid=rid, doc=v["doc"], q=0, tokens=v["tokens"], matched=_m, lookups=_n))
        pend = deque(range(n)); hold = deque()
        tS = time.monotonic(); wS = time.time()
        while True:
            if time.monotonic() - tS > 3 * 3600: EV.emit("warn", msg=f"{name} 3시간 초과"); break
            now = time.monotonic() - tS
            while pend and arr[pend[0]] <= now: hold.append(pend.popleft())
            while hold and (args.max_concurrency <= 0 or live[0] < args.max_concurrency):
                i = hold.popleft(); rid = f"{name}-{i}"; toks = prompt(i, 0)
                sm = time.monotonic()
                eng.add_request(rid, {"prompt_token_ids": toks}, sp); torch.cuda.nvtx.mark(f"req_submit {rid} {len(toks)}tok")
                meta = doc_meta[i] if (args.prompt_source == "bailian" and i < len(doc_meta)) else {}
                st[rid] = dict(doc=i, chat_id=meta.get("chat_id"), turn=meta.get("turn"), input_length=meta.get("input_length"),
                               trace_ts=meta.get("timestamp"), tokens=len(toks), arrival_s=round(arr[i], 4),
                               arrival_wall=round(wS + arr[i], 4), submit_mono=sm, submit_wall=time.time(),
                               queue_s=round(max(0.0, (sm - tS) - arr[i]), 4), first_mono=None, finish_mono=None, ntok=0)
                live[0] += 1
            if live[0] == 0:
                if hold: continue
                if pend:
                    time.sleep(min(1.0, max(0.0, arr[pend[0]] - (time.monotonic() - tS)))); continue
                break
            _engine_step(name, st, cap, on_finish)
        d = drain()
        torch.cuda.nvtx.range_pop()
        if cap["on"]: torch.cuda.profiler.stop(); EV.emit("nsys", msg=f"capture stop {name} at phase end")
        fin = [v["finish_mono"] for v in st.values() if v["finish_mono"]]
        wall = round((max(fin) - tS) if fin else (time.monotonic() - tS), 3)  # 첫 도착 → 마지막 완료
        EV.phase(name + "_end", wall_s=wall, matched_tokens=matched[0], drain_s=d)
        return dict(wall_s=wall, n_requests=n, matched_tokens=matched[0], matched=matched[0], drain_s=d,
                    hit_requests=sum(1 for v in st.values() if v.get("matched_of", 0) > 0),
                    arrival_span_s=round(arr[-1], 3) if arr else 0.0, time_scale=args.time_scale,
                    max_concurrency=args.max_concurrency, submitted=len(st),
                    ttft=[round(v["first_mono"] - v["submit_mono"], 3) for v in st.values() if v["first_mono"]],
                    e2e=[round(v["finish_mono"] - v["submit_mono"], 3) for v in st.values() if v["finish_mono"]],
                    queue_s=[v["queue_s"] for v in st.values()])
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        kv_alloc_gib = round(cc.num_gpu_blocks * cc.block_size * (2 * n_layer * d_model * 2) / 2**30, 2)
    except Exception: kv_alloc_gib = kv_gib
    def mount_of(path):
        """path가 놓인 마운트 지점·장치·파일시스템. KV 디렉터리(--kv-root)가 어느 저장 계층인지
           result.json만 보고 알 수 있게 기록한다(로컬 SSD / 원격 NVMe-oF 마운트 구분)."""
        p = os.path.abspath(path); best = None
        try:
            for line in open("/proc/mounts"):
                f = line.split()
                if len(f) < 3: continue
                mp = f[1].replace("\\040", " ")
                if p == mp or p.startswith(mp.rstrip("/") + "/"):
                    if best is None or len(mp) > len(best[1]): best = (f[0], mp, f[2])
        except OSError:
            return None
        return dict(device=best[0], mount=best[1], fstype=best[2]) if best else None
    res = dict(args=vars(args), kv_gib=kv_gib, kv_alloc_gib=kv_alloc_gib, host_fraction=host_fraction, tiers=tiers,
               kv_root=os.path.abspath(args.kv_root), kv_root_fs=mount_of(args.kv_root),
               kv_roots=[dict(dir=d, weight=w, capacity_gb=c, fs=mount_of(d)) for d, w, c in KV_ROOTS],
               kv_extra_config=KV_EXTRA_CONFIG, phases={})
    N = len(docs)
    if args.mode == "stream":
        res["phases"]["stream"] = run_stream("stream")
    elif args.mode == "forced_hit":
        # replay는 cold_fill과 같은 순서·같은 질문(q=0)이라 프롬프트 토큰열이 글자 그대로 같다.
        res["phases"]["cold_fill"] = run_phase("cold_fill", list(range(N)), 0)
        res["commit_check_cold_fill"] = commit_check("cold_fill")
        EV.phase("settle"); time.sleep(args.settle_sec)
        res["phases"]["replay"] = run_phase("replay", list(range(N)), 0)
    else:
        res["phases"]["cold_fill"] = run_phase("cold_fill", list(range(N)), 0)
        EV.phase("settle"); time.sleep(args.settle_sec)
        res["phases"]["reverse_retrieve"] = run_phase("reverse_retrieve", list(reversed(range(N))), 1)
    EV.phase("final_settle"); time.sleep(args.final_settle_sec)
    if TRACE is not None:
        TRACE.flush()
        res["kv_trace"] = dict(rows=TRACE.n_rows, store_blocked_calls=TRACE.n_store_blocked,
                               path=os.path.abspath(TRACE.path))
    ks = kvstat()
    res["kv_io"] = dict(read_n=ks.get("reads", 0), read_gib=round(ks.get("read_bytes", 0) / 2**30, 2), write_n=ks.get("writes", 0),
                        write_gib=round(ks.get("write_bytes", 0) / 2**30, 2), read_busy_s=round(ks.get("read_busy_ns", 0) / 1e9, 1),
                        write_busy_s=round(ks.get("write_busy_ns", 0) / 1e9, 1), errors=ks.get("errors", 0), registered_tensors=ks.get("registered_tensors", 0))
    # 파일 1개 처리를 구간으로 나눈 스레드 시간 합(s): 이벤트 대기 / open+HandleRegister / cuFile 호출 / Deregister+close+rename
    res["kv_io"]["write_stages_s"] = {k: round(ks.get(f"w_{k}_ns", 0) / 1e9, 1) for k in ("ev", "open", "io", "fin")}
    res["kv_io"]["read_stages_s"] = {k: round(ks.get(f"r_{k}_ns", 0) / 1e9, 1) for k in ("open", "io", "fin")}
    res["kv_io"]["write_calls"] = ks.get("w_calls", 0); res["kv_io"]["read_calls"] = ks.get("r_calls", 0)
    try:
        if args.kv_transport == "hybrid":
            import vllm.v1.kv_offload.hybrid.spec as _hs
            res["kv_manager"] = _hs.LAST_MANAGER.stats() if _hs.LAST_MANAGER is not None else None
            res["kv_worker_hybrid"] = {k: v for k, v in _hs.LAST_WORKER.stats().items() if k != "ssd"} if _hs.LAST_WORKER is not None else None
        elif args.kv_transport == "cufile":
            import vllm.v1.kv_offload.cufile_fs.spec as _cfs
            res["kv_manager"] = _cfs.LAST_MANAGER.stats() if _cfs.LAST_MANAGER is not None else None
        elif args.kv_transport == "mooncake":
            # 포크 계측 전역이 없다. 풀 점유량은 master의 prometheus 엔드포인트에서 긁는다.
            res["kv_manager"] = None
            import urllib.request
            _host = args.mooncake_master.split(":")[0]
            _m = {}
            try:
                _txt = urllib.request.urlopen(f"http://{_host}:9003/metrics", timeout=5).read().decode()
                for ln in _txt.splitlines():
                    if ln.startswith("#") or " " not in ln: continue
                    k, _, v = ln.rpartition(" ")
                    if k.startswith(("master_", "segment_")):
                        try: _m[k] = float(v)
                        except ValueError: pass
            except Exception as _e:
                _m = {"error": repr(_e)}
            res["kv_mooncake"] = _m
        else:
            # cpu/tiering/lmcache/none은 포크 계측 전역이 없다. 전송량은 kv_files_by_root로 본다.
            res["kv_manager"] = None
    except Exception as e:
        res["kv_manager"] = {"error": repr(e)}
    try:
        import vllm.v1.kv_offload.split_policy as _sp
        res["kv_split"] = dict(_sp.LAST_SPLIT_STATS)
        res["kv_split"]["tail_wait_s_total"] = round(res["kv_split"]["tail_wait_s_total"], 3)
    except Exception as e:
        res["kv_split"] = {"error": repr(e)}
    if PWMGR is not None:
        res["kv_pending_wait_cost_s_per_token"] = round(PW_COST[0], 8)
    res["gpu_max_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    res["weight_ssd_reads"], res["weight_ssd_gib"] = wstat()[0], round(wstat()[1] / 2**30, 2)
    if args.kv_transport != "none":
        # 루트마다 따로 세고(kv_files_by_root) 합계는 예전 키 이름 그대로(kv_files, kv_bytes_gib).
        by_root, n_tot, b_tot = [], 0, 0
        for d, w, c in KV_ROOTS:
            if not os.path.isdir(d): continue
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs]
            b = sum(os.path.getsize(f) for f in files)
            by_root.append(dict(dir=d, weight=w, capacity_gb=c, files=len(files), bytes_gib=round(b / 2**30, 3)))
            n_tot += len(files); b_tot += b
        if by_root:
            res["kv_files_by_root"] = by_root
            res["kv_files"], res["kv_bytes_gib"] = n_tot, round(b_tot / 2**30, 3)
    if args.profile_out:
        by_doc = {}
        for e in prof:
            d = by_doc.setdefault(e["doc"], dict(doc=e["doc"], reuse=0, phases=[], tokens=e["tokens"], matched_total=0))
            d["reuse"] += 1; d["phases"].append(e["phase"]); d["matched_total"] += e["matched"]
        pdoc = sorted(by_doc.values(), key=lambda d: d["doc"])
        if args.prompt_source == "bailian":
            for d in pdoc: d["meta"] = doc_meta[d["doc"]]
        po = dict(run_dir=R, model=args.model, prompt_source=args.prompt_source, kv_transport=args.kv_transport,
                  n_docs=len(docs), input_file=INPUT_FILE, requests=prof, docs=pdoc,
                  totals=dict(requests=len(prof), prompt_tokens=sum(e["tokens"] for e in prof),
                              matched_tokens=sum(e["matched"] for e in prof),
                              reused_docs=sum(1 for d in pdoc if d["reuse"] > 1)))
        os.makedirs(os.path.dirname(os.path.abspath(args.profile_out)) or ".", exist_ok=True)
        json.dump(po, open(args.profile_out, "w"), indent=1)
        res["profile_out"] = os.path.abspath(args.profile_out)
    json.dump(res, open(os.path.join(R, "result.json"), "w"), indent=1)
    EV.phase("complete")
    open(os.path.join(R, "workload.exitcode"), "w").write("0\n")
    print("RESULT", json.dumps({k: res[k] for k in ("tiers", "kv_io", "gpu_max_gib")}), flush=True)
    for ph, v in res["phases"].items(): print(f"  {ph}: wall {v['wall_s']} s, ttft median {sorted(v['ttft'])[len(v['ttft'])//2] if v['ttft'] else None}, matched {v.get('matched', v.get('matched_tokens', 0))}", flush=True)
    try: llm.llm_engine.engine_core.shutdown()
    except Exception as e: print("shutdown:", e)
except BaseException as e:
    EV.emit("error", err=repr(e)); open(os.path.join(R, "workload.exitcode"), "w").write("1\n"); raise
finally:
    _tr = globals().get("TRACE")
    try:
        if _tr is not None: _tr.close()
    except Exception as e: print("kvtrace close:", e)
    stop_monitors()
    if LMC is not None:
        try: LMC.terminate(); LMC.wait(timeout=30)
        except Exception: LMC.kill()
    subprocess.run([sys.executable, os.path.join(ROOT, "lib", "obs", "summarize.py"), R], stdout=open(os.path.join(R, "summary.csv"), "w"), check=False)
