# W1: Qwen-Bailian coder trace replay — 각 설계의 "최선 구성" 간 end-to-end 비교.
#   V1: D(expfs+cufile,b256) vs C(tiering,b16) / V2: D(b64) vs C(b16). 러너 간 비교 금지.
#
# 방법(명시):
#  - closed-loop 순차 리플레이, arrival 순서 보존(타임스탬프 간격 미재현 — 캐시에 시간
#    만료가 없어 재사용 거리는 요청 순서로 보존됨).
#  - 프롬프트 재구성: hash_id → 결정적 16-token 블록(시드=hash) — 동일 블록=동일 토큰
#    → trace의 prefix hit/miss 패턴 보존 (공식 replayer와 동일 원리).
#  - rain 제약: 2032 토큰(127 블록) 절단 — 재사용은 프롬프트 머리에 집중되므로 구조 보존.
#  - max_tokens=1 (TTFT=wall; decode 부하 부재는 한계로 기록. 다음 턴 입력은 trace
#    블록에서 오므로 재사용 패턴엔 영향 없음).
#  - cache capacity: GPU 풀 동일(gpu_util 0.7), SSD 무제한 동일. C는 추가로 CPU 8GiB
#    (현행 설계의 최선 그대로 — D보다 티어가 하나 더 있는 구성임을 명시).
#
# 기록: TTFT, storage-matched tokens, num_cached_tokens(GPU+storage), SSD read bytes/IO,
#       store bytes/IO, CPU time, per-req 메타. 요약: p50/p95/p99, hit 분해, 처리량,
#       host-memory traffic(유도: C=2×loaded bytes, D=0).
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "lib"))  # expfs·gdslib 등

cnt = {"fs_load_n": 0, "fs_load_b": 0, "fs_store_n": 0, "fs_store_b": 0,
       "tp_read_n": 0, "tp_read_b": 0, "tp_write_n": 0, "tp_write_b": 0,
       "matched": 0, "guard_skips": 0}

# WORKAROUND (문서화): vLLM 버그 — _build_store_jobs가 finished_req_ids를 순회하는데
# TieringOffloadingManager.on_request_finished가 이미 _req_state를 지운 뒤 마지막
# chunk의 prepare_store가 도착하면 KeyError(tiering/manager.py:542). V1 러너 + 본
# trace에서 요청 ~80 부근 재현(2회: req 84, 79). 가드: 미지 req_id의 store 준비를
# 스킵(None 반환 = "store 불가" 정상 경로)하고 횟수를 기록. 해당 요청의 마지막
# chunk 저장만 소실 — hit 지표 영향은 guard_skips로 정량화.
import vllm.v1.kv_offload.tiering.manager as tmgr

_ps = tmgr.TieringOffloadingManager.prepare_store
def _ps_guard(self, keys, req_context):
    if req_context.req_id not in self._req_state:
        cnt["guard_skips"] += 1
        return None
    return _ps(self, keys, req_context)
tmgr.TieringOffloadingManager.prepare_store = _ps_guard

import vllm.v1.kv_offload.tiering.fs.manager as fsm
import expfs

_ol, _os_ = fsm.load_block, fsm.store_block
def _cl(path, view, off, bs):
    cnt["fs_load_n"] += 1; cnt["fs_load_b"] += bs
    return _ol(path, view, off, bs)
def _cs(path, buf, off, bs):
    cnt["fs_store_n"] += 1; cnt["fs_store_b"] += bs
    return _os_(path, buf, off, bs)
fsm.load_block, fsm.store_block = _cl, _cs

for cls in (expfs.CuFileTransport, expfs.PosixBounceTransport):
    _r, _w = cls.read_chunk, cls.write_chunk
    def mkr(fn):
        def inner(self, path, spans, cb):
            cnt["tp_read_n"] += 1; cnt["tp_read_b"] += sum(s[2] for s in spans)
            return fn(self, path, spans, cb)
        return inner
    def mkw(fn):
        def inner(self, path, spans, cb):
            cnt["tp_write_n"] += 1; cnt["tp_write_b"] += sum(s[2] for s in spans)
            return fn(self, path, spans, cb)
        return inner
    cls.read_chunk, cls.write_chunk = mkr(_r), mkw(_w)

# storage-matched tokens (connector lookup 결과)
import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as osched
_gm = osched.OffloadingConnectorScheduler.get_num_new_matched_tokens
def _gmw(self, request, num_computed_tokens):
    r = _gm(self, request, num_computed_tokens)
    if r[0]:
        cnt["matched"] += r[0]
    return r
osched.OffloadingConnectorScheduler.get_num_new_matched_tokens = _gmw

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def block_tokens(h, vocab):
    rng = random.Random(0xB10C0000 + h)
    return [rng.randrange(1000, vocab - 1000) for _ in range(16)]


def shm_observe(label):
    """/dev/shm 관찰 기록 (삭제하지 않음 — 회수는 엔진의 몫, 로그로 검증)."""
    import glob as _g
    files = sorted(_g.glob("/dev/shm/vllm_offload_*.mmap"))
    used = sum(os.path.getsize(f) for f in files)
    print(f"[shm:{label}] files={len(files)} used={used/2**30:.2f}GiB "
          f"{[os.path.basename(f) for f in files]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=["C", "D", "E"])
    ap.add_argument("--runner", required=True, choices=["v1", "v2"])
    ap.add_argument("--block", type=int, default=None,
                    help="offload block_size 오버라이드 (기본: C=16, D/E=러너별)")
    ap.add_argument("--n-requests", type=int, default=600)
    ap.add_argument("--model", default="facebook/opt-2.7b")
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(HERE, "w1_results.csv"))
    args = ap.parse_args()

    defaults = {"C": 16, "D": {"v1": 256, "v2": 64}[args.runner],
                "E": {"v1": 256, "v2": 64}[args.runner]}
    block = args.block if args.block is not None else defaults[args.design]
    tag = f"{args.runner}-{args.design}-b{block}"
    kvroot = os.path.join(HERE, f"kvroot-{tag}")
    shutil.rmtree(kvroot, ignore_errors=True)

    shm_observe("before-engine")  # 관찰만 — 회수는 엔진 자체 로직("Reclaimed" 로그)
    if args.design == "C":
        extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": 8 << 30,
                 "block_size": block,
                 "secondary_tiers": [{"type": "fs", "root_dir": kvroot}]}
    else:
        extra = {"spec_name": "ExperimentalFilesystemSpec", "spec_module_path": "expfs",
                 "expfs_root_dir": kvroot,
                 "expfs_transport": "cufile" if args.design == "D" else "posix",
                 "block_size": block}

    llm = LLM(model=args.model,
              kv_transfer_config=KVTransferConfig(
                  kv_connector="OffloadingConnector", kv_role="kv_both",
                  kv_connector_extra_config=extra),
              gpu_memory_utilization=args.gpu_util, max_model_len=2048,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    vocab = tok.vocab_size
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    reqs = []
    with open(os.path.join(HERE, "trace", "qwen_coder.jsonl")) as f:
        for line in f:
            reqs.append(json.loads(line))
            if len(reqs) >= args.n_requests:
                break

    bcache: dict[int, list[int]] = {}
    rows = []
    t_all0 = time.perf_counter()
    c_all0 = time.process_time()
    for i, r in enumerate(reqs):
        blocks = r["hash_ids"][:127]
        toks = []
        for h in blocks:
            if h not in bcache:
                bcache[h] = block_tokens(h, vocab)
            toks.extend(bcache[h])
        tail = min(r["input_length"] - 16 * len(r["hash_ids"]), 2040 - len(toks))
        if tail > 0:
            rng = random.Random(0x7A11 + r["chat_id"])
            toks.extend(rng.randrange(1000, vocab - 1000) for _ in range(tail))

        before = dict(cnt)
        c0, t0 = time.process_time(), time.perf_counter()
        out = llm.generate([{"prompt_token_ids": toks}], sp, use_tqdm=False)
        ttft = time.perf_counter() - t0
        cpu_s = time.process_time() - c0
        d = {k: cnt[k] - before[k] for k in cnt}
        cached = getattr(out[0], "num_cached_tokens", None)
        rows.append(dict(
            tag=tag, runner=args.runner, design=args.design, block=block, i=i,
            chat_id=r["chat_id"], turn=r["turn"], prompt_tokens=len(toks),
            ttft_s=round(ttft, 4), cpu_s=round(cpu_s, 4),
            matched_tokens=d["matched"], cached_tokens=cached,
            ssd_read_ios=d["fs_load_n"] + d["tp_read_n"],
            ssd_read_bytes=d["fs_load_b"] + d["tp_read_b"],
            store_ios=d["fs_store_n"] + d["tp_write_n"],
            store_bytes=d["fs_store_b"] + d["tp_write_b"],
        ))
        if i % 50 == 0:
            free = shutil.disk_usage("/").free
            print(f"[{tag}] {i}/{len(reqs)} ttft={ttft:.3f} free={free/2**30:.0f}G", flush=True)
            if free < 25 << 30:
                print(f"[{tag}] DISK GUARD: aborting at {i}", flush=True)
                break
            # 크래시 대비 증분 flush
            part = os.path.join(HERE, f"w1_partial_{tag}.csv")
            with open(part, "w", newline="") as pf:
                w = csv.DictWriter(pf, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    wall = time.perf_counter() - t_all0
    cpu_total = time.process_time() - c_all0
    print(f"[{tag}] TOTAL wall={wall:.1f}s cpu={cpu_total:.1f}s "
          f"ssd_read={cnt['fs_load_b']+cnt['tp_read_b']:,}B "
          f"store={cnt['fs_store_b']+cnt['tp_write_b']:,}B", flush=True)

    hdr = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if hdr:
            w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, f"w1_total_{tag}.json"), "w") as f:
        json.dump(dict(tag=tag, wall_s=wall, cpu_s=cpu_total,
                       prompt_tokens=sum(x["prompt_tokens"] for x in rows), **cnt), f)
    # 정상 종료 경로: 엔진을 명시적으로 내려 region cleanup("Removed mmap")을 보장.
    # (이게 없으면 인터프리터 exit가 cleanup을 건너뛰어 파일이 남는다 — 다음 시작이
    #  회수하긴 하나, 정상 종료는 잔재 0이어야 한다는 원칙)
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"[shutdown] explicit engine shutdown failed: {e!r}", flush=True)
    shm_observe("after-shutdown")


if __name__ == "__main__":
    main()
