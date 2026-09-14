#!/usr/bin/env python3
"""실행 전 용량 검사. 모델 config와 요청 토큰 수로 KV를 계산해 GPU KV 예산, RAM, 디스크와 대조하고 capacity.json을 씀.
GPU 예산이 작업 집합보다 크면 2단계가 SSD를 읽지 않으므로 경고(--strict면 실패).
usage: plan.py --model M --tokens-per-request N --requests K --gpu-kv-gib G --cache-dir D [--host-gib H] --out capacity.json"""
import argparse, json, os, shutil, sys
from transformers import AutoConfig

def main():
    ap = argparse.ArgumentParser()
    for k in ("--model", "--cache-dir", "--out"): ap.add_argument(k, required=True)
    ap.add_argument("--tokens-per-request", type=int, required=True); ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--gpu-kv-gib", type=float, required=True); ap.add_argument("--host-gib", type=float, default=0.0)
    ap.add_argument("--block", type=int, default=64); ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    c = AutoConfig.from_pretrained(a.model)
    L, d = int(c.num_hidden_layers), int(c.hidden_size)
    heads = int(getattr(c, "num_key_value_heads", None) or c.num_attention_heads); hd = d // int(c.num_attention_heads)
    per_tok = 2 * L * heads * hd * 2
    per_req = -(-a.tokens_per_request // a.block) * a.block * per_tok
    ws = per_req * a.requests
    avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
    d0 = a.cache_dir
    while not os.path.exists(d0): d0 = os.path.dirname(d0) or "/"
    free = shutil.disk_usage(d0).free
    warn = []
    if ws <= a.gpu_kv_gib * 2**30 * 0.9: warn.append("작업 집합이 GPU KV 예산 안에 들어가 2단계가 SSD를 읽지 않음. 요청 수를 늘리거나 GPU KV를 줄일 것")
    if free < ws * 1.1: warn.append("캐시 디렉터리 디스크 여유가 작업 집합보다 작음")
    if a.host_gib and a.host_gib * 2**30 > avail * 0.8: warn.append("host 티어가 MemAvailable의 80%를 넘음")
    out = dict(model=a.model, layers=L, kv_heads=heads, head_dim=hd, kv_bytes_per_token=per_tok, kv_bytes_per_request=per_req,
               requests=a.requests, working_set_bytes=ws, gpu_kv_bytes=int(a.gpu_kv_gib * 2**30), batch_fit=int(a.gpu_kv_gib * 2**30 // per_req),
               mem_available_bytes=avail, disk_free_bytes=free, warnings=warn)
    json.dump(out, open(a.out, "w"), indent=1)
    for w in warn: print("WARNING:", w)
    print(f"plan: KV {per_tok/1e6:.2f} MB/token, {per_req/2**30:.2f} GiB/request, working set {ws/2**30:.1f} GiB, GPU holds {out['batch_fit']} requests")
    if a.strict and warn: sys.exit(2)

if __name__ == "__main__":
    main()
