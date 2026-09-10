#!/usr/bin/env bash
# 17: KV int8 양자화(cufile_q8) — 게이트 2 위에 얹어 fp16 저장(mx-gate2)과 비교. 섞인 파도 3라운드, 4MiB 조각.
#   양자화는 저장 계층에서만: 행(9216 fp16 = 토큰 하나의 K/V 벡터)마다 absmax 스케일, 파일 크기 절반.
#   기대: 파도 prefill forward 14.4s 의 로드 대기분(약 1.1s)이 절반으로. 출력 토큰은 fp16 과 달라질 수 있음(기록).
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
export CUFILE_ENV_PATH_JSON=$(cd ../08-cufile-bounce && pwd)/cufile-pb4096.json
export VLLM_KV_LOAD_WAVE_GATE=2 VLLM_KV_LOAD_WAVE_WAIT_S=30
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
tag=mx-gate2-q8
rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
log "== $tag (cufile_q8, GATE=2)"
./memguard.sh $tag $O/memguard.log & guard=$!
python run_phase_66b.py --rounds 3 --n-prompts 16 --hot-prompts 4 --out-dir $O --kv-transport cufile_q8 --tag $tag > $O/$tag.log 2>&1
kill $guard 2>/dev/null; wait $guard 2>/dev/null
[ -f $O/$tag.json ] && grep -a '^RESULT' $O/$tag.log | cut -c1-300 || { log "FAILED $tag"; grep -aE 'Error|Traceback' $O/$tag.log | tail -4; }
log "17 완료"
