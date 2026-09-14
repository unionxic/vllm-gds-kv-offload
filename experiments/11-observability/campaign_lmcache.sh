#!/usr/bin/env bash
# LMCache GDS L1(GPU↔NVMe 직접) 비교 런. 먼저 opt-2.7b 스모크(출력 토큰 정합성), 그다음 66B RAM 0.5 기본값 조건
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
unset VLLM_KV_LOAD_WAVE_GATE CUFILE_ENV_PATH_JSON VLLM_OFFLOAD_SSD_REGISTER_MAX_MB
O=$(cd ../../results/native-66b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
run(){ local tag=$1; shift
  [ -f $O/$tag/result.json ] && { log "skip $tag"; return 0; }
  rm -rf $O/ssd-$tag $O/kv-$tag; ../07-combined/memguard.sh $tag $O/memguard.log & local g=$!
  log "start $tag"
  python run_obs.py --run-dir $O/$tag --kv-transport lmcache --ssd-root $O/ssd-$tag --kv-root $O/kv-$tag "$@" > $O/$tag.log 2>&1; local rc=$?
  kill $g 2>/dev/null; wait $g 2>/dev/null; rm -rf $O/kv-$tag $O/ssd-$tag
  [ -f $O/$tag/result.json ] && log "done  $tag rc=$rc" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error|KILL' $O/$tag.log $O/$tag/lmcache.log $O/memguard.log 2>/dev/null | tail -3 >> $O/campaign.log; }
}
run smoke-2.7b-lmcache --model facebook/opt-2.7b --n-docs 8 --kv-batch 2 --decode-tokens 8 --settle-sec 2 --final-settle-sec 2 --lmcache-l1-gb 8
run pure-ram0.5-lmcache --model facebook/opt-66b --n-docs 8 --decode-tokens 8 --host-ram-fraction 0.5 --pure --poll-sleep-ms 0 --gpu-util 0.85 --settle-sec 15 --final-settle-sec 15 --lmcache-l1-gb 40
log "== lmcache 런 종료"
