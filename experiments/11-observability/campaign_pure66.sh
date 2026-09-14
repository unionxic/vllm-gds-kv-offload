#!/usr/bin/env bash
# 손대지 않은 기본값으로 66B. host memory 비율(RAM 대비) 0.1~0.7, 재계산/SSD 적중. 게이트·cuFile 조각·KV 예산·block_size·폴링 양보 전부 기본.
# 유일한 예외: VLLM_OFFLOAD_PIN_EXACT=1(pinned 올림 버그 수정, 없으면 0.7에서 머신이 죽음)과 메모리 워치독.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
unset VLLM_KV_LOAD_WAVE_GATE CUFILE_ENV_PATH_JSON VLLM_OFFLOAD_SSD_REGISTER_MAX_MB
O=$(cd ../../results/native-66b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
log "== pure 66B: RAM ${RATIOS:-0.5} × (none, cufile), 기본값(게이트 없음, cuFile 기본 json, KV 예산 vLLM 기본, poll sleep 0)"
for f in ${RATIOS:-0.5}; do for kvt in none cufile; do tag=pure-ram$f-$kvt
  [ -f $O/$tag/result.json ] && { log "skip $tag"; continue; }
  rm -rf $O/ssd-66b $O/kv-66b; ../07-combined/memguard.sh $tag $O/memguard.log & g=$!
  log "start $tag"
  python run_obs.py --run-dir $O/$tag --model facebook/opt-66b --n-docs 8 --decode-tokens 8 --host-ram-fraction $f --pure --poll-sleep-ms 0 \
    --kv-transport $kvt --settle-sec 15 --final-settle-sec 15 --ssd-root $O/ssd-66b --kv-root $O/kv-66b > $O/$tag.log 2>&1
  rc=$?; kill $g 2>/dev/null; wait $g 2>/dev/null; rm -rf $O/kv-66b
  [ -f $O/$tag/result.json ] && log "done  $tag rc=$rc" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error|KILL' $O/$tag.log $O/memguard.log | tail -2 >> $O/campaign.log; }
done; done
rm -rf $O/ssd-66b; log "== pure 66B 종료"
