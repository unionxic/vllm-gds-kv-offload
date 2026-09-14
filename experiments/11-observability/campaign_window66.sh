#!/usr/bin/env bash
# 저장 창 검증: 66B RAM 0.5 기본값 조건(게이트 없음, 1 MiB 조각, KV 자동, gpu_util 0.85)에 cufile_fs_store_window=host만 추가.
# 비교 대상 results/native-66b/pure-ram0.5-cufile(저장 단계 1,061 s, 두 단계 합계 1,992 s)
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
unset VLLM_KV_LOAD_WAVE_GATE CUFILE_ENV_PATH_JSON VLLM_OFFLOAD_SSD_REGISTER_MAX_MB
O=$(cd ../../results/native-66b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
tag=${TAG:-pure-ram0.5-cufile-window}; [ -f $O/$tag/result.json ] && exit 0
rm -rf $O/ssd-66b $O/kv-66b; ../07-combined/memguard.sh $tag $O/memguard.log & g=$!
log "start $tag (kv_extra ${KV_EXTRA:-default host})"
python run_obs.py --run-dir $O/$tag --model facebook/opt-66b --n-docs 8 --decode-tokens 8 --host-ram-fraction 0.5 --pure --poll-sleep-ms 0 --gpu-util 0.85 \
  --kv-transport cufile --kv-extra "${KV_EXTRA:-{\"cufile_fs_store_window\": \"host\"\}}" --settle-sec 15 --final-settle-sec 15 --ssd-root $O/ssd-66b --kv-root $O/kv-66b > $O/$tag.log 2>&1
rc=$?; kill $g 2>/dev/null; wait $g 2>/dev/null; rm -rf $O/kv-66b $O/ssd-66b
[ -f $O/$tag/result.json ] && log "done  $tag rc=$rc" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error|KILL' $O/$tag.log $O/memguard.log | tail -2 >> $O/campaign.log; }
