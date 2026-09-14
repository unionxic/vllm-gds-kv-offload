#!/usr/bin/env bash
# in-tree native CuFileFsSpec으로 66B 기준표 조건(RAM 0.7, 문서 8, 배치 2, decode 8) 재현: 재계산 / SSD 적중
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1 VLLM_KV_LOAD_WAVE_GATE=2
export CUFILE_ENV_PATH_JSON=$(cd ../08-cufile-bounce && pwd)/cufile-pb4096.json
O=$(cd ../../results/native-66b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
for kvt in none cufile; do tag=opt-66b-ram0.7-$kvt
  [ -f $O/$tag/result.json ] && { log "skip $tag"; continue; }
  rm -rf $O/ssd-66b $O/kv-66b; ../07-combined/memguard.sh $tag $O/memguard.log & g=$!
  log "start $tag"
  python run_obs.py --run-dir $O/$tag --model facebook/opt-66b --n-docs 8 --kv-batch 2 --decode-tokens 8 --host-weight-fraction 0.723 \
    --kv-transport $kvt --settle-sec 15 --final-settle-sec 15 --ssd-root $O/ssd-66b --kv-root $O/kv-66b > $O/$tag.log 2>&1
  rc=$?; kill $g 2>/dev/null; wait $g 2>/dev/null; rm -rf $O/kv-66b
  [ -f $O/$tag/result.json ] && log "done  $tag rc=$rc" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error|KILL' $O/$tag.log $O/memguard.log | tail -2 >> $O/campaign.log; }
done
rm -rf $O/ssd-66b; log "== native 66B 종료"
