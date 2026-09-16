#!/usr/bin/env bash
# 3B 소스 섞기 격자: 재계산 비율 K × host 비율 G × 모드(split=앞 계산·뒤 적재 동시, serial=앞 적재 뒤 계산 직렬).
# 조건: Qwen2.5-3B, host 0.02(가중치 host 17/SSD 19 layer), Bailian 24건 4k, KV 예산 2요청. 결과 results/grid3b/<tag>.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=$(mkdir -p ../../results/grid3b && cd ../../results/grid3b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
KS=${KS:-"0 0.25 0.5 0.75 1"}; GS=${GS:-"0 0.5 1"}; MODES=${MODES:-"split serial"}; HOSTGB=${HOSTGB:-1.8}
log "== grid3b K($KS) × G($GS) × ($MODES)"
for g in $GS; do for k in $KS; do for mode in $MODES; do
  tag=k$k-g$g-$mode; [ -f $O/$tag/result.json ] && { log "skip $tag"; continue; }
  case $g in 0) kvt=(--kv-transport cufile);; 1) kvt=(--kv-transport cpu --kv-host-gb 8);; *) kvt=(--kv-transport hybrid --kv-host-gb $HOSTGB);; esac
  case $k in 1) kvt=(--kv-transport none); split=off;; *) split=$mode:$k;; esac
  [ "$k" = 0 ] && split=off
  rm -rf $O/kv; log "start $tag"
  python run_obs.py --run-dir $O/$tag --model Qwen/Qwen2.5-3B-Instruct --prompt-source bailian --n-docs 24 --decode-tokens 8 --host-ram-fraction 0.02 \
    --kv-batch 2 --max-model-len 4096 --settle-sec 3 --final-settle-sec 3 --poll-sleep-ms 0 --ssd-root $O/ssd --kv-root $O/kv "${kvt[@]}" --kv-split $split > $O/$tag.log 2>&1
  rc=$?; rm -rf $O/kv
  [ -f $O/$tag/result.json ] && log "done  $tag rc=$rc" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error' $O/$tag.log | tail -1 >> $O/campaign.log; }
done; done; done
rm -rf $O/ssd; log "== grid3b 종료"
