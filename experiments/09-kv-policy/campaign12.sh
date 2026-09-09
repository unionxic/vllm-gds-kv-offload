#!/usr/bin/env bash
# 12: 빈 step 확정. ph-cufile 에서 step 이 36→1325 로 늘고 step 당 12.56s→0.33s 가 됐다.
#   집계만 남겨서 그 빈 step 이 KV 읽기 대기와 맞물리는지 확인할 수 없었다.
#   step 원본과 KV IO 타임라인을 그대로 저장해 다시 돌린다. 조건은 ph-cufile 과 동일.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f 'python run_phase_66b|python run_policy_66b' >/dev/null; do sleep 20; done
tag=${TAG:-ph-cufile-steps}
rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
log "== $tag (step 원본 저장)"
./memguard.sh $tag $O/memguard.log & guard=$!
python run_phase_66b.py --out-dir $O --kv-transport ${KVT:-cufile} --tag $tag > $O/$tag.log 2>&1
kill $guard 2>/dev/null; wait $guard 2>/dev/null
[ -f $O/$tag.json ] && grep -a '^RESULT' $O/$tag.log | cut -c1-300 || { log "FAILED $tag"; grep -aE 'Error|Traceback' $O/$tag.log | tail -3; }
log "12 완료"
