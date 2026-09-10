#!/usr/bin/env bash
# 11: 구간 분리 측정. campaign10 종료 후 시작.
#   기존 decode 손해는 generate 두 번의 차이로 구한 값이라 KV 로드 시간이 섞였을 수 있다.
#   run_phase_66b.py 로 엔진 step 을 직접 돌려 구간을 나누고, decode step 에 KV 읽기가 실제로 겹치는지 잰다.
#   조건은 기존 b-kv5.5-* 와 동일(프리픽스 448, 16프롬프트, GPU KV 5.5GiB, 상주 2층, host 0.85).
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  log "== $tag ($*)"
  ../07-combined/memguard.sh $tag $O/memguard.log & local guard=$!
  "$@" > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-700; grep -a '^WARN' $O/$tag.log; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Traceback|Killed|잔여' $O/$tag.log | tail -3; return 1
}
while pgrep -f 'campaign10\.s[h]|run_policy_66b\.p[y]|run_phase_66b\.p[y]' >/dev/null; do sleep 20; done
log "11 구간 분리 시작"
P="python run_phase_66b.py --out-dir $O"
run ph-none   $P --kv-transport none   --tag ph-none
run ph-cufile $P --kv-transport cufile --tag ph-cufile
log "11 nsys 런(중첩 육안 확인용)"
run ph-cufile-nsys /usr/local/cuda/bin/nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false --sample=none \
  --cpuctxsw=none -o $O/ph-cufile-nsys --force-overwrite true \
  python run_phase_66b.py --out-dir $O --kv-transport cufile --tag ph-cufile-nsys
log "11 완료"
