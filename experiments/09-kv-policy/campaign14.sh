#!/usr/bin/env bash
# 14: 스케줄러 파도 게이트(VLLM_KV_LOAD_WAVE_GATE) A/B. campaign13 종료 후.
#   원인: 로드가 먼저 끝난 요청이 혼자 forward 를 돌아 파도가 쪼개짐(파도당 +2 forward). 게이트는 같은 파도의 로드가
#   모두 끝날 때까지(최대 WAVE_WAIT_S) 승격을 미룬다. 판정: forward 수 38→32, 라운드2 wall 508→444 근처.
#   A: 2라운드·전부 반복(ph-cufile-steps 와 동일 조건) — 직접 비교용
#   B: 3라운드·4개 반복(adm-storeall 과 동일 조건) — admission 결과와 대조용
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
export VLLM_KV_LOAD_WAVE_GATE=1 VLLM_KV_LOAD_WAVE_WAIT_S=30
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  log "== $tag ($*)"
  ../07-combined/memguard.sh $tag $O/memguard.log & local guard=$!
  python run_phase_66b.py --out-dir $O "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-300; grep -a '^WARN' $O/$tag.log; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Traceback|Killed' $O/$tag.log | tail -3; return 1
}
while pgrep -f 'campaign13\.s[h]|python run_phase_66b|python run_policy_66b' >/dev/null; do sleep 20; done
log "14 파도 게이트 시작"
run gate-cufile-2r  --kv-transport cufile
run gate-storeall   --rounds 3 --n-prompts 16 --hot-prompts 4 --kv-transport cufile
log "14 완료"
