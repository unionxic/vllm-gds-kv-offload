#!/usr/bin/env bash
# 13: "어느 KV를 저장할지" — 재사용 이력 기반 admission 을 재사용이 치우친 워크로드에서 비교.
#   근거: Marconi(MLSys25)·HotPrefix(SIGMOD25)·Baleen(FAST24)·S3-FIFO(SOSP23) 가 공통으로 한 번만 보이는 항목(one-hit wonder)을
#   저장하지 않는 것을 admission 의 출발점으로 둔다. lib/value_admission 의 seen_twice 가 그 구현(두 번째 관측부터 저장).
#   워크로드: 3라운드 × 16프롬프트. 앞 4개는 라운드마다 같은 토큰열(재사용), 12개는 라운드마다 새 토큰열(1회성).
#   arm: none(재계산) / storeall(전부 저장, 07·09 baseline) / seen2(seen_twice admission, staged) / staged-block(staged 로 전부 저장 — seen2 의 전송 대조군)
#   측정: run_phase_66b.py — 라운드별 wall, forward 수, 재사용/1회성 프롬프트별 첫 토큰, 저장·읽기 바이트, staged 통계.
#   현재 스케줄러(로드 완료 순으로 승격)에서는 읽기가 파도를 쪼개 손해이므로, admission 이 읽기를 줄이는 만큼 이득이 나올 것.
#   스케줄러를 고친 뒤에는 읽기가 이득으로 바뀌어 같은 admission 이 손해가 될 수 있다 — 그 역전 자체가 결과.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
W="--rounds 3 --n-prompts 16 --hot-prompts 4 --out-dir $O"
run(){ local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  log "== $tag ($*)"
  ../07-combined/memguard.sh $tag $O/memguard.log & local guard=$!
  python run_phase_66b.py $W "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-300; grep -a '^WARN' $O/$tag.log; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Traceback|Killed' $O/$tag.log | tail -3; return 1
}
while pgrep -f 'python run_phase_66b|python run_policy_66b' >/dev/null; do sleep 20; done
log "13 admission 시작"
run adm-none         --kv-transport none
run adm-storeall     --kv-transport cufile
run adm-seen2        --kv-transport cufile_staged --staging-policy value --value-mode seen_twice
run adm-staged-block --kv-transport cufile_staged --staging-policy block
log "13 완료"
