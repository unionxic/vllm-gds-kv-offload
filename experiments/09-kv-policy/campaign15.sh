#!/usr/bin/env bash
# 15: 파도 게이트의 깨끗한 A/B. campaign14 의 gate-cufile-2r 은 forward 38→32 로 구조는 회복했으나 forward 하나가 13.3→15.2s 로
#   일제히 느려져(게이트가 개입하지 않는 쓰기 전용 1라운드까지) wall 비교가 오염됐다. 같은 시스템 상태에서 셋을 연달아 잰다.
#   판정: (게이트 cufile wall) < (재계산 wall) 이면 프리픽스 448토큰에서도 KV 로드가 이득으로 뒤집힌 것.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
# cuFile 1MiB bounce 경로가 간헐적으로 3배 느려지는 모드(08 RESULTS.md)에 이 A/B 가 오염되지 않도록 4MiB 조각을 쓴다(8MiB 는 KV 쓰기가 더해지면 BAR1 부족).
# 08 실측: 4MiB 이상 조각은 느린 모드를 타지 않음. 가중치 SSD 층과 KV 읽기 양쪽에 같이 적용되며 세 arm 에 동일.
export CUFILE_ENV_PATH_JSON=$(cd ../08-cufile-bounce && pwd)/cufile-pb4096.json
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  log "== $tag ($* GATE=${VLLM_KV_LOAD_WAVE_GATE:-0})"
  ./memguard.sh $tag $O/memguard.log & local guard=$!
  python run_phase_66b.py --out-dir $O "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-260; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Traceback|Killed' $O/$tag.log | tail -3; return 1
}
while pgrep -f 'campaign14\.s[h]|run_phase_66[b]\.py' >/dev/null; do sleep 20; done
log "15 깨끗한 A/B 시작 (CUFILE_ENV_PATH_JSON=$CUFILE_ENV_PATH_JSON)"
log "  cuFile 경로 상태: $(cd ../08-cufile-bounce && python bench_bounce.py $HOME/gds_test/test.bin 4 64 2 0 2>&1 | grep -oE '[0-9.]+ GiB/s')  (4MiB 조각, 정상 ≈3.2)"
VLLM_KV_LOAD_WAVE_GATE=0 run ab-none   --kv-transport none
VLLM_KV_LOAD_WAVE_GATE=0 run ab-cufile --kv-transport cufile
VLLM_KV_LOAD_WAVE_GATE=1 VLLM_KV_LOAD_WAVE_WAIT_S=30 run ab-gate --kv-transport cufile
log "15 완료"
