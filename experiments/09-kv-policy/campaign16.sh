#!/usr/bin/env bash
# 16: 섞인 파도(재사용 4 + 1회성 12, 3라운드)에서 게이트 수준 비교. campaign15 종료 후.
#   gate-storeall(수준 1)은 로드 없는 1회성 요청이 먼저 나가 파도를 쪼갰다(forward 35, 패턴 0,1,1,5...). 수준 2는
#   동료가 로드 중이면 로드 없는 신규 요청도 잡아둔다. 판정: 수준 2에서 forward 35→32, 라운드2·3 wall 이 재계산 아래로.
#   앞선 gate-storeall 은 cuFile 느린 모드(forward 19s)라 wall 비교 불가 → 네 arm 모두 4MiB 조각으로 다시 잰다(8MiB 는 KV 쓰기까지 더해지면 BAR1 부족으로 cuFileWrite 실패).
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
export CUFILE_ENV_PATH_JSON=$(cd ../08-cufile-bounce && pwd)/cufile-pb4096.json
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
W="--rounds 3 --n-prompts 16 --hot-prompts 4 --out-dir $O"
run(){ local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  log "== $tag ($* GATE=${VLLM_KV_LOAD_WAVE_GATE:-0})"
  ./memguard.sh $tag $O/memguard.log & local guard=$!
  python run_phase_66b.py $W "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-260; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Traceback|Killed' $O/$tag.log | tail -3; return 1
}
while pgrep -f 'bash ./campaign1[5]|run_phase_66[b]\.py' >/dev/null; do sleep 20; done
log "16 섞인 파도 게이트 비교 시작"
log "  cuFile 경로 상태: $(cd ../08-cufile-bounce && python bench_bounce.py $HOME/gds_test/test.bin 4 64 2 0 2>&1 | grep -oE '[0-9.]+ GiB/s')"
VLLM_KV_LOAD_WAVE_GATE=0 run mx-none   --kv-transport none
VLLM_KV_LOAD_WAVE_GATE=0 run mx-gate0  --kv-transport cufile
VLLM_KV_LOAD_WAVE_GATE=1 run mx-gate1  --kv-transport cufile
VLLM_KV_LOAD_WAVE_GATE=2 run mx-gate2  --kv-transport cufile
log "16 완료"
