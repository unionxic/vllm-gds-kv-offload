#!/usr/bin/env bash
# 07 캠페인: OPT-66B 가중치 3단(GPU 4층 + host fraction + SSD) + KV→SSD(expfs). 프리픽스 재사용 2라운드.
#   레짐 1: host 0.92(RAM 실제 포화; 실패 시 0.9) × KV cufile/posix/none
#   레짐 2: host 0.3(가중치 스트리밍이 SSD 포화) × KV cufile/posix
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/combined/opt66b; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ # tag, args...
  local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  log "== $tag"
  python run_combo_66b.py "$@" --tag $tag > $O/$tag.log 2>&1
  grep -a '^RESULT' $O/$tag.log | cut -c1-600 || { log "FAILED $tag"; grep -aE 'Error|Killed|out of memory' $O/$tag.log | tail -3; return 1; }
}
while pgrep -f 'smoke_comb[o]' >/dev/null; do sleep 15; done
# 레짐 1: host 최대. 0.92가 pinned 확보에 실패하면 0.9로 후퇴
HF=0.92
if ! run h0.92-kvcufile --host-fraction 0.92 --kv-transport cufile; then
  log "0.92 실패 → 0.9로 후퇴"; HF=0.9
  run h0.9-kvcufile --host-fraction 0.9 --kv-transport cufile
fi
run h${HF}-kvposix  --host-fraction $HF --kv-transport posix
run h${HF}-kvnone   --host-fraction $HF --kv-transport none
# 레짐 2: 가중치 스트리밍 최악(06과 동일 배치) + KV
run h0.3-kvcufile --host-fraction 0.3 --kv-transport cufile
run h0.3-kvposix  --host-fraction 0.3 --kv-transport posix
log "07 캠페인 완료"
