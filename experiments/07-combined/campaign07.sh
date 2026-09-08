#!/usr/bin/env bash
# 07 캠페인: OPT-66B 가중치 3단(GPU 4층 + host fraction + SSD) + KV→SSD(expfs). 프리픽스 재사용 2라운드.
#   레짐 1: host 0.85(실패 시 0.80→0.75; RAM 실제 포화, PIN_EXACT) × KV cufile/posix/none
#   레짐 2: host 0.3(가중치 스트리밍이 SSD 포화) × KV cufile/posix
set -u -o pipefail; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
# pinned 호스트 티어를 cudaHostRegister 정확 크기로(캐싱 할당자의 2^n 올림 제거, 09-08 14:45 확인: 올림 시 1.405배)
export VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/combined/opt66b; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ # tag, args...
  local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  log "== $tag"
  ./memguard.sh $tag $O/memguard.log & local guard=$!
  python run_combo_66b.py "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  # 성공 판정은 json 존재로(파이프 종료코드는 cut의 것이라 못 믿음 — 09-08 14:33 사다리 미작동 원인)
  [ -f $O/$tag.json ] && grep -a '^RESULT' $O/$tag.log | cut -c1-600 && return 0
  log "FAILED $tag"; grep -aE 'Error|Killed|out of memory|KILL' $O/$tag.log $O/memguard.log | tail -3; return 1
}
while pgrep -f 'smoke_comb[o]' >/dev/null; do sleep 15; done
# 레짐 1: host 최대. 0.92는 머신 자체가 죽었음(09-08 14:04). 올림(1.45배) 상태에서는 0.85~0.70 전부 워치독 KILL.
#   VLLM_OFFLOAD_PIN_EXACT=1로 올림을 없앤 뒤 0.85(pinned 106GiB+오버헤드 ~4GiB) → 0.80 → 0.75 사다리.
HF=
for hf in 0.85 0.80 0.75; do
  if run h${hf}-kvcufile --host-fraction $hf --kv-transport cufile; then HF=$hf; break; fi
  log "host $hf 실패 → 다음 단으로 후퇴"; sleep 20
done
[ -z "$HF" ] && { log "레짐1 전부 실패, 중단"; exit 1; }
run h${HF}-kvposix  --host-fraction $HF --kv-transport posix
run h${HF}-kvnone   --host-fraction $HF --kv-transport none
# 레짐 2: 가중치 스트리밍 최악(06과 동일 배치) + KV
run h0.3-kvcufile --host-fraction 0.3 --kv-transport cufile
run h0.3-kvposix  --host-fraction 0.3 --kv-transport posix
log "07 캠페인 완료"
