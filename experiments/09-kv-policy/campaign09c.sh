#!/usr/bin/env bash
# 09 프리픽스 확대: 448 → 1700 토큰. 시퀀스당 KV 1.0GiB → 3.6GiB.
#   448 구성에서 프리픽스 적중이 재계산보다 34% 느렸다(NOTES.md). 원인은 가중치 스트리밍이 prefill 시간을 정하기 때문.
#   프리픽스를 늘리면 KV 읽기 바이트가 비례해 커지고 재계산 쪽 forward 수는 8192토큰 한도까지 안 늘어난다.
#   그 균형이 어디서 뒤집히는지(또는 안 뒤집히는지) 실측한다.
#   자리 확보: GPU 상주 0층(num_in_group 64) → 가중치 전부 스트리밍(host 56층 + SSD 8층), 남는 자리를 KV에.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
COMMON="--host-fraction 0.85 --num-in-group 64 --prefix-tokens 1700 --tail-tokens 32 --max-model-len 1792 --n-prompts 6"
run(){ # tag kv transport
  local tag=$1 kv=$2 tr=$3
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  log "== $tag (kv ${kv}GiB, prefix 1700, $tr)"
  ../07-combined/memguard.sh $tag $O/memguard.log & local guard=$!
  python run_policy_66b.py $COMMON --kv-cache-gib $kv --out-dir $O --kv-transport $tr --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-500; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Killed|out of memory|KILL|Traceback' $O/$tag.log $O/memguard.log | tail -3; return 1
}
while pgrep -f 'run_policy_66b\.p[y]' >/dev/null; do sleep 20; done
# GPU KV 사다리: 프리픽스가 길어 prefill 활성화가 커지므로 여유를 두고 내려간다
KV=
for kv in 10.0 9.0 8.0; do
  if run L-kv$kv-none $kv none; then KV=$kv; break; fi
  log "kv $kv 실패 → 다음 단"; sleep 15
done
[ -z "$KV" ] && { log "프리픽스 확대 전부 실패, 중단"; exit 1; }
run L-kv$KV-cufile $KV cufile
run L-kv$KV-posix  $KV posix
log "09 프리픽스 확대 완료 (KV=$KV)"
