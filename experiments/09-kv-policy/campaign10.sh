#!/usr/bin/env bash
# 10: KV 읽기 손해를 세 성분으로 분해한다.
#   관측: 프리픽스 448, 16프롬프트에서 KV 31.5GiB를 읽는데 재계산 대비 손해가 24.5초.
#   디스크 대역폭(2.9GB/s)으로는 11초면 되는 양이므로 13초가 설명되지 않는다.
#   프리픽스 캐시를 안 쓰는 decode 구간도 44.5→50.8초로 느려졌으니 가중치 스트림과의 경합이 의심된다.
#   손해 = 바이트 비례분 + IO 건수 비례분(고정 오버헤드) + 가중치 경합분 으로 보고 축을 하나씩 흔든다.
#     축 A 바이트: 프롬프트 16→8. 바이트와 IO 건수가 함께 절반.
#     축 B IO 건수: kv_block 64→256. 바이트는 그대로, 청크가 4배 커져 IO 건수가 4분의 1.
#     축 C 경합: host_fraction 0.85→0.75. host 티어가 줄어 가중치 SSD 층이 6층(11.4GiB)에서 13층(24.7GiB)으로 증가.
#   각 축마다 KV 저장 안 함 기준선을 따로 잡아 차이를 본다. 기존 b-kv5.5-{none,cufile}이 기본 조건(16프롬프트, block64, h0.85).
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ # tag, args...
  local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  log "== $tag ($*)"
  ./memguard.sh $tag $O/memguard.log & local guard=$!
  python run_policy_66b.py --num-in-group 62 --kv-cache-gib 5.5 --out-dir $O "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-420; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Killed|out of memory|KILL|Traceback' $O/$tag.log $O/memguard.log | tail -3; return 1
}
while pgrep -f 'run_policy_66b\.p[y]' >/dev/null; do sleep 20; done
log "10 비용 분해 시작"
# 축 A: 바이트와 IO 건수를 함께 절반
run c-n8-none    --host-fraction 0.85 --n-prompts 8 --kv-transport none
run c-n8-cufile  --host-fraction 0.85 --n-prompts 8 --kv-transport cufile
# 축 B: 바이트 고정, IO 건수만 4분의 1 (기준선은 기존 b-kv5.5-none 재사용)
run c-blk256-cufile --host-fraction 0.85 --n-prompts 16 --kv-block 256 --kv-transport cufile
# 축 C: 가중치 SSD 트래픽 2배
run c-h0.75-none   --host-fraction 0.75 --n-prompts 16 --kv-transport none
run c-h0.75-cufile --host-fraction 0.75 --n-prompts 16 --kv-transport cufile
log "10 비용 분해 완료"
