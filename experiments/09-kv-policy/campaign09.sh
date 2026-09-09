#!/usr/bin/env bash
# 09 1단계: KV가 실제로 배치를 이루는 구성에서 transport 재비교.
#   07은 GPU KV 1.5GiB라 480토큰 프롬프트 하나만 들어가 batch=1 순차 → forward 146회, KV가 SSD 트래픽의 0.28~2.8%뿐.
#   여기서는 GPU 상주를 4층→2층으로 줄여(num_in_group 62) 확보한 자리를 KV에 주고(5.5GiB) 배치 5~6을 만든다.
#   그러면 forward 수가 1/5로 줄어 가중치 트래픽이 같이 줄고, SSD에서 가중치 6층(11.4GiB/forward)과
#   KV 로드(배치당 5~6GiB)가 비슷한 크기로 경합한다. 2단계 정책(deferred store, read priority)이 의미를 갖는 지점.
#   host 0.85 고정(가중치 대부분 RAM, 디스크는 KV와 6층만) + PIN_EXACT 필수.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ # tag kv_gib transport
  local tag=$1 kv=$2 tr=$3
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  log "== $tag (kv ${kv}GiB, 상주 2층, $tr)"
  ./memguard.sh $tag $O/memguard.log & local guard=$!
  python run_policy_66b.py --host-fraction 0.85 --num-in-group 62 --kv-cache-gib $kv \
    --n-prompts 16 --out-dir $O --kv-transport $tr --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-500; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Killed|out of memory|KILL|Traceback' $O/$tag.log $O/memguard.log | tail -3; return 1
}
while pgrep -f 'run_66b\.p[y]|run_combo_66b\.p[y]' >/dev/null; do sleep 20; done
# GPU KV 5.5GiB 확정(첫 런에서 gpu_max 12.74GiB, 예산 14.4GiB 안). 사다리 불필요.
KV=5.5
run b-kv$KV-cufile $KV cufile
run b-kv$KV-posix $KV posix
run b-kv$KV-none  $KV none
log "09 1단계 완료 (KV=$KV)"
