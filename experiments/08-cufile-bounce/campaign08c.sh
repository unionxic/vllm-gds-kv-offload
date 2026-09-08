#!/usr/bin/env bash
# 08c: 1MiB 대조군이 06(28s)보다 2.3배 느린(66s, 2회 재현) 원인 분리. 08b 종료 후 시작.
#   (1) gdsio로 cuFile 원시 경로 속도: 미등록 bounce 1MiB / 4MiB, 등록 1MiB (각 10초)
#   (2) A/B 런: 대조군 + VLLM_OFFLOAD_PIN_EXACT=1 (07 h0.3 런과 같은 조건), 대조군 + num_in_group 60 (GPU 상주 4층, 07과 동일 BAR1 배치)
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/cufile-bounce; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
G=/usr/local/cuda/gds/tools/gdsio; T=$HOME/gds_test; mkdir -p $T
while pgrep -f 'campaign08b\.s[h]|run_66b\.p[y]' >/dev/null; do sleep 30; done
log "08b 종료 확인, 08c 시작"
[ -f $T/test.bin ] && [ "$(stat -c %s $T/test.bin)" -ge 2147483648 ] || $G -f $T/test.bin -d 0 -w 4 -s 2G -i 1M -x 0 -I 1 >/dev/null 2>&1
for cfg in "bounce-1M:-b -i 1M" "bounce-4M:-b -i 4M" "registered-1M:-i 1M" "registered-4M:-i 4M"; do
  name=${cfg%%:*}; opts=${cfg#*:}
  log "GDSIO $name: $($G -f $T/test.bin -d 0 -w 4 -s 2G $opts -x 0 -I 0 -T 10 2>&1 | grep -oE 'Throughput: [0-9.]+ GiB/sec|Avg_Latency: [0-9.]+ usecs|IOs: [0-9]+' | tr '\n' ' ')"
done
run(){ # tag extra-args... (env VLLM_OFFLOAD_PIN_EXACT은 호출측에서)
  local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b
  log "== $tag ($* PIN_EXACT=${VLLM_OFFLOAD_PIN_EXACT:-0})"
  python ../06-weight-offload/run_66b.py --transport cufile --host-fraction 0.3 --prefetch-step 1 --io-threads 4 --ring-mb 0 --gpu-util 0.9 "$@" --tag $tag --out $O/$tag.json > $O/$tag.log 2>&1
  grep -aq '^RESULT' $O/$tag.log || { log "FAILED $tag"; grep -aE 'Error|Killed|Traceback' $O/$tag.log | tail -3; return 1; }
  grep -a '^RESULT' $O/$tag.log | cut -c1-400
}
VLLM_OFFLOAD_PIN_EXACT=1 run c-h0.3-pb1024-pinexact-r1
run c-h0.3-pb1024-g60-r1 --num-in-group 60
log "08c 완료"
