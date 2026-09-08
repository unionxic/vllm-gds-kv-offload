#!/usr/bin/env bash
# 08b: step2+thr8 조합을 4MiB 조각으로 재시도. 8MiB×16은 step2(정적 버퍼 풀 2배)에서 BAR1 부족으로 매핑 실패(20:11, dmesg 0x800000 no space).
#   campaign08.sh 종료 후 시작. run()/verify()는 campaign08.sh와 동일.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/cufile-bounce; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
json_for(){ [ "$1" = 1024 ] && echo "" || echo "$PWD/cufile-pb$1.json"; }
verify(){
  local pb=$1 j=$(json_for $1); rm -f $O/verify-pb$pb.log
  env ${j:+CUFILE_ENV_PATH_JSON=$j} CUFILE_LOGGING_LEVEL=TRACE CUFILE_LOGFILE_PATH=$O/verify-pb$pb.log python check_props.py 64 2>&1 | grep -E 'cuFileRead' | sed "s/^/  verify pb$pb: /"
  grep -oE 'size [0-9]{7,8}' $O/verify-pb$pb.log | sort | uniq -c | sort -rn | head -2 | sed "s/^/  verify pb$pb chunks: /"
}
run(){
  local tag=$1 pb=$2 step=$3 thr=$4 j=$(json_for $2)
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  local gpu_util=0.9; [ "$step" != 1 ] && gpu_util=0.75
  log "== $tag (per_buffer_cache_size_kb=$pb io_batchsize=$((131072/pb)) step=$step thr=$thr)"
  verify $pb
  env ${j:+CUFILE_ENV_PATH_JSON=$j} python ../06-weight-offload/run_66b.py --transport cufile --host-fraction 0.3 \
    --prefetch-step $step --io-threads $thr --ring-mb 0 --gpu-util $gpu_util --tag $tag --out $O/$tag.json > $O/$tag.log 2>&1
  grep -aq '^RESULT' $O/$tag.log || { log "FAILED $tag"; grep -aE 'Error|Killed|Traceback' $O/$tag.log | tail -3; return 1; }
  grep -a '^RESULT' $O/$tag.log | cut -c1-400
}
while pgrep -f 'campaign08\.s[h]|run_66b\.p[y]' >/dev/null; do sleep 30; done
log "08 종료 확인, 08b 시작"
run c-h0.3-pb4096-s2-t8-r1 4096 2 8
log "08b 완료"
