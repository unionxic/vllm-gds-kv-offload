#!/usr/bin/env bash
# 08: cuFile 미등록 버퍼 우회 경로(bounce cache)의 조각 크기 실험.
#   배경: 정적 버퍼(fc1/fc2 680MB)가 BAR1 256MiB에 등록되지 않아 cuFile이 내부 GPU 캐시(per_buffer_cache_size_kb=1024)로
#   1MiB씩 DMA 후 D2D로 옮김(06 §2.1: 82만 회). 06은 ring(8MiB 등록 슬롯)으로 우회해 step2+thr8과 함께 6% 이득.
#   질문: cufile.json의 조각 크기만 4/8/16MiB로 키우면(ring 없이) 같은 이득이 나오나. 제약: cache(128MiB)/per_buffer >= io_batchsize
#   → io_batchsize를 32/16/8로 낮춤(동기 cuFileRead만 쓰므로 무관). 06 기본 구성(h0.3, cufile, step1, thr4)과 동일 조건.
#   07 캠페인이 끝날 때까지 대기 후 시작. 결과: results/cufile-bounce/<tag>.json|.log, 로그: results/cufile-bounce/campaign08.log
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/cufile-bounce; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
json_for(){ [ "$1" = 1024 ] && echo "" || echo "$PWD/cufile-pb$1.json"; }
verify(){ # pb → 64MiB 미등록 읽기의 조각 크기 히스토그램(TRACE)으로 설정 적용 확인
  local pb=$1 j=$(json_for $1); rm -f $O/verify-pb$pb.log
  env ${j:+CUFILE_ENV_PATH_JSON=$j} CUFILE_LOGGING_LEVEL=TRACE CUFILE_LOGFILE_PATH=$O/verify-pb$pb.log python check_props.py 64 2>&1 | grep -E 'cuFileRead' | sed "s/^/  verify pb$pb: /"
  grep -oE 'size [0-9]{7,8}' $O/verify-pb$pb.log | sort | uniq -c | sort -rn | head -2 | sed "s/^/  verify pb$pb chunks: /"
  grep -h 'default 1MB' $O/verify-pb$pb.log | head -1 | sed "s/^/  verify pb$pb OVERRIDE: /"
}
run(){ # tag pb step threads
  local tag=$1 pb=$2 step=$3 thr=$4 j=$(json_for $2)
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  local gpu_util=0.9; [ "$step" != 1 ] && gpu_util=0.75   # 06 run_66b.sh와 동일(step2는 정적 버퍼 풀 2배)
  log "== $tag (per_buffer_cache_size_kb=$pb io_batchsize=$((131072/pb)) step=$step thr=$thr)"
  verify $pb
  env ${j:+CUFILE_ENV_PATH_JSON=$j} python ../06-weight-offload/run_66b.py --transport cufile --host-fraction 0.3 \
    --prefetch-step $step --io-threads $thr --ring-mb 0 --gpu-util $gpu_util --tag $tag --out $O/$tag.json > $O/$tag.log 2>&1
  grep -aq '^RESULT' $O/$tag.log || { log "FAILED $tag"; grep -aE 'Error|Killed|Traceback' $O/$tag.log | tail -3; return 1; }
  grep -a '^RESULT' $O/$tag.log | cut -c1-400
}
while pgrep -f 'campaign07\.s[h]|run_combo_66b\.p[y]|run_66b\.p[y]' >/dev/null; do sleep 30; done
log "07 캠페인 종료 확인, 08 시작"
run c-h0.3-pb1024-r1 1024 1 4
BEST=1024
for pb in 4096 8192 16384; do run c-h0.3-pb$pb-r1 $pb 1 4 && BEST=$pb; done
log "가장 큰 성공 조각: ${BEST}KiB → step2 thr8 arm과 반복에 사용"
run c-h0.3-pb$BEST-s2-t8-r1 $BEST 2 8
run c-h0.3-pb$BEST-r2 $BEST 1 4
run c-h0.3-pb1024-r2 1024 1 4
log "08 캠페인 완료"
