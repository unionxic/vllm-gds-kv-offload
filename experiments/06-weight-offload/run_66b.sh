#!/usr/bin/env bash
# OPT-66B 매트릭스: transport × host_fraction × 반복. 성능 런과 nsys 런 분리.
#   ./run_66b.sh                      # 기본: cufile/posix × h0.3 × r1..r3
#   ARMS="cufile posix" FRACS="0.1 0.3 0.5" REPS="1" ./run_66b.sh
#   NSYS=1 ARMS="cufile posix" FRACS="0.3" REPS="1" ./run_66b.sh   # nsys 타임라인 런(수치는 참고용)
set -u
cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
ARMS=${ARMS:-"cufile posix"}; FRACS=${FRACS:-"0.3"}; REPS=${REPS:-"1 2 3"}; NSYS=${NSYS:-0}
STEP=${STEP:-1}; THREADS=${THREADS:-4}; RING=${RING:-0}   # 정책 축: prefetch_step / io_threads / ring slot MiB
SUF=""; [ "$STEP" != 1 ] && SUF="$SUF-s$STEP"; [ "$THREADS" != 4 ] && SUF="$SUF-t$THREADS"; [ "$RING" != 0 ] && SUF="$SUF-ring$RING"
# step≥2는 정적 버퍼 풀이 두 배(3.8GiB)인데 프로파일에 안 잡혀 KV 산정 후 OOM → gpu_util 0.75로 여유 확보(KV≈1,450tok ≥ 1,056 필요)
GPU_UTIL=${GPU_UTIL:-$([ "$STEP" != 1 ] && echo 0.75 || echo 0.9)}
EXTRA="--prefetch-step $STEP --io-threads $THREADS --ring-mb $RING --gpu-util $GPU_UTIL"
OUT=../../results/weight-offload/opt66b; mkdir -p $OUT
for f in $FRACS; do for a in $ARMS; do for r in $REPS; do
  tag="${a:0:1}-h${f}${SUF}-r${r}"; [ "$NSYS" = 1 ] && tag="${tag}-nsys"
  [ -f $OUT/$tag.json ] && { echo "skip $tag (exists)"; continue; }
  rm -rf ../../results/weight-offload/ssd-66b   # 런마다 재생성되는 공유 SSD 티어(80GiB) 정리 후 여유 확인
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { echo "disk free ${free_gb}G < 90G, abort"; exit 1; }
  echo "== $tag $(date +%H:%M:%S)"
  if [ "$NSYS" = 1 ]; then
    /usr/local/cuda/bin/nsys profile -t cuda,nvtx,osrt --cuda-memory-usage=false --sample=none --cpuctxsw=none \
      -o $OUT/$tag --force-overwrite true \
      python run_66b.py --transport $a --host-fraction $f $EXTRA --tag $tag > $OUT/$tag.log 2>&1
  else
    python run_66b.py --transport $a --host-fraction $f $EXTRA --tag $tag > $OUT/$tag.log 2>&1
  fi
  grep -a '^RESULT' $OUT/$tag.log | cut -c1-300 || { echo "FAILED $tag"; tail -5 $OUT/$tag.log; }
done; done; done
