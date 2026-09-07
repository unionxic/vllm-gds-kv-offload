#!/usr/bin/env bash
# 66B 캠페인: (현재 런 종료 대기) → ring QA(opt-2.7b) → 매트릭스 단계별 실행. 로그: results/weight-offload/opt66b/campaign.log
set -u
cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/weight-offload/opt66b
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f 'run_66b.p[y]' >/dev/null; do sleep 30; done
log "이전 런 종료 확인"
log "== ring QA (opt-2.7b)"
python smoke_ring.py 16 1 4 > $O/qa-ring16.log 2>&1; grep -a -E '^RESULT|^QA' $O/qa-ring16.log
grep -q '^QA PASS' $O/qa-ring16.log || { log "ring QA FAIL → ring arm 제외"; RING_OK=0; }
RING_OK=${RING_OK:-1}
log "== phase A: cufile/posix × h0.3 × r1-3 (기본)"
ARMS="cufile posix" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
if [ "$RING_OK" = 1 ]; then
  log "== phase B: cufile ring16 × h0.3 × r1-3"
  RING=16 ARMS="cufile" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
  log "== phase C: cufile ring16 + step2 + threads8 × h0.3 × r1-3"
  RING=16 STEP=2 THREADS=8 ARMS="cufile" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
fi
log "== phase D: step2 + threads8 (ring 없음) cufile/posix × h0.3 × r1-3"
STEP=2 THREADS=8 ARMS="cufile posix" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
log "== phase E: host fraction 스윕 0.1/0.5 × cufile/posix × r1"
ARMS="cufile posix" FRACS="0.1 0.5" REPS="1" ./run_66b.sh
log "== phase F: nsys 런 cufile/posix × h0.3 × r1"
NSYS=1 ARMS="cufile posix" FRACS="0.3" REPS="1" ./run_66b.sh
log "캠페인 완료"
