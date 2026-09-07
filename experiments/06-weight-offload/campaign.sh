#!/usr/bin/env bash
# 66B 캠페인: (현재 런 종료 대기) → ring QA(opt-2.7b) → 매트릭스 단계별 실행. 로그: results/weight-offload/opt66b/campaign.log
set -u
cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/weight-offload/opt66b
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f 'run_66b.p[y]' >/dev/null; do sleep 30; done
log "이전 런 종료 확인"
RING_OK=1   # ring QA는 22:41 PASS(qa-ring16.log) — 재실행 생략
log "== phase A: cufile/posix × h0.3 × r1-3 (기본)"
ARMS="cufile posix" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
if [ "$RING_OK" = 1 ]; then
  log "== phase B: cufile ring16 × h0.3 × r1-3"
  RING=16 ARMS="cufile" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
  log "== phase C: cufile ring8 + step2 + threads8 × h0.3 × r1-3 (threads8×2×16MiB=256MiB는 BAR1 초과라 슬롯 8MiB)"
  RING=8 STEP=2 THREADS=8 ARMS="cufile" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
fi
log "== phase D: step2 + threads8 (ring 없음) cufile/posix × h0.3 × r1-3"
STEP=2 THREADS=8 ARMS="cufile posix" FRACS="0.3" REPS="1 2 3" ./run_66b.sh
log "== phase E: host fraction 스윕 0.1/0.5 × cufile/posix × r1"
ARMS="cufile posix" FRACS="0.1 0.5" REPS="1" ./run_66b.sh
log "== phase F: nsys 런 cufile/posix × h0.3 × r1"
NSYS=1 ARMS="cufile posix" FRACS="0.3" REPS="1" ./run_66b.sh
log "== 추가: ring16 r4 (r3가 시스템 지연 이상치라 보강)"
RING=16 ARMS="cufile" FRACS="0.3" REPS="4" ./run_66b.sh
log "캠페인 완료"
