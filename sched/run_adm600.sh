#!/bin/bash
# admission 4단계: 600요청 확장. 150에서 random보다 useful hit 회복한 정책만.
#   러너별 arm: tiering / random_skip GDS / seen_twice GDS / seen_twice POSIX.
#   (oracle은 미래 지식 필요로 별도 dry-pass 요구 — 시간상 보류, 오프라인 oracle 참조.)
set -u
cd "$(dirname "$0")/.."
source env.sh
export PROF_INSTRUMENT=1
RUNNER=${1:-v2}
if [ "$RUNNER" = "v1" ]; then export VLLM_USE_V2_MODEL_RUNNER=0; BLK=256; SLOTS=2; TAG=v1b; else BLK=64; SLOTS=6; TAG=v2b; fi
SPECS=(
  "C:C:baseline:"
  "Sskip:S:baseline:--staging-policy skip --staging-slots $SLOTS"
  "Sseen:S:baseline:--staging-policy value --value-mode seen_twice --staging-slots $SLOTS --cpu-fallback-slots 8"
  "Pseen:P:baseline:--staging-policy value --value-mode seen_twice --staging-slots $SLOTS --cpu-fallback-slots 8"
)
for r in 1 2 3; do
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r name design pol extra <<< "$spec"
    id="adm600-${TAG}-${name}-r${r}"
    [ -f "sched/runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt 30 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
    echo "=== $id ==="
    python sched/run_bench.py --run-id "$id" --design "$design" \
      --policy "$pol" --block "$BLK" --n 600 $extra --note "admission 600 ${RUNNER}" 2>&1 | tail -1
  done
done
echo "adm600 ${RUNNER} done"
