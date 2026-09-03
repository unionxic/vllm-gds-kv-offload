#!/bin/bash
# 비차단 staging 정책 비교 (V2 b64, Bailian 150, 각 3회, PROF 계측).
# 판정: 블록 점유(JOB_HOLD)가 D2D 시간으로 제한되는가, foreground가 슬롯 대기 안 하는가.
set -u
cd "$(dirname "$0")/.."
source env.sh
export PROF_INSTRUMENT=1
# name:design:policy:extra
SPECS=(
  "C:C:baseline:"
  "Sblock:S:baseline:--staging-policy block"
  "Sskip:S:baseline:--staging-policy skip"
  "Scpufb:S:baseline:--staging-policy cpu_fallback --cpu-fallback-slots 8"
)
for r in 1 2 3; do
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r name design pol extra <<< "$spec"
    id="nb-${name}-r${r}"
    [ -f "sched/runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt 25 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
    echo "=== $id ==="
    python sched/run_bench.py --run-id "$id" --design "$design" \
      --policy "$pol" --block 64 $extra --note "비차단 정책 비교" 2>&1 | tail -1
  done
done
echo "nonblock matrix done"
