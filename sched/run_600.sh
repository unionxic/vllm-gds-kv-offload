#!/bin/bash
# 600요청 확장: 비차단 정책이 150요청 게이트를 통과 → 최선 구성으로 규모 확대.
# V2 b64. 대조군(tiering, staging block) + 최선(skip, cpu_fallback). 각 3회.
# kvroot가 런당 최대 ~200GB라 매 런 후 삭제(run_bench가 처리) + 디스크 가드.
set -u
cd "$(dirname "$0")/.."
source env.sh
export PROF_INSTRUMENT=1
SPECS=(
  "C:C:baseline:"
  "Sblock:S:baseline:--staging-policy block"
  "Sskip:S:baseline:--staging-policy skip"
  "Scpufb:S:baseline:--staging-policy cpu_fallback --cpu-fallback-slots 8"
)
for r in 1 2 3; do
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r name design pol extra <<< "$spec"
    id="s600-${name}-r${r}"
    [ -f "sched/runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt 30 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
    echo "=== $id ==="
    python sched/run_bench.py --run-id "$id" --design "$design" \
      --policy "$pol" --block 64 --n 600 $extra --note "600요청 확장" 2>&1 | tail -1
  done
done
echo "s600 matrix done"
