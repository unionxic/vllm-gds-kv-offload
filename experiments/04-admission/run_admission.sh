#!/bin/bash
# admission 3단계: Bailian 150, 러너 내부 정책 비교. PROF 계측.
#   usage: run_admission.sh <v1|v2> [n]
#   arms: C(tiering) / Sskip(random_skip) / Sseen(seen_twice) / Svalue(value_density)
#   value가 random보다 개선되면 최선 정책에 posix-staging(Pvalue) 추가는 별도.
set -u
cd "$(dirname "$0")/.."
source env.sh
export PROF_INSTRUMENT=1
RUNNER=${1:-v2}
N=${2:-150}
if [ "$RUNNER" = "v1" ]; then export VLLM_USE_V2_MODEL_RUNNER=0; BLK=256; SLOTS=2; TAG=v1b; else BLK=64; SLOTS=6; TAG=v2b; fi
# name:design:policy:extra
SPECS=(
  "C:C:baseline:"
  "Sskip:S:baseline:--staging-policy skip --staging-slots $SLOTS"
  "Sseen:S:baseline:--staging-policy value --value-mode seen_twice --staging-slots $SLOTS --cpu-fallback-slots 8"
  "Svalue:S:baseline:--staging-policy value --value-mode value_density --staging-slots $SLOTS --cpu-fallback-slots 8"
)
for r in 1 2 3; do
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r name design pol extra <<< "$spec"
    id="adm-${TAG}-${name}-r${r}"
    [ -f "sched/runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt 25 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
    echo "=== $id ==="
    python sched/run_bench.py --run-id "$id" --design "$design" \
      --policy "$pol" --block "$BLK" --n "$N" $extra --note "admission ${RUNNER}" 2>&1 | tail -1
  done
done
echo "admission ${RUNNER} matrix done"
