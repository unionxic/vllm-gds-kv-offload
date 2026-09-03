#!/bin/bash
# V1 러너 + b256 재검증: 수정된 expfs로 과거 패배가 cuFile 탓인지
# event spin·블록 수명 결합 탓인지 분리. 5 비교군 × 3회, PROF 계측 포함.
set -u
cd "$(dirname "$0")/.."
source env.sh
export VLLM_USE_V2_MODEL_RUNNER=0
export PROF_INSTRUMENT=1
for r in 1 2 3; do
  for spec in "C:C:baseline:16::" \
              "Dbev:D:baseline:256:blocking_event:" \
              "Ddefbev:D:deferred_store:256:blocking_event:" \
              "S:S:baseline:256::--staging-slots 2" \
              "P:P:baseline:256::--staging-slots 2"; do
    IFS=':' read -r name design pol blk fix extra <<< "$spec"
    id="v1b-${name}-r${r}"
    [ -f "sched/runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt 25 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
    echo "=== $id ==="
    FIX="$fix" python sched/run_bench.py --run-id "$id" --design "$design" \
      --policy "$pol" --block "$blk" $extra --note "V1+b256 재검증" 2>&1 | tail -1
  done
done
echo "v1b256 matrix done"
