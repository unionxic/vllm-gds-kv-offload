#!/bin/bash
# V1 러너 + b256 재검증: 수정된 expfs로 과거 패배가 cuFile 탓인지
# event spin·블록 수명 결합 탓인지 분리. PROF 계측 포함. 각 3회.
# V1 b256 chunk=80MiB → ring 2슬롯(BAR1 256MB 내). 비차단 정책 포함.
set -u
cd "$(dirname "$0")/.."
source env.sh
export VLLM_USE_V2_MODEL_RUNNER=0
export PROF_INSTRUMENT=1
# name:design:policy:extra
SPECS=(
  "C:C:baseline:"
  "Dbev:D:baseline:"
  "Ddefbev:D:deferred_store:"
  "Sblock:S:baseline:--staging-slots 2 --staging-policy block"
  "Sskip:S:baseline:--staging-slots 2 --staging-policy skip"
  "Scpufb:S:baseline:--staging-slots 2 --staging-policy cpu_fallback --cpu-fallback-slots 8"
  "P:P:baseline:--staging-slots 2 --staging-policy skip"
)
for r in 1 2 3; do
  for spec in "${SPECS[@]}"; do
    IFS=':' read -r name design pol extra <<< "$spec"
    id="v1b-${name}-r${r}"
    [ -f "sched/runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    [ "$free_gb" -lt 25 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
    fix=""
    [ "$name" = "Dbev" -o "$name" = "Ddefbev" ] && fix="blocking_event"
    echo "=== $id ==="
    FIX="$fix" python sched/run_bench.py --run-id "$id" --design "$design" \
      --policy "$pol" --block 256 $extra --note "V1+b256 재검증" 2>&1 | tail -1
  done
done
echo "v1b256 matrix done"
