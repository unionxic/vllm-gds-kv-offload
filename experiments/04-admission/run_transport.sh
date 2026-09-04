#!/bin/bash
# admission 결론 ②: champion(seen_twice)에서 cuFile-staging vs posix-staging.
#   동일 admission·제어부, transport만 교체 → cuFile 순수 효과 분리.
#   usage: run_transport.sh <v1|v2>
set -u
cd "$(dirname "$0")/../.."
source env.sh
export PROF_INSTRUMENT=1
RUNNER=${1:-v2}
if [ "$RUNNER" = "v1" ]; then export VLLM_USE_V2_MODEL_RUNNER=0; BLK=256; SLOTS=2; TAG=v1b; else BLK=64; SLOTS=6; TAG=v2b; fi
# cuFile은 이미 adm-${TAG}-Sseen로 측정됨. 여기선 posix-staging seen_twice만.
for r in 1 2 3; do
  id="adm-${TAG}-Pseen-r${r}"
  [ -f "results/bailian/window150-runs/${id}/meta.json" ] && { echo "skip $id"; continue; }
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "$free_gb" -lt 25 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
  echo "=== $id ==="
  python harness/run_bench.py --run-id "$id" --design P \
    --policy baseline --block "$BLK" --n 150 \
    --staging-policy value --value-mode seen_twice \
    --staging-slots "$SLOTS" --cpu-fallback-slots 8 \
    --note "transport 분리 ${RUNNER}" 2>&1 | tail -1
done
echo "transport ${RUNNER} done"
