#!/bin/bash
# open-loop admission 검증: staging+비차단 admission이 Poisson 지속 부하에서
#   과거 지연 store(D1)처럼 뒤집히는지, tiering(C) 대비 유지되는지.
#   과거 W3: D1이 λ=0.55(>서비스율 0.39)에서 p50 132s로 발산. staging은 비차단이라
#   서비스율이 높을 것 — 그 가설을 검증.
set -u
cd "$(dirname "$0")"
source ../../env.sh
ARMS=(C D1 SVseen SVcpu)
RATES=(0.3 0.55)
for r in 1 2 3; do
  for arm in "${ARMS[@]}"; do
    for rate in "${RATES[@]}"; do
      id="ola-${arm}-p${rate}-r${r}"
      [ -f "../../results/leval-openloop/raw/${id}/meta.json" ] && { echo "skip $id"; continue; }
      free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
      [ "$free_gb" -lt 25 ] && { echo "DISK GUARD ${free_gb}G"; exit 1; }
      echo "=== $id ==="
      python openloop.py --run-id "$id" --arm "$arm" \
        --arrival poisson --rate "$rate" --docs 64 \
        --note "open-loop admission 검증" 2>&1 | tail -1
    done
  done
done
echo "ol_admission done"
