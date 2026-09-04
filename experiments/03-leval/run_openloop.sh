#!/bin/bash
# open-loop/동시성 행렬 1차(closed-loop 스위프). GPU 단독 사용 전제, 순차 실행.
set -u
cd "$(dirname "$0")/../.."
source env.sh
for conc in 1 2 4 8; do
  for arm in C D1 DS4; do
    id="ol-${arm}-c${conc}-r1"
    if [ -f "results/leval-openloop/raw/${id}/meta.json" ]; then
      echo "skip ${id}"; continue
    fi
    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
    if [ "$free_gb" -lt 25 ]; then echo "DISK GUARD ${free_gb}G"; exit 1; fi
    echo "=== ${id} ==="
    python experiments/03-leval/openloop.py --run-id "$id" --arm "$arm" --conc "$conc" \
      2>&1 | tail -5
  done
done
echo "closed sweep done"
