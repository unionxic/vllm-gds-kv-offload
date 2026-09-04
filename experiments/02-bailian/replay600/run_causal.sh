#!/bin/bash
# W1 causal-closure: V2에서 C-b64 / E-b64 / D-b64(재측정) — 동일 trace·순서.
#   인과 사슬: C-b16↔C-b64(bucketing) / C-b64↔E-b64(control plane) / E-b64↔D-b64(cuFile 단독)
# /dev/shm 원칙: 삭제하지 않음. replay.py가 [shm:before-engine]/[shm:after-shutdown]으로
# 관찰 기록, 회수는 엔진의 "Reclaimed" 로그로 검증, 정상 종료는 잔재 0 확인.
set -u
cd "$(dirname "$0")"
source ../env.sh
export PYTHONPATH=$PWD/../../../lib:${PYTHONPATH:-}

for spec in "C 64" "E 64" "D 64"; do
  set -- $spec
  echo "=== v2 $1 b$2 ==="
  echo "--- shm before (observe only) ---"
  ls -lh /dev/shm/vllm_offload_*.mmap 2>/dev/null || echo "(none)"
  # 기존 v2-D-b64 결과 보존: 재측정 totals가 같은 이름을 덮으므로 원본 백업
  [ "$1" = "D" ] && [ -f w1_total_v2-D-b64.json ] && [ ! -f w1_total_v2-D-b64_orig.json ] \
    && cp w1_total_v2-D-b64.json w1_total_v2-D-b64_orig.json
  python replay.py --design "$1" --runner v2 --block "$2" --n-requests 600 --out w1_causal.csv
  rc=$?
  echo "run exit=$rc"
  rm -rf "kvroot-v2-$1-b$2"   # 디스크상 kvroot는 우리 소유 데이터 — 용량 회수
  echo "--- shm after (observe only) ---"
  ls /dev/shm/vllm_offload_*.mmap 2>/dev/null || echo "(none)"
  df -h / | tail -1
done
