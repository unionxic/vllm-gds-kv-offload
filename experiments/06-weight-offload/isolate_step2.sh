#!/usr/bin/env bash
# step2 토큰 불일치 원인 분리(opt-2.7b): cpu/ssd-cufile step2(ring 없음) → ring16 step2 thr4 → ring16 step1 thr8 → ring8 step2 thr8
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 10; done
echo "[$(date +%H:%M:%S)] GPU free"
echo "== qa_ssd cpu ssd-cufile --prefetch-step 2"; python qa_ssd.py cpu ssd-cufile --prefetch-step 2 2>&1 | grep -aE 'PASS|FAIL|token|native|decode' | cut -c1-200
for cfg in "16 2 4" "16 1 8" "8 2 8"; do
  echo "== smoke_ring $cfg $(date +%H:%M:%S)"; python smoke_ring.py $cfg 2>&1 | grep -aE '^RESULT|^QA|Error|error' | cut -c1-300
done
echo "[$(date +%H:%M:%S)] done"
