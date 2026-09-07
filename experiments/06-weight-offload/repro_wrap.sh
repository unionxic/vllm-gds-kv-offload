#!/usr/bin/env bash
# 경계 슬롯 충돌 재현/검증: opt-2.7b group32/num31 → 오프로드 모듈 31개(홀수) × step2. 인자: 라벨
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
while pgrep -f 'isolate_step[2]' >/dev/null; do sleep 10; done
echo "[$(date +%H:%M:%S)] == $1: cpu ssd-cufile group32/num31 step2"
python qa_ssd.py cpu ssd-cufile --group-size 32 --num-in-group 31 --prefetch-step 2 2>&1 | grep -aE 'PASS|FAIL|token|native|decode' | cut -c1-200
echo "[$(date +%H:%M:%S)] done"
