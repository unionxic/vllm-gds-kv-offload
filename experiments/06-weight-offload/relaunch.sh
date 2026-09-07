#!/usr/bin/env bash
# step1 회귀 QA(ring16 s1 t4) PASS 확인 후 campaign.sh 재개
cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/weight-offload/opt66b
while pgrep -f 'repro_wra[p]|qa_ss[d]' >/dev/null; do sleep 10; done
echo "[$(date +%H:%M:%S)] == step1 회귀 QA (wrap fix 후)" >> $O/campaign.log
python smoke_ring.py 16 1 4 > $O/qa-ring16-postfix.log 2>&1; grep -a -E '^RESULT|^QA' $O/qa-ring16-postfix.log >> $O/campaign.log
grep -q '^QA PASS' $O/qa-ring16-postfix.log || { echo "회귀 QA FAIL → 중단" >> $O/campaign.log; exit 1; }
exec ./campaign.sh >> $O/campaign.log 2>&1
