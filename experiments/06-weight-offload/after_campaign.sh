#!/usr/bin/env bash
# campaign.sh 종료 후 보강 런: phase C r1(수정 전 garbage 런 대체)
cd "$(dirname "$0")"; O=../../results/weight-offload/opt66b
while pgrep -f 'campaign\.s[h]' >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] == 보강: phase C r1 재실행(wrap fix 후)" >> $O/campaign.log
RING=8 STEP=2 THREADS=8 ARMS="cufile" FRACS="0.3" REPS="1" ./run_66b.sh >> $O/campaign.log 2>&1
echo "[$(date +%H:%M:%S)] 보강 완료" >> $O/campaign.log
