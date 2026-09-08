#!/usr/bin/env bash
# 08d: vLLM 읽기 패턴 마이크로벤치(bench_bounce.py). 08c 종료 후. 결과는 campaign08.log의 BENCH 줄.
set -u; cd "$(dirname "$0")"; source ../../env.sh
log(){ echo "[$(date +%H:%M:%S)] $*"; }
while pgrep -f 'campaign08c\.s[h]|run_66b\.p[y]' >/dev/null; do sleep 30; done
log "08c 종료 확인, 08d 시작"
F=$HOME/gds_test/test.bin
for j in default cufile-pb4096.json cufile-pb8192.json; do
  for thr in 4 8; do for chunk in 64 256; do
    env $( [ $j = default ] || echo CUFILE_ENV_PATH_JSON=$PWD/$j ) python bench_bounce.py $F $thr $chunk 2 0 2>&1 | grep BENCH
  done; done
done
python bench_bounce.py $F 4 64 2 1 2>&1 | grep BENCH   # 등록 버퍼(64MiB×4=256MiB: BAR1 한계 근처, 실패 가능)
log "08d 완료"
