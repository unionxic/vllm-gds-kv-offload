#!/usr/bin/env bash
# 08e: 1MiB 대조군 지연의 원인 검증. 가설 = cuFile IO 스레드가 하이퍼스레드 형제 코어에 몰려 직렬화.
#   근거(09-09 00:3x, bench_bounce 4스레드 64MiB): 물리코어 고정(0-17) 4/4가 3.2GiB/s,
#   HT쌍(0,18,1,19) 4중 3이 1.1~2.1GiB/s, 미고정 4중 1이 2.1GiB/s. GPU 클럭/shadow버퍼/cuFile 경로는 두 모드가 동일.
#   A: 물리코어 고정 런, B: 미고정 재확인 런. 각 런의 forward 구간에서 ssd 스레드가 올라간 CPU를 샘플링.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
O=../../results/cufile-bounce; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
sample_cpus(){ # pid logfile: ssd 스레드가 최근 실행된 CPU 번호 분포
  local pid=$1 out=$2
  while kill -0 $pid 2>/dev/null; do
    for t in /proc/$pid/task/*; do
      n=$(cat $t/comm 2>/dev/null); case "$n" in vllm-ssd*) echo "$n cpu$(awk '{print $39}' $t/stat 2>/dev/null)";; esac
    done
    sleep 2
  done > $out
}
run(){ # tag, [taskset args...]
  local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b
  log "== $tag ($*)"
  "$@" python ../06-weight-offload/run_66b.py --transport cufile --host-fraction 0.3 \
    --prefetch-step 1 --io-threads 4 --ring-mb 0 --gpu-util 0.9 --tag $tag --out $O/$tag.json > $O/$tag.log 2>&1 &
  local p=$!
  sample_cpus $p $O/$tag.cpus &
  wait $p
  grep -aq '^RESULT' $O/$tag.log || { log "FAILED $tag"; grep -aE 'Error|Traceback' $O/$tag.log | tail -3; return 1; }
  grep -a '^RESULT' $O/$tag.log | cut -c1-320
  log "  ssd 스레드 CPU 분포: $(sort $O/$tag.cpus | uniq -c | sort -rn | head -6 | tr '\n' ' ')"
}
run c-h0.3-pb1024-phys-r1 taskset -c 0-17
run c-h0.3-pb1024-free-r1 env
log "08e 완료"
