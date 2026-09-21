#!/bin/bash
# 3단계: 전송 요청 크기·동시성 분리 sweep(GDS, 목적지 GPU). 로컬 seq와 원격 램디스크.
#   (a) io 크기 {64K,256K,1M,4M,16M} @ threads 8   (b) threads {1,2,4,8,16} @ io 1M
#   (c) 배치 깊이 -x 6 -B {4,16,64} @ io 1M threads 4   (d) NVMe-oF io 큐 수 {2,4,8,16,36} @ io 1M threads 8 (원격만, 재접속)
#   파일당 2 GiB 고정(완료 시간 기록). pause 프레임 차분 기록.
# usage: step3_concurrency.sh <outdir>
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/step3_concurrency.jsonl; LOG=$O/step3.log; GDSIO=/usr/local/cuda/gds/tools/gdsio
SEQ=/mnt/local-ssd/seq; REMOTE=/mnt/rain-nvmeof/gdsio; NQN=nqn.2026-09.kr.ac.ajou.rain:dram0; MNT=/mnt/rain-nvmeof; IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}')
pause(){ ethtool -S $IF | grep -E '^ *tx_global_pause_duration:' | awk '{print $2}'; }
run(){ # $1 set-dir $2 tag $3 threads $4 io $5 extra
  local d=$1 tag=$2 w=$3 io=$4 extra=$5 set=$(basename $(dirname $1))/$(basename $1)
  local p0=$(pause) t0=$(date +%s.%N) out; out=$($GDSIO -D $d -d 0 -w $w -s 2G -i $io -x ${XFER:-0} $extra -I 0 2>&1); local t1=$(date +%s.%N)
  local thr=$(echo "$out" | grep -oE 'Throughput: [0-9.]+ GiB/sec' | grep -oE '[0-9.]+' | head -1); [ -n "$thr" ] || thr=0
  echo "$set $tag threads=$w io=$io $extra :: $thr GiB/s $(python3 -c "print(round($t1-$t0,2))") s pause_dur+$(( $(pause) - p0 ))" | tee -a $LOG
  python3 -c "import json,sys; print(json.dumps(dict(set=sys.argv[1], sweep=sys.argv[2], threads=int(sys.argv[3]), io=sys.argv[4], extra=sys.argv[5], gib_s=float(sys.argv[6]), sec=float(sys.argv[7]), pause_dur=int(sys.argv[8]), nq=int(sys.argv[9]))))" "$set" "$tag" $w $io "$extra" $thr $(python3 -c "print(round($t1-$t0,3))") $(( $(pause) - p0 )) ${NQ:-36} >> $F
}
reconnect(){ sudo umount $MNT 2>/dev/null; sudo nvme disconnect -n $NQN >/dev/null 2>&1; sleep 1; sudo nvme connect -t rdma -a 30.0.0.3 -s 4420 -n $NQN --nr-io-queues $1 >/dev/null 2>&1 || return 1; sleep 2; sudo mount $MNT || return 1; echo "nvme-of reconnected nr-io-queues=$1" | tee -a $LOG; }
echo "== step3 $(date -Is)" | tee -a $LOG
for d in $SEQ $REMOTE; do
  for io in 64K 256K 1M 4M 16M; do XFER=0 run $d iosize 8 $io ""; done
  for w in 1 2 4 8 16; do XFER=0 run $d threads $w 1M ""; done
  for B in 4 16 64; do XFER=6 run $d batch 4 1M "-B $B"; done
done
for nq in 2 4 8 16 36; do reconnect $nq || continue; NQ=$nq XFER=0 run $REMOTE nvmeof_queues 8 1M ""; NQ=$nq XFER=0 run $REMOTE nvmeof_queues 4 4M ""; done
reconnect 36; echo "wrote $F"
