#!/bin/bash
# 혼합 링크 재측정(창 정렬 + NIC 바이트 카운터). sunny 에서 실행.
#   두 전송을 같은 순간에 시작해 같은 SEC 초 동안 돌리고, 그 창의 sunny NIC rx_bytes_phy 차분으로 링크 합계를 잰다.
#   조합: nvmeof→GPU 단독, rdma→GPU 단독, 둘 동시(목적 버퍼 둘 다 GPU 0), rdma→host 단독, nvmeof→GPU + rdma→host(수신 경로 분리).
# usage: mixed_link2.sh <outdir>  (env SEC=10 IOSIZE=1M THREADS=8)
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/mixed_link2.jsonl; LOG=$O/mixed_link2.log
GDSIO=/usr/local/cuda/gds/tools/gdsio; SEC=${SEC:-10}; IOSIZE=${IOSIZE:-1M}; THREADS=${THREADS:-8}
REMOTE=/mnt/rain-nvmeof/gdsio; SRV=30.0.0.3; SDEV=mlx5_1; CDEV=mlx5_0; PORT=18517; IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}')
mkdir -p $REMOTE; for i in $(seq 0 $((THREADS - 1))); do [ -f $REMOTE/gdsio.$i ] || dd if=/dev/urandom of=$REMOTE/gdsio.$i bs=1M count=2048 status=none; done
rxb(){ ethtool -S $IF | grep -E '^ *rx_bytes_phy:' | awk '{print $2}'; }
pause(){ ethtool -S $IF | grep -E '^ *tx_pause_ctrl_phy:' | awk '{print $2}'; }
gds(){ $GDSIO -D $REMOTE -d 0 -w $THREADS -s 2G -i $IOSIZE -x 0 -I 0 -T $SEC 2>&1 | grep -oE 'Throughput: [0-9.]+ GiB/sec' | awk '{print "nvmeof_gib_s", $2}'; }
rdma(){ # $1 = cuda|host
  local extra=""; [ "$1" = cuda ] && extra="--use_cuda=0"
  ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null; setsid nohup ib_read_bw -d $SDEV -p $PORT -s 1048576 -q 4 -D $SEC -F --report_gbits > /tmp/perftest_srv3.log 2>&1 < /dev/null &"; sleep 1.5
  timeout $((SEC + 30)) ib_read_bw -d $CDEV -p $PORT -s 1048576 -q 4 -D $SEC -F --report_gbits $extra $SRV 2>&1 | grep -E "^\s*1048576\s" | tail -1 | awk -v m=$1 '{print "rdma_" m "_gbps", $4}'; }
echo "== mixed_link2 $(date -Is) sec $SEC io $IOSIZE" | tee -a $LOG
for combo in nvmeof rdma_cuda nvmeof+rdma_cuda rdma_host nvmeof+rdma_host; do
  T=$(mktemp -d)
  # rdma 는 서버 기동에 1.5 s 걸리므로 먼저 띄우고, nvmeof 는 1.5 s 뒤 시작해 창을 맞춘다.
  case $combo in *rdma_cuda*) rdma cuda > $T/b & ;; *rdma_host*) rdma host > $T/b & ;; esac
  case $combo in *nvmeof*) (sleep 1.5; gds > $T/a) & ;; esac
  sleep 2.0; b0=$(rxb); p0=$(pause); t0=$(date +%s.%N); sleep $((SEC - 3)); b1=$(rxb); p1=$(pause); t1=$(date +%s.%N)
  wait; res=$(cat $T/a $T/b 2>/dev/null | tr '\n' ' ')
  link=$(python3 -c "print(round(($b1-$b0)/($t1-$t0)/1e9,2))")
  echo "combo=$combo :: $res :: link_rx_GB_s=$link pause=$((p1 - p0)) (창 $(python3 -c "print(round($t1-$t0,1))") s)" | tee -a $LOG
  python3 -c "
import json,re,sys; res=sys.argv[1]; p=dict((k,float(v)) for k,v in re.findall(r'(\w+) ([0-9.]+)', res))
print(json.dumps(dict(combo=sys.argv[2], **p, link_rx_gb_s=float(sys.argv[3]), pause_frames=int(sys.argv[4]))))" "$res" "$combo" "$link" "$((p1 - p0))" >> $F
  rm -rf $T; sleep 2
done
ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null"; echo "wrote $F"
