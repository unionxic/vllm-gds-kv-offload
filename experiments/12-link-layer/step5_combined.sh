#!/bin/bash
# 5단계: 세 경로(+RDMA)를 같은 시간 창에서 함께 돌릴 때 경로별 처리량·합산·pause 지속시간. KV 정책 없음.
#   각 경로 SEC 초(기본 15): H2D(h2d_sweep --sec, 64 MiB 청크 2 스트림), 로컬 seq gdsio -T, 원격 램디스크 gdsio -T, RDMA→GPU(ib_read_bw -D).
#   gdsio는 초기화 약 2.5 s가 있어 먼저 띄우고, 창은 t0+2 ~ t0+SEC 로 잡아 NIC rx_bytes_phy(원격 유입 합)와 nvidia-fs readMiB(GDS 합)를 차분한다.
#   설정 env: LIO/RIO(io 크기) LW/RW(threads) TXD QP. 조합 COMBOS.
# usage: step5_combined.sh <outdir>
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/step5_combined.jsonl; LOG=$O/step5.log; GDSIO=/usr/local/cuda/gds/tools/gdsio
SEQ=/mnt/local-ssd/seq; REMOTE=/mnt/rain-nvmeof/gdsio; IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}'); SEC=${SEC:-15}
LIO=${LIO:-1M}; RIO=${RIO:-4M}; LW=${LW:-4}; RW=${RW:-8}; TXD=${TXD:-16}; QP=${QP:-4}
SRV=30.0.0.3; SDEV=mlx5_1; CDEV=mlx5_0; PORT=18518; COMBOS=${COMBOS:-"local remote rdma h2d local+remote h2d+local h2d+remote remote+rdma h2d+local+remote h2d+local+remote+rdma"}
pause(){ ethtool -S $IF | grep -E '^ *tx_global_pause_duration:' | awk '{print $2}'; }
rxb(){ ethtool -S $IF | grep -E '^ *rx_bytes_phy:' | awk '{print $2}'; }
nv(){ grep -E '^Reads\s+: n=' /proc/driver/nvidia-fs/stats | grep -oE 'readMiB=[0-9]+' | cut -d= -f2; }
gds(){ $GDSIO -D $1 -d 0 -w $3 -s 2G -i $2 -x 0 -I 0 -T $((SEC + 2)) 2>&1 | grep -oE 'Throughput: [0-9.]+ GiB/sec' | grep -oE '[0-9.]+' | head -1 | awk -v t=$4 '{print t, $1}'; }
h2d(){ python3 $(dirname $0)/h2d_sweep.py --out /dev/null --tag h2d --sizes 64M --streams 2 --sec $SEC --dir h2d 2>/dev/null | grep -oE "'gbps': [0-9.]+" | grep -oE '[0-9.]+' | awk '{print "h2d", $1/1.0737}'; }
rdma(){ ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null; setsid nohup ib_read_bw -d $SDEV -p $PORT -s 1048576 -q $QP -t $TXD -D $SEC -F --report_gbits > /tmp/perftest_srv5.log 2>&1 < /dev/null &"; sleep 1.5
  timeout $((SEC + 30)) ib_read_bw -d $CDEV -p $PORT -s 1048576 -q $QP -t $TXD -D $SEC -F --report_gbits --use_cuda=0 $SRV 2>&1 | grep -E "^\s*1048576\s" | tail -1 | awk '{print "rdma", $4/8/1.0737}'; }
echo "== step5 $(date -Is) SEC $SEC L(io $LIO w $LW) R(io $RIO w $RW) rdma(q $QP txd $TXD)" | tee -a $LOG
for combo in $COMBOS; do
  T=$(mktemp -d)
  case $combo in *local*) gds $SEQ $LIO $LW local > $T/local & ;; esac
  case $combo in *remote*) gds $REMOTE $RIO $RW remote > $T/remote & ;; esac
  sleep 2.5
  case $combo in *rdma*) rdma > $T/rdma & ;; esac
  case $combo in h2d*|*+h2d*) h2d > $T/h2d & ;; esac
  sleep 2; p0=$(pause); b0=$(rxb); n0=$(nv); t0=$(date +%s.%N); sleep $((SEC - 4)); p1=$(pause); b1=$(rxb); n1=$(nv); t1=$(date +%s.%N)
  wait; res=$(cat $T/local $T/remote $T/h2d $T/rdma 2>/dev/null | tr '\n' ';')
  W=$(python3 -c "print(round($t1-$t0,2))"); LINK=$(python3 -c "print(round(($b1-$b0)/($t1-$t0)/2**30,2))"); NVFS=$(python3 -c "print(round(($n1-$n0)/1024/($t1-$t0),2))")
  echo "combo=$combo :: $res :: 창 $W s link_rx $LINK GiB/s gds_rx $NVFS GiB/s pause_dur+$((p1 - p0))" | tee -a $LOG
  python3 - "$combo" "$res" "$W" "$LINK" "$NVFS" "$((p1 - p0))" >> $F <<'PY'
import sys, json
combo, res, w, link, nvfs, pd = sys.argv[1:]
paths = {}
for item in res.split(";"):
    p = item.split()
    if len(p) == 2:
        paths[p[0]] = round(float(p[1]), 3)
print(json.dumps(dict(combo=combo, paths_gib_s=paths, sum_gib_s=round(sum(paths.values()), 3), window_s=float(w), link_rx_gib_s=float(link), gds_rx_gib_s=float(nvfs), pause_dur=int(pd))))
PY
  rm -rf $T; sleep 2
done
ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null"; echo "wrote $F"
