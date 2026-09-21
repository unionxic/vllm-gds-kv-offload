#!/bin/bash
# 5단계: 세 경로(+RDMA)를 고정 바이트로 함께 돌릴 때 경로별 처리량·완료 시간·합산·pause 지속시간. KV 정책 없음.
#   각 경로 16 GiB: H2D(h2d_sweep --bytes, 64 MiB 청크 2 스트림), 로컬 seq gdsio(8 x 2 GiB), 원격 램디스크 gdsio(8 x 2 GiB),
#   RDMA→GPU(ib_read_bw -n 반복, 1 MiB x 16384, rain 서버). 설정은 env로(3·4단계에서 찾은 값): LIO/RIO(io 크기), LW/RW(threads), LX/RX(xfer), LB/RB(batch), TXD(tx-depth), QP.
#   조합: local, remote, rdma, h2d, local+remote, h2d+local, h2d+remote, remote+rdma, h2d+local+remote, h2d+local+remote+rdma.
# usage: step5_combined.sh <outdir>
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/step5_combined.jsonl; LOG=$O/step5.log; GDSIO=/usr/local/cuda/gds/tools/gdsio
SEQ=/mnt/local-ssd/seq; REMOTE=/mnt/rain-nvmeof/gdsio; IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}')
LIO=${LIO:-1M}; RIO=${RIO:-1M}; LW=${LW:-8}; RW=${RW:-8}; LX=${LX:-0}; RX=${RX:-0}; LB=${LB:-}; RB=${RB:-}; TXD=${TXD:-16}; QP=${QP:-4}
SRV=30.0.0.3; SDEV=mlx5_1; CDEV=mlx5_0; PORT=18518; COMBOS=${COMBOS:-"local remote rdma h2d local+remote h2d+local h2d+remote remote+rdma h2d+local+remote h2d+local+remote+rdma"}
pause(){ ethtool -S $IF | grep -E '^ *tx_global_pause_duration:' | awk '{print $2}'; }
nv(){ grep -E '^Reads\s+: n=' /proc/driver/nvidia-fs/stats | grep -oE 'readMiB=[0-9]+' | cut -d= -f2; }
gds(){ # $1 dir $2 io $3 threads $4 xfer $5 batch $6 tag
  local t0=$(date +%s.%N) out; out=$($GDSIO -D $1 -d 0 -w $3 -s 2G -i $2 -x $4 ${5:+-B $5} -I 0 2>&1); local t1=$(date +%s.%N)
  echo "$6 $(echo "$out" | grep -oE 'Throughput: [0-9.]+ GiB/sec' | grep -oE '[0-9.]+' | head -1) $(python3 -c "print(round($t1-$t0,3))")"; }
h2d(){ local t0=$(date +%s.%N); local g=$(python3 $(dirname $0)/h2d_sweep.py --out /dev/null --tag h2d --sizes 64M --streams 2 --bytes $((16 * 2**30)) --dir h2d 2>/dev/null | grep -oE "'gbps': [0-9.]+" | grep -oE '[0-9.]+'); echo "h2d $(python3 -c "print(round($g/1.0737,3))") $(python3 -c "print(round($(date +%s.%N)-$t0,3))")"; }
rdma(){ ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null; setsid nohup ib_read_bw -d $SDEV -p $PORT -s 1048576 -q $QP -t $TXD -n $((16384 / QP)) -F --report_gbits > /tmp/perftest_srv5.log 2>&1 < /dev/null &"; sleep 1.5
  local t0=$(date +%s.%N); local g=$(timeout 300 ib_read_bw -d $CDEV -p $PORT -s 1048576 -q $QP -t $TXD -n $((16384 / QP)) -F --report_gbits --use_cuda=0 $SRV 2>&1 | grep -E "^\s*1048576\s" | tail -1 | awk '{print $4}'); echo "rdma $(python3 -c "print(round(${g:-0}/8/1.0737,3))") $(python3 -c "print(round($(date +%s.%N)-$t0-1.5,3))")"; }
echo "== step5 $(date -Is) L(io $LIO w $LW x $LX b '$LB') R(io $RIO w $RW x $RX b '$RB') rdma(q $QP txd $TXD)" | tee -a $LOG
for combo in $COMBOS; do
  T=$(mktemp -d); p0=$(pause); n0=$(nv); t0=$(date +%s.%N)
  case $combo in *rdma*) rdma > $T/rdma & ;; esac
  case $combo in *local*) gds $SEQ $LIO $LW $LX "$LB" local > $T/local & ;; esac
  case $combo in *remote*) gds $REMOTE $RIO $RW $RX "$RB" remote > $T/remote & ;; esac
  case $combo in h2d*|*+h2d*) h2d > $T/h2d & ;; esac
  wait; t1=$(date +%s.%N); res=$(cat $T/local $T/remote $T/h2d $T/rdma 2>/dev/null | tr '\n' ';')
  echo "combo=$combo :: $res :: wall $(python3 -c "print(round($t1-$t0,2))") s pause_dur+$(( $(pause) - p0 )) nvfs+$(( $(nv) - n0 )) MiB" | tee -a $LOG
  python3 - "$combo" "$res" "$(python3 -c "print(round($t1-$t0,3))")" "$(( $(pause) - p0 ))" "$(( $(nv) - n0 ))" >> $F <<'PY'
import sys, json
combo, res, wall, pd, nv = sys.argv[1:]
paths = {}
for item in res.split(";"):
    p = item.split()
    if len(p) == 3:
        paths[p[0]] = dict(gib_s=float(p[1]), sec=float(p[2]))
tot = sum(v["gib_s"] for v in paths.values())
print(json.dumps(dict(combo=combo, paths=paths, sum_gib_s=round(tot, 3), wall_s=float(wall), pause_dur=int(pd), nvfs_mib=int(nv))))
PY
  rm -rf $T; sleep 2
done
ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null"; echo "wrote $F"
