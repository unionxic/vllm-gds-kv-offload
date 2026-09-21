#!/bin/bash
# RoCE 링크 perftest 매트릭스. sunny(클라이언트, 30.0.0.4, mlx5_0)에서 실행, rain(서버, 30.0.0.3, mlx5_1)은 ssh 로 띄운다.
#   ib_read_bw / ib_write_bw: 메시지 크기 SIZES × QP 수 QPS × MTU MTUS, 각 -D SEC 초. 클라이언트 메모리는 host 와 GPU(--use_cuda=0) 둘 다.
#   ib_write_lat: 메시지 크기별 지연.
# 실행 전후 양쪽 NIC 의 pause/out_of_buffer/CNP/ack timeout 카운터 차분을 남긴다.
# usage: roce_sweep.sh <outdir>   (env SIZES QPS MTUS SEC)
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/roce_sweep.jsonl; LOG=$O/roce_sweep.log
SIZES=${SIZES:-"4096 65536 1048576 4194304"}; QPS=${QPS:-"1 4 16"}; MTUS=${MTUS:-"1024 4096"}; SEC=${SEC:-4}
SRV=30.0.0.3; SDEV=mlx5_1; CDEV=mlx5_0; PORT=18515
cnt(){ # 호스트(local|rain) 카운터 스냅샷 한 줄
  local cmd='IF=$(ip -br addr | awk "/30\\.0\\.0\\./{print \$1}"); ethtool -S $IF | grep -E "^\s*(rx_pause_ctrl_phy|tx_pause_ctrl_phy|rx_out_of_buffer|rx_global_pause_duration|tx_global_pause_duration|rx_discards_phy|tx_discards_phy):" | tr -s " " | tr "\n" " "; for h in /sys/class/infiniband/mlx5_*/ports/1/hw_counters; do for f in np_cnp_sent rp_cnp_handled np_ecn_marked_roce_packets local_ack_timeout_err packet_seq_err out_of_buffer; do printf "%s=%s " $f $(cat $h/$f 2>/dev/null); done; done'
  if [ "$1" = rain ]; then ssh rain "$cmd"; else bash -c "$cmd"; fi
}
run(){ # $1 tool, $2 size, $3 qp, $4 mtu, $5 mem(host|cuda)
  local tool=$1 s=$2 q=$3 m=$4 mem=$5 extra=""
  [ "$mem" = cuda ] && extra="--use_cuda=0"
  ssh rain "pkill -f '^ib_(read|write)_(bw|lat) ' 2>/dev/null; setsid nohup $tool -d $SDEV -p $PORT -s $s -q $q -m $m -D $SEC -F --report_gbits > /tmp/perftest_srv.log 2>&1 < /dev/null &"
  sleep 1.5
  local b0=$(cnt local) r0=$(cnt rain)
  local out; out=$(timeout $((SEC + 30)) $tool -d $CDEV -p $PORT -s $s -q $q -m $m -D $SEC -F --report_gbits $extra $SRV 2>&1)
  local b1=$(cnt local) r1=$(cnt rain)
  echo "$out" >> $LOG
  # 결과 줄: "#bytes #iterations BW peak[Gb/sec] BW average[Gb/sec] MsgRate[Mpps]" 다음 줄이 수치
  local line=$(echo "$out" | grep -E "^\s*$s\s" | tail -1)
  python3 - "$tool" "$s" "$q" "$m" "$mem" "$line" "$b0" "$b1" "$r0" "$r1" >> $F <<'PY'
import sys, json, re
tool, s, q, m, mem, line, b0, b1, r0, r1 = sys.argv[1:]
nums = re.findall(r"[-+]?\d+\.?\d*", line)
def delta(a, b):
    da = dict(re.findall(r"(\w+)[=:]\s*(\d+)", a)); db = dict(re.findall(r"(\w+)[=:]\s*(\d+)", b))
    return {k: int(db[k]) - int(da[k]) for k in da if k in db and int(db[k]) != int(da[k])}
row = dict(tool=tool, size=int(s), qp=int(q), mtu=int(m), mem=mem, raw=line.strip(),
           gbps_avg=float(nums[3]) if len(nums) > 3 else None, mpps=float(nums[4]) if len(nums) > 4 else None,
           sunny_delta=delta(b0, b1), rain_delta=delta(r0, r1))
print(json.dumps(row))
PY
  echo "$tool s=$s q=$q mtu=$m mem=$mem -> $line" | tee -a $LOG
}
echo "== roce_sweep $(date -Is) sizes($SIZES) qps($QPS) mtus($MTUS) sec $SEC" | tee -a $LOG
for tool in ib_read_bw ib_write_bw; do for mem in host cuda; do for m in $MTUS; do for q in $QPS; do for s in $SIZES; do
  run $tool $s $q $m $mem
done; done; done; done; done
# 지연은 QP 1, MTU 4096, host 메모리만
for s in 64 4096 65536 1048576; do
  ssh rain "pkill -f '^ib_write_lat ' 2>/dev/null; setsid nohup ib_write_lat -d $SDEV -p $PORT -s $s -n 2000 -F > /tmp/perftest_srv.log 2>&1 < /dev/null &"; sleep 1.5
  out=$(timeout 60 ib_write_lat -d $CDEV -p $PORT -s $s -n 2000 -F $SRV 2>&1); echo "$out" >> $LOG
  line=$(echo "$out" | grep -E "^\s*$s\s" | tail -1); echo "ib_write_lat s=$s -> $line" | tee -a $LOG
  python3 -c "import json,re,sys; l=sys.argv[1]; n=re.findall(r'[-+]?\d+\.?\d*', l); print(json.dumps(dict(tool='ib_write_lat', size=int(sys.argv[2]), raw=l.strip(), lat_us_typical=float(n[4]) if len(n)>4 else None, lat_us_avg=float(n[5]) if len(n)>5 else None, lat_us_99=float(n[7]) if len(n)>7 else None)))" "$line" "$s" >> $F
done
ssh rain "pkill -f '^ib_(read|write)_(bw|lat) ' 2>/dev/null"; echo "wrote $F"
