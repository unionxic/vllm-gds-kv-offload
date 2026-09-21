#!/bin/bash
# 한 RoCE 링크에서 NVMe-oF 읽기(gdsio, 원격 램디스크 → GPU)와 RDMA 읽기(ib_read_bw, rain → sunny GPU 메모리)를 같이 흘린다.
# 각각 단독, 둘 동시. NVMe-oF io queue 수(NQ) 를 바꿔 반복(재접속: umount → disconnect → connect --nr-io-queues → mount).
# sunny 에서 실행. usage: mixed_link.sh <outdir>   (env NQS="36 8 2" SEC IOSIZE)
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/mixed_link.jsonl; LOG=$O/mixed_link.log
GDSIO=/usr/local/cuda/gds/tools/gdsio; SEC=${SEC:-10}; IOSIZE=${IOSIZE:-1M}; THREADS=${THREADS:-8}; NQS=${NQS:-"36 8 2"}
REMOTE=/mnt/rain-nvmeof/gdsio; NQN=nqn.2026-09.kr.ac.ajou.rain:dram0; MNT=/mnt/rain-nvmeof
SRV=30.0.0.3; SDEV=mlx5_1; CDEV=mlx5_0; PORT=18516
cnt(){ IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}'); ethtool -S $IF | grep -E "^\s*(rx_pause_ctrl_phy|tx_pause_ctrl_phy|rx_out_of_buffer|tx_global_pause_duration|rx_global_pause_duration):" | tr -s " " | tr "\n" " "; }
rcnt(){ ssh rain 'IF=$(ip -br addr | awk "/30\\.0\\.0\\./{print \$1}"); ethtool -S $IF | grep -E "^\s*(rx_pause_ctrl_phy|tx_pause_ctrl_phy|rx_out_of_buffer|tx_global_pause_duration|rx_global_pause_duration):" | tr -s " " | tr "\n" " "'; }
reconnect(){ # $1 = nr io queues
  sudo umount $MNT 2>/dev/null; sudo nvme disconnect -n $NQN > /dev/null 2>&1; sleep 1
  sudo nvme connect -t rdma -a $SRV -s 4420 -n $NQN --nr-io-queues $1 > /dev/null 2>&1 || { echo "connect failed nq=$1"; return 1; }
  sleep 2; sudo mount $MNT || { echo "mount failed"; return 1; }
  echo "nvme-of $NQN reconnected nr-io-queues=$1 ($(for c in /sys/class/nvme/nvme*; do [ "$(cat $c/subsysnqn 2>/dev/null)" = $NQN ] && echo queue_count=$(cat $c/queue_count); done))"
}
mkdir -p $REMOTE; for i in $(seq 0 $((THREADS - 1))); do [ -f $REMOTE/gdsio.$i ] || dd if=/dev/urandom of=$REMOTE/gdsio.$i bs=1M count=2048 status=none; done
gds(){ $GDSIO -D $REMOTE -d 0 -w $THREADS -s 2G -i $IOSIZE -x 0 -I 0 -T $SEC 2>&1 | grep -oE 'Throughput: [0-9.]+ GiB/sec' | awk '{print "nvmeof_gib_s", $2}'; }
rdma(){ ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null; setsid nohup ib_read_bw -d $SDEV -p $PORT -s 1048576 -q 4 -D $SEC -F --report_gbits > /tmp/perftest_srv2.log 2>&1 < /dev/null &"; sleep 1.5
  timeout $((SEC + 30)) ib_read_bw -d $CDEV -p $PORT -s 1048576 -q 4 -D $SEC -F --report_gbits --use_cuda=0 $SRV 2>&1 | grep -E "^\s*1048576\s" | tail -1 | awk '{print "rdma_gbps", $4}'; }
echo "== mixed_link $(date -Is) io $IOSIZE sec $SEC nqs($NQS)" | tee -a $LOG
for nq in $NQS; do
  reconnect $nq | tee -a $LOG || continue
  for combo in nvmeof rdma nvmeof+rdma; do
    T=$(mktemp -d); c0=$(cnt); r0=$(rcnt)
    case $combo in *nvmeof*) gds > $T/a & ;; esac
    case $combo in *rdma*) rdma > $T/b & ;; esac
    wait; c1=$(cnt); r1=$(rcnt); res=$(cat $T/a $T/b 2>/dev/null | tr '\n' ' ')
    echo "nq=$nq combo=$combo :: $res" | tee -a $LOG
    python3 - "$nq" "$combo" "$res" "$c0" "$c1" "$r0" "$r1" >> $F <<'PY'
import sys, json, re
nq, combo, res, c0, c1, r0, r1 = sys.argv[1:]
p = dict((k, float(v)) for k, v in re.findall(r"(\w+) ([0-9.]+)", res))
def d(a, b):
    da = dict(re.findall(r"(\w+): (\d+)", a)); db = dict(re.findall(r"(\w+): (\d+)", b)); return {k: int(db[k]) - int(da[k]) for k in da if k in db and da[k] != db[k]}
tot = p.get("nvmeof_gib_s", 0) * 1.0737 + p.get("rdma_gbps", 0) / 8
print(json.dumps(dict(nq=int(nq), combo=combo, nvmeof_gib_s=p.get("nvmeof_gib_s"), rdma_gbps=p.get("rdma_gbps"), total_gb_s=round(tot, 2), sunny_delta=d(c0, c1), rain_delta=d(r0, r1))))
PY
    rm -rf $T; sleep 2
  done
done
reconnect 36 | tee -a $LOG; ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null"; echo "wrote $F"
