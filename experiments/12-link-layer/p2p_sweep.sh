#!/bin/bash
# GPU 유입 경로 조합을 IO 크기별로. sunny 에서 실행. gdsio(GDS 직접, -x 0) 로컬 NVMe(/mnt/local-ssd) / 원격 램디스크(/mnt/rain-nvmeof),
# H2D 는 h2d_sweep.py(64 MiB, 2 스트림). 조합: local, remote, local+remote, h2d, h2d+local, h2d+remote, h2d+local+remote.
# 같이 기록: nvidia-smi PCIe rx/tx 처리량 샘플(1 s), NIC 카운터 차분. usage: p2p_sweep.sh <outdir>  (env IOSIZES SEC THREADS)
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/p2p_sweep.jsonl; LOG=$O/p2p_sweep.log
GDSIO=/usr/local/cuda/gds/tools/gdsio; IOSIZES=${IOSIZES:-"64K 1M 4M"}; SEC=${SEC:-10}; THREADS=${THREADS:-8}; FILE_GB=${FILE_GB:-2}
LOCAL=/mnt/local-ssd/gdsio; REMOTE=/mnt/rain-nvmeof/gdsio
for d in $LOCAL $REMOTE; do mkdir -p $d; for i in $(seq 0 $((THREADS - 1))); do [ -f $d/gdsio.$i ] || $GDSIO -D $d -d 0 -w 1 -s ${FILE_GB}G -i 1M -x 0 -I 1 -T 1 -f $d/gdsio.$i > /dev/null 2>&1 || dd if=/dev/urandom of=$d/gdsio.$i bs=1M count=$((FILE_GB * 1024)) status=none; done; done
cnt(){ IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}'); ethtool -S $IF | grep -E "^\s*(rx_pause_ctrl_phy|tx_pause_ctrl_phy|rx_out_of_buffer|tx_global_pause_duration):" | tr -s " " | tr "\n" " "; }
gds(){ # $1 dir $2 iosize $3 tag → GiB/s 를 파일에
  $GDSIO -D $1 -d 0 -w $THREADS -s ${FILE_GB}G -i $2 -x 0 -I 0 -T $SEC 2>&1 | grep -oE 'Throughput: [0-9.]+ GiB/sec' | awk -v t=$3 '{print t, $2}'
}
h2d(){ python3 $(dirname $0)/h2d_sweep.py --out /dev/null --tag h2d --sizes 64M --streams 2 --sec $SEC --dir h2d 2>/dev/null | grep -oE "'gbps': [0-9.]+" | awk '{print "h2d", $2}'; }
pcie_mon(){ # 1 s 샘플: rx/tx KB/s → 평균 MB/s
  local n=0 rx=0 tx=0; for _ in $(seq 1 $1); do read a b < <(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits | tr -d ','); t=$(nvidia-smi -q -d PCIE | grep -E 'Rx Throughput|Tx Throughput' | grep -oE '[0-9]+' | tr '\n' ' '); set -- $t; rx=$((rx + ${1:-0})); tx=$((tx + ${2:-0})); n=$((n + 1)); sleep 1; done; echo "pcie_rx_MBps=$((rx / n / 1024)) pcie_tx_MBps=$((tx / n / 1024)) gen=$a width=$b"
}
echo "== p2p_sweep $(date -Is) io($IOSIZES) sec $SEC threads $THREADS" | tee -a $LOG
for io in $IOSIZES; do for combo in local remote local+remote h2d h2d+local h2d+remote h2d+local+remote; do
  T=$(mktemp -d); c0=$(cnt); pcie_mon $((SEC - 2)) > $T/pcie &
  case $combo in *local*) gds $LOCAL $io local > $T/local & ;; esac
  case $combo in *remote*) gds $REMOTE $io remote > $T/remote & ;; esac
  case $combo in h2d*) h2d > $T/h2d & ;; esac
  wait; c1=$(cnt)
  res=$(cat $T/local $T/remote $T/h2d 2>/dev/null | tr '\n' ' '); pm=$(cat $T/pcie)
  echo "io=$io combo=$combo :: $res :: $pm" | tee -a $LOG
  python3 - "$io" "$combo" "$res" "$pm" "$c0" "$c1" >> $F <<'PY'
import sys, json, re
io, combo, res, pm, c0, c1 = sys.argv[1:]
parts = dict((k, float(v)) for k, v in re.findall(r"(\w+) ([0-9.]+)", res))
gib = {k: v for k, v in parts.items() if k != "h2d"}; h = parts.get("h2d")
d0 = dict(re.findall(r"(\w+): (\d+)", c0)); d1 = dict(re.findall(r"(\w+): (\d+)", c1))
print(json.dumps(dict(io=io, combo=combo, gds_gib_s=gib, h2d_gb_s=h, total_gb_s=round(sum(v * 1.0737 for v in gib.values()) + (h or 0), 2),
                      pcie=dict(re.findall(r"(\w+)=(\S+)", pm)), nic_delta={k: int(d1[k]) - int(d0[k]) for k in d0 if k in d1 and d1[k] != d0[k]})))
PY
  rm -rf $T; sleep 2
done; done; echo "wrote $F"
