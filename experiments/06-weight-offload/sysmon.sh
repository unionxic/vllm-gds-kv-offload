#!/usr/bin/env bash
# 30초마다 메모리/회수/디스크/부하 스냅샷 → opt66b/sysmon.log (campaign 종료 후 자동 종료)
O=$(dirname "$0")/../../results/weight-offload/opt66b
prev=$(awk '$3=="nvme0n1"{print $6, $13}' /proc/diskstats)
while pgrep -f 'campaign\.s[h]|after_campaig[n]|run_66b\.p[y]' >/dev/null; do
  sleep 30
  cur=$(awk '$3=="nvme0n1"{print $6, $13}' /proc/diskstats)
  rd=$(( ( $(echo $cur|cut -d' ' -f1) - $(echo $prev|cut -d' ' -f1) ) * 512 / 30 / 1048576 ))
  io_ms=$(( ( $(echo $cur|cut -d' ' -f2) - $(echo $prev|cut -d' ' -f2) ) / 300 ))
  prev=$cur
  echo "$(date +%H:%M:%S) load=$(cut -d' ' -f1-3 /proc/loadavg) avail=$(awk '/MemAvailable/{printf "%.1fG",$2/1048576}' /proc/meminfo) dirty=$(awk '/^Dirty/{printf "%dM",$2/1024}' /proc/meminfo) pgscan_k=$(awk '$1=="pgscan_kswapd"{print $2}' /proc/vmstat) pgscan_d=$(awk '$1=="pgscan_direct"{print $2}' /proc/vmstat) compact=$(awk '$1=="compact_stall"{print $2}' /proc/vmstat) nvme_rd=${rd}MiB/s nvme_util=${io_ms}% cpu_top=$(ps -eo pcpu,comm --sort=-pcpu | sed -n 2p | tr -s ' ')" >> $O/sysmon.log
done
