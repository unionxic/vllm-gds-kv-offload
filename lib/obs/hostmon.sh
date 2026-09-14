#!/usr/bin/env bash
# host 지표 1초 샘플. RUN_DIR에 nvidia-smi.csv, gpu_dmon.log, vmstat.log, iostat.log, meminfo.csv, diskstats.csv를 남김.
# STOP_FILE이 생기면 종료. usage: RUN_DIR=... [DEV=nvme0n1] [INTERVAL=1] hostmon.sh
set -u
: "${RUN_DIR:?RUN_DIR}"; INTERVAL="${INTERVAL:-1}"; DEV="${DEV:-nvme0n1}"; STOP="${STOP_FILE:-$RUN_DIR/hostmon.stop}"
mkdir -p "$RUN_DIR"; rm -f "$STOP"
pids=()
cleanup(){ trap - EXIT INT TERM; for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done; for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l "$INTERVAL" >"$RUN_DIR/nvidia-smi.csv" & pids+=($!)
nvidia-smi dmon -s pucvmet -d "$INTERVAL" -o DT >"$RUN_DIR/gpu_dmon.log" 2>&1 & pids+=($!)
vmstat -w -t "$INTERVAL" >"$RUN_DIR/vmstat.log" & pids+=($!)
if command -v iostat >/dev/null; then iostat -t -y -dxm "$INTERVAL" >"$RUN_DIR/iostat.log" & pids+=($!); fi
# BLKIO=1 이면 블록 I/O 한 건마다 한 줄(lib/obs/blkio.bt: ns 시각, dev, rwbs, 섹터, 바이트, 지연 us)을 blkio.log에. sudoers에 /usr/bin/bpftrace NOPASSWD 필요.
# root 프로세스는 직접 못 죽이므로 뒤의 cat을 죽여 SIGPIPE로 끝냄(다음 I/O 줄에서 종료).
if [ "${BLKIO:-0}" = 1 ] && sudo -n -l /usr/bin/bpftrace >/dev/null 2>&1; then
  sudo -n /usr/bin/bpftrace "$(dirname "$0")/blkio.bt" 2>"$RUN_DIR/blkio.err" | cat >"$RUN_DIR/blkio.log" & pids+=($!)
fi
echo "wall_ns,MemAvailable_kB,Dirty_kB,Writeback_kB,Shmem_kB,Mlocked_kB" >"$RUN_DIR/meminfo.csv"
echo "wall_ns,read_MBps,write_MBps,read_lat_ms,write_lat_ms,inflight" >"$RUN_DIR/diskstats.csv"
prev=$(awk -v d="$DEV" '$3==d{print $4,$6,$7,$8,$10,$11,$12}' /proc/diskstats); pt=$(date +%s.%N)
while [ ! -e "$STOP" ]; do
  sleep "$INTERVAL"
  now=$(date +%s%N)
  awk -v now="$now" '/^MemAvailable:/{a=$2} /^Dirty:/{d=$2} /^Writeback:/{w=$2} /^Shmem:/{s=$2} /^Mlocked:/{m=$2} END{printf "%s,%s,%s,%s,%s,%s\n",now,a,d,w,s,m}' /proc/meminfo >>"$RUN_DIR/meminfo.csv"
  cur=$(awk -v d="$DEV" '$3==d{print $4,$6,$7,$8,$10,$11,$12}' /proc/diskstats); ct=$(date +%s.%N)
  echo "$prev $cur $pt $ct $now" | awk '{dt=$16-$15; rd=$8-$1; rs=($9-$2)*512/1048576/dt; rt=$10-$3; wr=$11-$4; ws=($12-$5)*512/1048576/dt; wt=$13-$6;
    printf "%s,%.1f,%.1f,%.2f,%.2f,%d\n",$17,rs,ws,(rd>0?rt/rd:0),(wr>0?wt/wr:0),$14}' >>"$RUN_DIR/diskstats.csv"
  prev=$cur; pt=$ct
done
