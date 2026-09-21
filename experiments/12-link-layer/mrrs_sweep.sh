#!/bin/bash
# GPU DevCtl MaxReadReq 를 바꿔 가며 h2d_sweep 실행. usage: mrrs_sweep.sh <outdir> [mrrs list, default "128 256 512 1024 2048 4096"]
# DevCtl(CAP_EXP+8) 비트 14:12 = MRRS 코드(0=128 … 5=4096). 끝나면 원래 값으로 되돌린다. MPS 는 건드리지 않는다.
set -u; O=${1:?outdir}; LIST=${2:-"128 256 512 1024 2048 4096"}; mkdir -p "$O"
GPU=$(lspci -D | grep -iE '(vga|3d).*nvidia' | awk '{print $1}' | head -1); [ -n "$GPU" ] || { echo "no gpu"; exit 1; }
CAP=$(sudo setpci -s $GPU CAP_EXP+8.w); echo "gpu $GPU DevCtl=0x$CAP ($(sudo lspci -s $GPU -vvv | grep -oE 'MaxPayload [0-9]+ bytes, MaxReadReq [0-9]+ bytes'))"
ORIG=$CAP
code(){ case $1 in 128) echo 0;; 256) echo 1;; 512) echo 2;; 1024) echo 3;; 2048) echo 4;; 4096) echo 5;; esac; }
setmrrs(){ local v=$((0x$ORIG & ~0x7000 | ($(code $1) << 12))); sudo setpci -s $GPU CAP_EXP+8.w=$(printf %04x $v); }
trap 'sudo setpci -s $GPU CAP_EXP+8.w=$ORIG; echo "restored DevCtl=0x$ORIG"' EXIT
F=$O/mrrs_sweep_$(hostname).jsonl
for m in $LIST; do
  setmrrs $m; now=$(sudo lspci -s $GPU -vvv | grep -oE 'MaxReadReq [0-9]+ bytes')
  echo "== MRRS $m ($now)"; python3 $(dirname $0)/h2d_sweep.py --out $F --tag "mrrs=$m" --sizes 64K,1M,16M,256M --streams 1,2 --sec 3 2>&1 | grep -v Warning
done
echo "wrote $F"
