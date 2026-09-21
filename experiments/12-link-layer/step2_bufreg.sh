#!/bin/bash
# 2단계: GPU 버퍼 등록/미등록(-b)과 전송 모드(동기 0 / 비동기 5 / 배치 6)에 따른 경로 차이. 로컬 seq 파일과 원격 램디스크.
#   nvidia-fs 통계(readMiB, Bar1-map, Registered_MiB)와 cuFile 경로를 같이 본다. io 1M threads 8 파일당 2 GiB, 2회 반복.
# usage: step2_bufreg.sh <outdir>
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/step2_bufreg.jsonl; LOG=$O/step2.log; GDSIO=/usr/local/cuda/gds/tools/gdsio
SEQ=/mnt/local-ssd/seq; REMOTE=/mnt/rain-nvmeof/gdsio; N=8
nv(){ grep -E '^Reads\s+: n=' /proc/driver/nvidia-fs/stats | grep -oE 'readMiB=[0-9]+' | cut -d= -f2; }
bar(){ grep -E '^Bar1-map' /proc/driver/nvidia-fs/stats | grep -oE 'n=[0-9]+' | cut -d= -f2; }
echo "== step2 $(date -Is)" | tee -a $LOG
for rep in 1 2; do for d in $SEQ $REMOTE; do set=$(basename $(dirname $d))/$(basename $d)
  for x in 0 5 6; do for reg in reg noreg; do extra=""; [ $x = 6 ] && extra="-B 16"; [ $reg = noreg ] && extra="$extra -b"
    n0=$(nv); b0=$(bar); t0=$(date +%s.%N); out=$($GDSIO -D $d -d 0 -w $N -s 2G -i 1M -x $x $extra -I 0 2>&1); t1=$(date +%s.%N)
    thr=$(echo "$out" | grep -oE 'Throughput: [0-9.]+ GiB/sec' | grep -oE '[0-9.]+' | head -1); [ -n "$thr" ] || thr=0
    echo "$set x$x $reg rep$rep $thr GiB/s $(python3 -c "print(round($t1-$t0,2))") s nvfs+$(( $(nv) - n0 )) MiB bar1map+$(( $(bar) - b0 ))" | tee -a $LOG
    python3 -c "import json,sys; print(json.dumps(dict(set=sys.argv[1], xfer=int(sys.argv[2]), reg=sys.argv[3], rep=int(sys.argv[4]), gib_s=float(sys.argv[5]), sec=float(sys.argv[6]), nvfs_mib=int(sys.argv[7]), bar1_maps=int(sys.argv[8]))))" "$set" $x $reg $rep $thr $(python3 -c "print(round($t1-$t0,3))") $(( $(nv) - n0 )) $(( $(bar) - b0 )) >> $F
  done; done
done; done; echo "wrote $F"
