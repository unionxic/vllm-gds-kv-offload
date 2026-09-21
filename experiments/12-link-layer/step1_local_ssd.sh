#!/bin/bash
# 1단계: 로컬 SSD 경로가 왜 불안정한가. 같은 파일·같은 범위에서 저장장치 읽기(dd direct)와 GDS 읽기를 비교.
#   파일 두 벌: frag(기존 gdsio가 쓴 조각난 파일, /mnt/local-ssd/gdsio) vs seq(dd로 순차 기록한 파일, /mnt/local-ssd/seq).
#   각 벌에서: dd iflag=direct 8프로세스 병렬(저장장치→CPU), gdsio -x 1(Storage→CPU), -x 0(GDS), -x 2(CPU→GPU), -x 6(GPU_BATCH -B 16).
#   io 1M·threads 8·파일당 2 GiB 고정(완료 시간도 잰다), 각 3회 반복. 원시 파티션 읽기(dd /dev/nvme0n1p3, 읽기 전용)도 1회.
# usage: step1_local_ssd.sh <outdir>
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/step1_local_ssd.jsonl; LOG=$O/step1.log; GDSIO=/usr/local/cuda/gds/tools/gdsio
FRAG=/mnt/local-ssd/gdsio; SEQ=/mnt/local-ssd/seq; N=8; SZ=2G
mkdir -p $SEQ; for i in $(seq 0 $((N-1))); do [ -f $SEQ/gdsio.$i ] || dd if=/dev/urandom of=$SEQ/gdsio.$i bs=1M count=2048 status=none oflag=direct; done
sync; echo "== step1 $(date -Is)" | tee -a $LOG
for d in $FRAG $SEQ; do for i in 0 3 7; do echo "$d/gdsio.$i: $(filefrag $d/gdsio.$i | grep -oE '[0-9]+ extents')" | tee -a $LOG; done; done
nvfs(){ grep -E '^Reads\s+: n=' /proc/driver/nvidia-fs/stats | grep -oE 'readMiB=[0-9]+' | cut -d= -f2; }
rec(){ python3 -c "import json,sys; print(json.dumps(dict(set=sys.argv[1], mode=sys.argv[2], rep=int(sys.argv[3]), gib_s=float(sys.argv[4]), sec=float(sys.argv[5]), nvfs_mib=int(sys.argv[6]))))" "$@" >> $F; }
for rep in 1 2 3; do for d in $FRAG $SEQ; do set=$(basename $d)
  # dd direct 병렬: 각 파일 2 GiB
  n0=$(nvfs); t0=$(date +%s.%N); for i in $(seq 0 $((N-1))); do dd if=$d/gdsio.$i of=/dev/null bs=1M iflag=direct status=none & done; wait; t1=$(date +%s.%N)
  s=$(python3 -c "print(round($N*2/($t1-$t0),3))"); echo "$set dd_direct rep$rep $s GiB/s $(python3 -c "print(round($t1-$t0,2))") s" | tee -a $LOG; rec $set dd_direct $rep $s $(python3 -c "print(round($t1-$t0,3))") $(( $(nvfs) - n0 ))
  for x in 1 0 2 6; do extra=""; [ $x = 6 ] && extra="-B 16"; name=x$x
    n0=$(nvfs); t0=$(date +%s.%N); out=$($GDSIO -D $d -d 0 -w $N -s $SZ -i 1M -x $x $extra -I 0 2>&1); t1=$(date +%s.%N)
    thr=$(echo "$out" | grep -oE 'Throughput: [0-9.]+ GiB/sec' | grep -oE '[0-9.]+' | head -1); [ -n "$thr" ] || thr=0
    echo "$set gdsio_$name rep$rep $thr GiB/s $(python3 -c "print(round($t1-$t0,2))") s nvfs+$(( $(nvfs) - n0 )) MiB" | tee -a $LOG; rec $set gdsio_$name $rep $thr $(python3 -c "print(round($t1-$t0,3))") $(( $(nvfs) - n0 ))
  done
done; done
echo "raw partition:" | tee -a $LOG; sudo dd if=/dev/nvme0n1p3 of=/dev/null bs=1M count=16384 iflag=direct 2>&1 | tail -1 | tee -a $LOG
for i in 0 1 2 3 4 5 6 7; do sudo dd if=/dev/nvme0n1p3 of=/dev/null bs=1M count=2048 skip=$((i*2048)) iflag=direct status=none & done; t0=$(date +%s.%N); wait; echo "raw partition 8 parallel: $(python3 -c "print(round(16/($(date +%s.%N)-$t0),2))") GiB/s" | tee -a $LOG
echo "wrote $F"
