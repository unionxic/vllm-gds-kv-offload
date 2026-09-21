#!/bin/bash
# 4단계: RDMA 발행 효율(목적지 sunny GPU 메모리). (a) perftest ib_read_bw: 메시지 크기 {64K,256K,1M,4M} x tx-depth {1,4,16,64,128} x QP {1,4}, 각 4 s.
#   (b) Mooncake transfer_engine_bench(read, VRAM): block_size {64K,256K,1M,4M} x batch_size {1,8,32,128} x threads {1,4,8,16}, 각 5 s.
#   pause 지속시간 차분 기록. rain 서버/타깃은 ssh로 기동.
# usage: step4_rdma.sh <outdir>
set -u; O=${1:?outdir}; mkdir -p "$O"; F=$O/step4_rdma.jsonl; LOG=$O/step4.log; IF=$(ip -br addr | awk '/30\.0\.0\./{print $1}')
SRV=30.0.0.3; SDEV=mlx5_1; CDEV=mlx5_0; PORT=18519; SEC=4
BENCH=~/.venvs/gdsllm/lib/python3.10/site-packages/mooncake/transfer_engine_bench; RBENCH=/home/unionxic/miniconda3/envs/gdsllm/lib/python3.10/site-packages/mooncake/transfer_engine_bench
META=http://30.0.0.3:8080/metadata; TGT=30.0.0.3:$((20000 + RANDOM % 20000))  # rpc_meta 중복 키 회피
pause(){ ethtool -S $IF | grep -E '^ *tx_global_pause_duration:' | awk '{print $2}'; }
echo "== step4 $(date -Is)" | tee -a $LOG
[ "${SKIP_PERFTEST:-0}" = 1 ] || for s in 65536 262144 1048576 4194304; do for q in 1 4; do for td in 1 4 16 64 128; do
  ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null; setsid nohup ib_read_bw -d $SDEV -p $PORT -s $s -q $q -t $td -D $SEC -F --report_gbits > /tmp/perftest_srv4.log 2>&1 < /dev/null &"; sleep 1.5
  p0=$(pause); line=$(timeout $((SEC + 30)) ib_read_bw -d $CDEV -p $PORT -s $s -q $q -t $td -D $SEC -F --report_gbits --use_cuda=0 $SRV 2>&1 | grep -E "^\s*$s\s" | tail -1); g=$(echo "$line" | awk '{print $4}')
  echo "perftest s=$s q=$q txd=$td :: ${g:-0} Gb/s pause_dur+$(( $(pause) - p0 ))" | tee -a $LOG
  python3 -c "import json,sys; print(json.dumps(dict(tool='ib_read_bw', size=int(sys.argv[1]), qp=int(sys.argv[2]), txd=int(sys.argv[3]), gbps=float(sys.argv[4] or 0), pause_dur=int(sys.argv[5]))))" $s $q $td "${g:-0}" $(( $(pause) - p0 )) >> $F
done; done; done
ssh rain "pkill -f '^ib_read_bw -d $SDEV -p $PORT' 2>/dev/null"
# (b) transfer engine: rain 타깃(host DRAM 4 GiB), sunny 이니시에이터(VRAM 2 GiB)
# 타깃은 fd를 모두 끊은 bash -c 로 띄워야 ssh 가 돌아온다(conda activate 를 && 로 묶고 & 를 붙이면 서브셸이 타깃을 기다려 ssh 가 멈춤).
ssh rain "pkill -f '^$RBENCH --mode=target' 2>/dev/null; nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate gdsllm && exec $RBENCH --mode=target --metadata_server=$META --local_server_name=$TGT --protocol=rdma --device_name=$SDEV --use_vram=false --buffer_size=$((2 * 2**30))' > /tmp/te_target.log 2>&1 < /dev/null &"; sleep 8; ssh rain "tail -2 /tmp/te_target.log" | tee -a $LOG
for bs in 65536 262144 1048576 4194304; do for b in 1 8 32 128; do for th in 1 4 8 16; do
  p0=$(pause); out=$(timeout 60 $BENCH --mode=initiator --metadata_server=$META --segment_id=$TGT --local_server_name=30.0.0.4:$((20000 + RANDOM % 20000)) --protocol=rdma --device_name=$CDEV --operation=read --use_vram=true --gpu_id=0 --buffer_size=$((2 * 2**30)) --block_size=$bs --batch_size=$b --threads=$th --duration=5 --report_unit=GB 2>&1); g=$(echo "$out" | grep -oiE '[0-9.]+ ?GB/s' | tail -1 | grep -oE '[0-9.]+')
  echo "te block=$bs batch=$b threads=$th :: ${g:-0} GB/s pause_dur+$(( $(pause) - p0 ))" | tee -a $LOG
  python3 -c "import json,sys; print(json.dumps(dict(tool='transfer_engine_bench', block=int(sys.argv[1]), batch=int(sys.argv[2]), threads=int(sys.argv[3]), gb_s=float(sys.argv[4] or 0), pause_dur=int(sys.argv[5]))))" $bs $b $th "${g:-0}" $(( $(pause) - p0 )) >> $F
done; done; done
ssh rain "pkill -f '^$RBENCH --mode=target' 2>/dev/null"; echo "wrote $F"
