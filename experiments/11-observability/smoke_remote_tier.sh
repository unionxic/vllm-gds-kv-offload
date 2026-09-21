#!/usr/bin/env bash
# 원격 티어 KV 오프로드 smoke. 조건 4개를 순서대로 한 번씩 돌려 result.json을 남긴다.
#   none        재계산(KV 오프로드 끔). 출력 토큰열 기준.
#   local-ssd   cufile, KV 디렉터리 = 로컬 SSD(results/smoke-remote-tier/kv-local)
#   remote-dram cufile, KV 디렉터리 = /mnt/sunny-nvmeof/kv-smoke (원격 호스트 DRAM ramdisk, NVMe-oF RDMA + GDS)
#   remote-ssd  cufile, KV 디렉터리 = /mnt/sunny-ssd/kv-smoke (원격 NVMe 파티션, NVMe-oF RDMA + GDS)
# 워크로드는 campaign_grid3b.sh의 3B 조건을 따르되 입력은 campaign_qwen72.sh 기본인 longbench,
# 가중치는 GPU 상주(--no-weight-offload)로 고정해 KV 경로만 조건 간 차이가 되게 한다.
# 조건별로 result.json이 있으면 건너뜀. --force면 해당 조건 결과 폴더를 지우고 다시 돌림.
# /mnt/sunny-*/gdsio 는 건드리지 않는다. 지우는 것은 이 스크립트가 만든 kv-smoke 디렉터리뿐.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
O=$(mkdir -p ../../results/smoke-remote-tier && cd ../../results/smoke-remote-tier && pwd)
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a $O/campaign.log; }

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
NDOCS=${NDOCS:-8}; MAXLEN=${MAXLEN:-4096}; DECODE=${DECODE:-8}; KVB=${KVB:-2}
NVMEOF_ROOT=${NVMEOF_ROOT:-/mnt/sunny-nvmeof/kv-smoke}
RSSD_ROOT=${RSSD_ROOT:-/mnt/sunny-ssd/kv-smoke}
CONDS=${CONDS:-"none local-ssd remote-dram remote-ssd"}

for m in /mnt/sunny-nvmeof /mnt/sunny-ssd; do
  mountpoint -q $m || { log "ABORT: $m 이 마운트되어 있지 않음"; exit 2; }
done

log "== smoke-remote-tier: $MODEL, longbench ${NDOCS}건, max_model_len $MAXLEN, KV 예산 ${KVB}요청, 조건($CONDS)"
for cond in $CONDS; do
  D=$O/$cond
  if [ -f $D/result.json ]; then
    [ $FORCE = 1 ] && rm -rf $D || { log "skip $cond (result.json 있음)"; continue; }
  fi
  rm -rf $D
  case $cond in
    none)        kvt=none;   KVDIR=$O/kv-local;;
    local-ssd)   kvt=cufile; KVDIR=$O/kv-local;;
    remote-dram) kvt=cufile; KVDIR=$NVMEOF_ROOT;;
    remote-ssd)  kvt=cufile; KVDIR=$RSSD_ROOT;;
    *) log "unknown cond $cond"; continue;;
  esac
  # KV 디렉터리는 마운트 지점 자체나 그 바로 아래 gdsio 같은 공용 폴더가 아니어야 함(rm -rf 보호).
  case "$KVDIR" in */kv-smoke|*/kv-local) ;; *) log "ABORT: KV 디렉터리 이름이 kv-smoke/kv-local이 아님: $KVDIR"; exit 3;; esac
  mountpoint -q "$KVDIR" 2>/dev/null && { log "ABORT: KV 디렉터리가 마운트 지점임: $KVDIR"; exit 3; }
  rm -rf "$KVDIR"; mkdir -p "$KVDIR"
  log "start $cond (kv-transport=$kvt, kv-root=$KVDIR)"
  python run_obs.py --run-dir $D --model $MODEL --prompt-source longbench --n-docs $NDOCS \
    --decode-tokens $DECODE --max-model-len $MAXLEN --kv-batch $KVB --kv-block 64 \
    --no-weight-offload --settle-sec 3 --final-settle-sec 3 --poll-sleep-ms 0 \
    --ssd-root $O/ssd --kv-root "$KVDIR" --kv-transport $kvt > $O/$cond.log 2>&1
  rc=$?
  if [ -f $D/result.json ]; then log "done  $cond rc=$rc"; else
    log "FAILED $cond rc=$rc"; grep -aE 'Traceback|Error' $O/$cond.log | tail -2 | tee -a $O/campaign.log >/dev/null; fi
  rm -rf "$KVDIR"
done
rm -rf $O/ssd $O/kv-local
log "== smoke-remote-tier 종료"
