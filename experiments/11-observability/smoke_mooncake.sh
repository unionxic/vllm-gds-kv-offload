#!/usr/bin/env bash
# Mooncake 기준선 smoke. 3B 모델로 조건 세 개를 한 번씩 돌려 result.json을 남긴다.
#   none        재계산(KV 오프로드 끔). 출력 토큰열 기준.
#   mooncake    포크의 MooncakeStoreConnector. KV 풀은 원격 호스트 DRAM(Mooncake Store),
#               전송은 Mooncake Transfer Engine RDMA. 원격 서비스는 sunny의 ~/bin/mooncake_up.sh.
#   ours-ratio  포크 HybridSpec, 배치 ratio(host 몫 HOSTSHARE) + 파일 티어 루트 2개
#               (로컬 NVMe GDS W_LOCAL : 원격 호스트 DRAM NVMe-oF RDMA + GDS W_REMOTE).
# 조건별로 result.json이 있으면 건너뜀. --force면 해당 조건 결과 폴더를 지우고 다시 돌림.
# /mnt/sunny-*/gdsio 는 건드리지 않는다. 지우는 것은 이 스크립트가 만든 kv-mc 디렉터리뿐.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
OUT=${OUT:-../../results/smoke-mooncake}
O=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a $O/campaign.log; }

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
NDOCS=${NDOCS:-8}; MAXLEN=${MAXLEN:-4096}; DECODE=${DECODE:-8}; KVB=${KVB:-2}
HOSTGB=${HOSTGB:-8}; HOSTSHARE=${HOSTSHARE:-0.47}
W_LOCAL=${W_LOCAL:-13}; W_REMOTE=${W_REMOTE:-40}
MC_MASTER=${MC_MASTER:-30.0.0.4:50051}; MC_META=${MC_META:-http://30.0.0.4:8080/metadata}
MC_DEV=${MC_DEV:-mlx5_1}; MC_IP=${MC_IP:-30.0.0.3}; MC_STAGING_GB=${MC_STAGING_GB:-8}
LOCAL_ROOT=$O/kv-mc-local
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/sunny-nvmeof/kv-mc}
CONDS=${CONDS:-"none mooncake ours-ratio"}

mountpoint -q /mnt/sunny-nvmeof || { log "ABORT: /mnt/sunny-nvmeof 이 마운트되어 있지 않음"; exit 2; }

# 이 스크립트가 만든 KV 디렉터리만 지운다(이름이 kv-mc*가 아니거나 마운트 지점이면 중단).
clean_root(){
  case "$1" in */kv-mc|*/kv-mc-local) ;; *) log "ABORT: KV 디렉터리 이름이 kv-mc*가 아님: $1"; exit 3;; esac
  mountpoint -q "$1" 2>/dev/null && { log "ABORT: KV 디렉터리가 마운트 지점임: $1"; exit 3; }
  rm -rf "$1"
}

if echo "$CONDS" | grep -qw mooncake; then
  curl -s --max-time 5 "http://${MC_MASTER%%:*}:9003/metrics" | grep -q master_total_capacity_bytes \
    || { log "ABORT: Mooncake master 메트릭에 닿지 않음(sunny에서 ~/bin/mooncake_up.sh 실행 필요)"; exit 5; }
  log "Mooncake 풀 용량: $(curl -s --max-time 5 "http://${MC_MASTER%%:*}:9003/metrics" | awk '/^master_total_capacity_bytes /{printf "%.1f GiB", $2/1073741824}')"
fi

log "== smoke-mooncake: $MODEL, longbench ${NDOCS}건, max_model_len $MAXLEN, KV 예산 ${KVB}요청, host ${HOSTGB} GB, 조건($CONDS)"
for cond in $CONDS; do
  D=$O/$cond
  if [ -f $D/result.json ]; then
    [ $FORCE = 1 ] && rm -rf $D || { log "skip $cond (result.json 있음)"; continue; }
  fi
  rm -rf $D
  EXTRA=""; NEED_REMOTE=0
  case $cond in
    none)       kvt=none;     KVDIR=$LOCAL_ROOT;;
    mooncake)   kvt=mooncake; KVDIR=$LOCAL_ROOT
                EXTRA="--mooncake-master $MC_MASTER --mooncake-metadata $MC_META --mooncake-device $MC_DEV --mooncake-local-ip $MC_IP --mooncake-staging-gb $MC_STAGING_GB";;
    ours-ratio) kvt=hybrid;   KVDIR=$LOCAL_ROOT; NEED_REMOTE=1
                EXTRA="--kv-host-gb $HOSTGB --kv-placement ratio --kv-host-share $HOSTSHARE --kv-roots $LOCAL_ROOT:$W_LOCAL,$REMOTE_ROOT:$W_REMOTE";;
    *) log "unknown cond $cond"; continue;;
  esac
  clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
  mkdir -p "$LOCAL_ROOT"
  [ $NEED_REMOTE = 1 ] && mkdir -p "$REMOTE_ROOT"
  log "start $cond (kv-transport=$kvt$([ -n "$EXTRA" ] && echo ", $EXTRA"))"
  python run_obs.py --run-dir $D --model $MODEL --prompt-source longbench --n-docs $NDOCS \
    --decode-tokens $DECODE --max-model-len $MAXLEN --kv-batch $KVB --kv-block 64 \
    --no-weight-offload --settle-sec 3 --final-settle-sec 3 --poll-sleep-ms 0 \
    --ssd-root $O/ssd --kv-root "$KVDIR" --kv-transport $kvt $EXTRA > $O/$cond.log 2>&1
  rc=$?
  if [ -f $D/result.json ]; then log "done  $cond rc=$rc"; else
    log "FAILED $cond rc=$rc"; grep -aE 'Traceback|Error' $O/$cond.log | tail -2 | tee -a $O/campaign.log >/dev/null; fi
  clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
done
rm -rf $O/ssd
log "== smoke-mooncake 종료"
