#!/usr/bin/env bash
# GPU 공급 경로 세 갈래(host pinned DRAM H2D / 로컬 NVMe GDS / 원격 호스트 DRAM NVMe-oF RDMA + GDS)를
# 한 프리픽스 안에서 섞는 배치 smoke. 조건 4개를 순서대로 한 번씩 돌려 result.json을 남긴다.
#   none        재계산(KV 오프로드 끔). 출력 토큰열 기준.
#   host-first  hybrid, 단일 루트(로컬 SSD) + host 티어 0.5 GB, 배치 host_first(기존 동작)
#   ratio       hybrid, 배치 ratio(host 몫 0.47) + 루트 2개(로컬 SSD 13 : 원격 DRAM 40)
#   multi-root  cufile 전용(host 티어 없음), 루트 2개(로컬 SSD 13 : 원격 DRAM 40)
# 가중치 비 13:40은 두 경로의 대역 몫을 그대로 쓴 것이고, host 몫 0.47은 조건 ratio에서
# host 경로가 받는 블록 비율이다. 워크로드 플래그는 smoke_remote_tier.sh와 같다.
# 조건별로 result.json이 있으면 건너뜀. --force면 해당 조건 결과 폴더를 지우고 다시 돌림.
# /mnt/sunny-*/gdsio 는 건드리지 않는다. 지우는 것은 이 스크립트가 만든 kv-p2 디렉터리뿐.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
OUT=${OUT:-../../results/smoke-three-path}  # 결과 폴더. 다른 모델로 돌릴 때 OUT을 바꿔 3B 결과를 덮지 않게 함
O=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a $O/campaign.log; }

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
NDOCS=${NDOCS:-8}; MAXLEN=${MAXLEN:-4096}; DECODE=${DECODE:-8}; KVB=${KVB:-2}
HOSTGB=${HOSTGB:-0.5}; HOSTSHARE=${HOSTSHARE:-0.47}
W_LOCAL=${W_LOCAL:-13}; W_REMOTE=${W_REMOTE:-40}
LOCAL_ROOT=$O/kv-p2-local
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/sunny-nvmeof/kv-p2}
REMOTE_MNT=${REMOTE_MNT:-$(dirname "$REMOTE_ROOT")}  # 원격 티어 마운트 지점(rain에서는 /mnt/sunny-nvmeof, sunny에서는 /mnt/rain-nvmeof)
CONDS=${CONDS:-"none host-first ratio multi-root"}

mountpoint -q "$REMOTE_MNT" || { log "ABORT: $REMOTE_MNT 이 마운트되어 있지 않음"; exit 2; }

# 이 스크립트가 만든 KV 디렉터리만 지운다(이름이 kv-p2*가 아니거나 마운트 지점이면 중단).
clean_root(){
  case "$1" in */kv-p2|*/kv-p2-local) ;; *) log "ABORT: KV 디렉터리 이름이 kv-p2*가 아님: $1"; exit 3;; esac
  mountpoint -q "$1" 2>/dev/null && { log "ABORT: KV 디렉터리가 마운트 지점임: $1"; exit 3; }
  rm -rf "$1"
}

log "== smoke-three-path: $MODEL, longbench ${NDOCS}건, max_model_len $MAXLEN, KV 예산 ${KVB}요청, host ${HOSTGB} GB, 가중치 ${W_LOCAL}:${W_REMOTE}, host 몫 $HOSTSHARE, 조건($CONDS)"
for cond in $CONDS; do
  D=$O/$cond
  if [ -f $D/result.json ]; then
    [ $FORCE = 1 ] && rm -rf $D || { log "skip $cond (result.json 있음)"; continue; }
  fi
  rm -rf $D
  EXTRA=""
  case $cond in
    none)       kvt=none;   KVDIR=$LOCAL_ROOT;;
    host-first) kvt=hybrid; KVDIR=$LOCAL_ROOT; EXTRA="--kv-host-gb $HOSTGB --kv-placement host_first";;
    ratio)      kvt=hybrid; KVDIR=$LOCAL_ROOT
                EXTRA="--kv-host-gb $HOSTGB --kv-placement ratio --kv-host-share $HOSTSHARE --kv-roots $LOCAL_ROOT:$W_LOCAL,$REMOTE_ROOT:$W_REMOTE";;
    multi-root) kvt=cufile; KVDIR=$LOCAL_ROOT
                EXTRA="--kv-roots $LOCAL_ROOT:$W_LOCAL,$REMOTE_ROOT:$W_REMOTE";;
    *) log "unknown cond $cond"; continue;;
  esac
  clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
  mkdir -p "$LOCAL_ROOT"
  case $cond in ratio|multi-root) mkdir -p "$REMOTE_ROOT";; esac
  log "start $cond (kv-transport=$kvt, kv-root=$KVDIR$([ -n "$EXTRA" ] && echo ", $EXTRA"))"
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
log "== smoke-three-path 종료"
