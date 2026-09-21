#!/usr/bin/env bash
# 강제 적중 비교. "3경로 적재 자체가 Mooncake보다 빠른가"만 남기려고 write-behind, 적중률,
# 캐시 정책을 빼고 본다. 조건마다
#   cold_fill  문서 NDOCS건을 한 번 저장(질문 0)
#   커밋 확인  매니저 pending 0 + 워커 outstanding 0 + 디스크 파일 수 일치(ours),
#              저장 큐 비움 + master_key_count 정지(mooncake)
#   replay     같은 순서·같은 질문 0이라 프롬프트 토큰열이 cold_fill과 글자 그대로 같다.
#              --no-store-phase replay 로 저장을 막고 --reset-gpu-cache-before replay 로
#              GPU 프리픽스 캐시를 비워 모든 요청이 티어에서 적재하게 한다.
# 조건은 성능 비교 두 개(mooncake, ours-ratio)뿐이다. 두 조건 모두 host 예산 HOSTGB로 같다.
# /mnt/sunny-*/gdsio 는 건드리지 않는다. 지우는 것은 이 스크립트가 만든 kv-fh 디렉터리뿐.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
OUT=${OUT:-../../results/forced-hit-8b}
O=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a $O/campaign.log; }

MODEL=${MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4}
MAXLEN=${MAXLEN:-8192}; CAP=${CAP:-8000}; DECODE=${DECODE:-8}; KVB=${KVB:-2}; NDOCS=${NDOCS:-32}
HOSTGB=${HOSTGB:-8}; HOSTSHARE=${HOSTSHARE:-0.47}
W_LOCAL=${W_LOCAL:-13}; W_REMOTE=${W_REMOTE:-40}
LOCAL_ROOT=$O/kv-fh-local
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/sunny-nvmeof/kv-fh}
MC_MASTER=${MC_MASTER:-30.0.0.4:50051}; MC_META=${MC_META:-http://30.0.0.4:8080/metadata}
MC_DEV=${MC_DEV:-mlx5_1}; MC_IP=${MC_IP:-30.0.0.3}; MC_STAGING_GB=${MC_STAGING_GB:-8}
CONDS=${CONDS:-"mooncake ours-ratio"}
COMMIT_WAIT=${COMMIT_WAIT:-900}

# 파일 티어(원격 루트)를 쓰는 조건이 하나라도 있을 때만 /mnt/sunny-nvmeof 를 요구한다.
# mooncake와 재계산 기준만 도는 호스트(sunny)에는 이 마운트가 없다.
NEED_REMOTE_ANY=0
for _c in $CONDS; do case $_c in ref-none|none|b1-tiering|mooncake) ;; *) NEED_REMOTE_ANY=1;; esac; done
[ "$NEED_REMOTE_ANY" = 1 ] && { mountpoint -q /mnt/sunny-nvmeof || { log "ABORT: /mnt/sunny-nvmeof 이 마운트되어 있지 않음"; exit 2; }; }

clean_root(){
  case "$1" in */kv-fh|*/kv-fh-local) ;; *) log "ABORT: KV 디렉터리 이름이 kv-fh*가 아님: $1"; exit 3;; esac
  mountpoint -q "$1" 2>/dev/null && { log "ABORT: KV 디렉터리가 마운트 지점임: $1"; exit 3; }
  rm -rf "$1"
}

if [ "$NEED_REMOTE_ANY" = 1 ]; then
  FREE_GIB=$(df -B1 --output=avail /mnt/sunny-nvmeof | tail -1 | awk '{printf "%.1f", $1/1073741824}')
  log "원격 램디스크 여유: $FREE_GIB GiB"
fi
if echo "$CONDS" | grep -qw mooncake; then
  curl -s --max-time 5 "http://${MC_MASTER%%:*}:9003/metrics" | grep -q master_total_capacity_bytes \
    || { log "ABORT: Mooncake master 메트릭에 닿지 않음(sunny에서 ~/bin/mooncake_up.sh 실행 필요)"; exit 5; }
  log "Mooncake 풀 용량: $(curl -s --max-time 5 "http://${MC_MASTER%%:*}:9003/metrics" | awk '/^master_total_capacity_bytes /{printf "%.1f GiB", $2/1073741824}')"
fi

log "== forced_hit: $MODEL, ${NDOCS}건, max_model_len $MAXLEN, 프리픽스 상한 $CAP, KV 예산 ${KVB}요청, host ${HOSTGB} GB, 가중치 ${W_LOCAL}:${W_REMOTE}, host 몫 $HOSTSHARE, 조건($CONDS)"
for cond in $CONDS; do
  D=$O/$cond
  if [ -f $D/result.json ]; then
    [ $FORCE = 1 ] && rm -rf $D || { log "skip $cond (result.json 있음)"; continue; }
  fi
  rm -rf $D
  EXTRA=""; NEED_REMOTE=0
  case $cond in
    mooncake)   kvt=mooncake; KVDIR=$LOCAL_ROOT
                EXTRA="--mooncake-master $MC_MASTER --mooncake-metadata $MC_META --mooncake-device $MC_DEV --mooncake-local-ip $MC_IP --mooncake-staging-gb $MC_STAGING_GB";;
    ours-ratio) kvt=hybrid;  KVDIR=$LOCAL_ROOT; NEED_REMOTE=1
                EXTRA="--kv-host-gb $HOSTGB --kv-placement ratio --kv-host-share $HOSTSHARE --kv-roots $LOCAL_ROOT:$W_LOCAL,$REMOTE_ROOT:$W_REMOTE";;
    *) log "unknown cond $cond"; continue;;
  esac
  clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
  mkdir -p "$LOCAL_ROOT"
  [ $NEED_REMOTE = 1 ] && mkdir -p "$REMOTE_ROOT"
  log "start $cond (kv-transport=$kvt$([ -n "$EXTRA" ] && echo ", $EXTRA"))"
  python run_obs.py --run-dir $D --model $MODEL --prompt-source longbench --n-docs $NDOCS \
    --mode forced_hit --kv-trace --no-store-phase replay --reset-gpu-cache-before replay \
    --commit-check-sec $COMMIT_WAIT \
    --prompt-cap $CAP --decode-tokens $DECODE --max-model-len $MAXLEN --kv-batch $KVB --kv-block 64 \
    --gpu-util 0.9 --no-weight-offload --settle-sec 5 --final-settle-sec 5 --poll-sleep-ms 0 \
    --ssd-root $O/ssd --kv-root "$KVDIR" --kv-transport $kvt $EXTRA > $O/$cond.log 2>&1
  rc=$?
  if [ -f $D/result.json ]; then log "done  $cond rc=$rc"; else
    log "FAILED $cond rc=$rc"; grep -aE 'Traceback|Error' $O/$cond.log | tail -2 | tee -a $O/campaign.log >/dev/null; fi
  clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
done
rm -rf $O/ssd
log "== forced_hit 종료"
