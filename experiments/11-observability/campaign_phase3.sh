#!/usr/bin/env bash
# phase3 본실험. 합의한 기준선 하나와 우리 조건 하나만 성능 비교에 쓴다.
#   ref-none    재계산(KV 오프로드 끔). 출력 토큰열 정합성 기준이며 성능 비교 행이 아니다.
#   b1-tiering  vLLM in-tree TieringOffloadingSpec. CPU 1차 티어 + fs 2차 티어(로컬 SSD),
#               전 구간이 host DRAM 경유. 기준선.
#   ours-ratio  포크 HybridSpec, 배치 ratio(host 몫 HOSTSHARE) + 파일 티어 루트 2개
#               (로컬 NVMe GDS W_LOCAL : 원격 호스트 DRAM NVMe-oF RDMA + GDS W_REMOTE).
#   mooncake    논문 기준선(Mooncake Store + Transfer Engine, FAST 2025). KV 풀은 원격 호스트
#               DRAM 한 층이고 전송은 RDMA. 원격 서비스는 sunny의 ~/bin/mooncake_up.sh.
#   current / pending-wait / cpu-landing
#               write-behind miss 대책 비교용. 티어 구성은 ours-ratio와 같고 조건 정의는 case 절 참고.
# KV_EXTRA에 JSON을 주면 모든 조건의 --kv-extra에 합쳐진다(조건별 값이 우선).
# GPU 런은 flock "$GPULOCK"(기본 /tmp/claude-gpu.lock)으로 하나씩 줄 세운다. GPULOCK=""이면 끈다.
# MODE=stream이면 longbench 두 라운드 대신 bailian trace를 도착 시각대로 한 줄로 흘린다
#   (--prompt-source bailian --bailian-offset BOFF --mode stream --time-scale TSCALE
#    --max-concurrency MAXCONC). 이때 CONDS는 성능 비교 두 조건(mooncake, ours-ratio)만 쓴다.
# 두 조건 모두 host 예산 HOSTGB로 같고, 티어 스레드 수도 --kv-threads 한 값으로 같다.
# 가중치 13:40은 두 파일 경로의 대역 몫, host 몫 0.47은 조건 ours-ratio가 host로 보내는 블록 비율.
# 조건별로 result.json이 있으면 건너뜀. --force면 해당 조건 결과 폴더를 지우고 다시 돌림.
# /mnt/sunny-*/gdsio 는 건드리지 않는다. 지우는 것은 이 스크립트가 만든 kv-p3 디렉터리뿐.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
MODE=${MODE:-phases}
if [ "$MODE" = stream ]; then OUT=${OUT:-../../results/phase3-stream-8b}; NDOCS=${NDOCS:-96}
else OUT=${OUT:-../../results/phase3-8k-host8}; NDOCS=${NDOCS:-32}; fi
O=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a $O/campaign.log; }

MODEL=${MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4}
MAXLEN=${MAXLEN:-8192}; CAP=${CAP:-8000}; DECODE=${DECODE:-8}; KVB=${KVB:-2}
HOSTGB=${HOSTGB:-8}; HOSTSHARE=${HOSTSHARE:-0.47}
W_LOCAL=${W_LOCAL:-13}; W_REMOTE=${W_REMOTE:-40}
LOCAL_ROOT=$O/kv-p3-local
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/sunny-nvmeof/kv-p3}
BOFF=${BOFF:-29560}; TSCALE=${TSCALE:-100}; MAXCONC=${MAXCONC:-4}
MC_MASTER=${MC_MASTER:-30.0.0.4:50051}; MC_META=${MC_META:-http://30.0.0.4:8080/metadata}
MC_DEV=${MC_DEV:-mlx5_1}; MC_IP=${MC_IP:-30.0.0.3}; MC_STAGING_GB=${MC_STAGING_GB:-8}
if [ "$MODE" = stream ]; then
  CONDS=${CONDS:-"mooncake ours-ratio"}
  # BTRACE=트레이스 파일(기본 bailian), BBLOCK=hash_id 하나의 토큰 수(bailian 16, Mooncake 공개 트레이스 512).
  # Mooncake 트레이스는 timestamp가 ms라 TSCALE에 1/1000을 곱해 준다(예: 배율 30 → 0.03).
  BTRACE=${BTRACE:-}; BBLOCK=${BBLOCK:-16}
  WORKLOAD="--prompt-source bailian --bailian-offset $BOFF --bailian-block $BBLOCK ${BTRACE:+--bailian-trace $BTRACE} --mode stream --time-scale $TSCALE --max-concurrency $MAXCONC"
else
  CONDS=${CONDS:-"ref-none b1-tiering ours-ratio"}
  WORKLOAD="--prompt-source longbench"
fi
# 원격 램디스크 여유에서 남겨둘 몫(GiB). 파일 티어에는 축출이 없어 저장 라운드에 쓴 만큼 그대로 쌓인다.
REMOTE_MARGIN_GIB=${REMOTE_MARGIN_GIB:-3}

# 파일 티어(원격 루트)를 쓰는 조건이 하나라도 있을 때만 /mnt/sunny-nvmeof 를 요구한다.
# mooncake와 재계산 기준만 도는 호스트(sunny)에는 이 마운트가 없다.
NEED_REMOTE_ANY=0
for _c in $CONDS; do case $_c in ref-none|none|b1-tiering|mooncake) ;; *) NEED_REMOTE_ANY=1;; esac; done
[ "$NEED_REMOTE_ANY" = 1 ] && { mountpoint -q /mnt/sunny-nvmeof || { log "ABORT: /mnt/sunny-nvmeof 이 마운트되어 있지 않음"; exit 2; }; }

# 이 스크립트가 만든 KV 디렉터리만 지운다(이름이 kv-p3*가 아니거나 마운트 지점이면 중단).
clean_root(){
  case "$1" in */kv-p3|*/kv-p3-local) ;; *) log "ABORT: KV 디렉터리 이름이 kv-p3*가 아님: $1"; exit 3;; esac
  mountpoint -q "$1" 2>/dev/null && { log "ABORT: KV 디렉터리가 마운트 지점임: $1"; exit 3; }
  rm -rf "$1"
}

# 원격 루트 용량 사전 검사. hybrid는 write_through가 기본이라 저장한 블록이 전부 파일 티어에도
# 남고, 그중 W_REMOTE/(W_LOCAL+W_REMOTE) 몫이 원격 루트로 간다. 여유를 넘으면 NDOCS를 줄인다.
if [ "$NEED_REMOTE_ANY" = 0 ]; then
  : # 원격 파일 티어를 쓰는 조건이 없으면 용량 사전 검사가 필요 없다.
elif [ "$MODE" = stream ]; then
  # stream은 프롬프트 길이가 trace마다 달라 longbench식 상한 계산이 맞지 않는다.
  # 원격 램디스크 여유만 확인하고, 실제 필요량은 런 뒤 kv_files_by_root로 본다.
  FREE_GIB=$(df -B1 --output=avail /mnt/sunny-nvmeof | tail -1 | awk '{printf "%.1f", $1/1073741824}')
  log "원격 램디스크 여유: $FREE_GIB GiB (stream 모드라 longbench 용량 사전 검사는 건너뜀)"
  awk -v f="$FREE_GIB" 'BEGIN{exit !(f < 30)}' && { log "ABORT: 원격 램디스크 여유가 30 GiB 미만"; exit 4; }
elif echo "$CONDS" | grep -qw ours-ratio; then
  FREE_B=$(df -B1 --output=avail /mnt/sunny-nvmeof | tail -1)
  FIT=$(python - "$MODEL" "$NDOCS" "$CAP" "$W_LOCAL" "$W_REMOTE" "$FREE_B" "$REMOTE_MARGIN_GIB" <<'PY'
import sys
from transformers import AutoConfig
model, ndocs, cap, wl, wr, free_b, margin = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
c = AutoConfig.from_pretrained(model)
kvh = int(getattr(c, "num_key_value_heads", None) or c.num_attention_heads)
hd = c.hidden_size // c.num_attention_heads
per_tok = 2 * int(c.num_hidden_layers) * kvh * hd * 2  # K+V x layer x kv_head x head_dim x fp16
per_doc = per_tok * cap
share = wr / (wl + wr)
usable = max(0.0, free_b - margin * 2**30)
fit = int(usable / (per_doc * share))
print(ndocs, min(ndocs, fit), round(per_doc * ndocs * share / 2**30, 1), round(usable / 2**30, 1))
PY
) || { log "ABORT: 원격 용량 사전 검사 실패"; exit 4; }
  set -- $FIT
  log "원격 용량 검사: 문서 $1건이면 원격 루트에 $3 GiB 필요, 여유 $4 GiB(마진 $REMOTE_MARGIN_GIB GiB 제외)"
  if [ "$2" -lt "$1" ]; then
    [ "$2" -lt 4 ] && { log "ABORT: 원격 여유가 너무 적음(문서 $2건)"; exit 4; }
    log "NDOCS $1 -> $2 (원격 램디스크 여유에 맞춤)"; NDOCS=$2
  fi
fi

if echo "$CONDS" | grep -qw mooncake; then
  curl -s --max-time 5 "http://${MC_MASTER%%:*}:9003/metrics" | grep -q master_total_capacity_bytes \
    || { log "ABORT: Mooncake master 메트릭에 닿지 않음(sunny에서 ~/bin/mooncake_up.sh 실행 필요)"; exit 5; }
  log "Mooncake 풀 용량: $(curl -s --max-time 5 "http://${MC_MASTER%%:*}:9003/metrics" | awk '/^master_total_capacity_bytes /{printf "%.1f GiB", $2/1073741824}')"
fi

log "== phase3($MODE): $MODEL, ${NDOCS}건, max_model_len $MAXLEN, 프리픽스 상한 $CAP, KV 예산 ${KVB}요청, host ${HOSTGB} GB, 가중치 ${W_LOCAL}:${W_REMOTE}, host 몫 $HOSTSHARE, 조건($CONDS)"
for cond in $CONDS; do
  D=$O/$cond
  if [ -f $D/result.json ]; then
    [ $FORCE = 1 ] && rm -rf $D || { log "skip $cond (result.json 있음)"; continue; }
  fi
  rm -rf $D
  EXTRA=""; NEED_REMOTE=0
  # 우리 조건 공통 인자. 배치 정책만 조건마다 다르다.
  OURS="--kv-host-gb $HOSTGB --kv-roots $LOCAL_ROOT:$W_LOCAL,$REMOTE_ROOT:$W_REMOTE"
  OURS_RATIO="--kv-placement ratio --kv-host-share $HOSTSHARE"
  case $cond in
    ref-none)   kvt=none;    KVDIR=$LOCAL_ROOT;;
    b1-tiering) kvt=tiering; KVDIR=$LOCAL_ROOT; EXTRA="--kv-host-gb $HOSTGB";;
    mooncake)   kvt=mooncake; KVDIR=$LOCAL_ROOT
                EXTRA="--mooncake-master $MC_MASTER --mooncake-metadata $MC_META --mooncake-device $MC_DEV --mooncake-local-ip $MC_IP --mooncake-staging-gb $MC_STAGING_GB";;
    ours-ratio) kvt=hybrid;  KVDIR=$LOCAL_ROOT; NEED_REMOTE=1; EXTRA="$OURS $OURS_RATIO";;
    # write-behind miss 대책 세 조건. 셋 다 ours-ratio와 같은 티어 구성이고 바뀌는 건 아래뿐.
    #   current      지금 동작 그대로(대조군)
    #   pending-wait 쓰기 중인 키를 만나면 남은 쓰기 시간과 재계산 시간을 재 보고 기다림
    #   cpu-landing  배치를 host_first로 바꿔 모든 블록이 host 티어에 먼저 앉음(write-through)
    current)      kvt=hybrid; KVDIR=$LOCAL_ROOT; NEED_REMOTE=1; EXTRA="$OURS $OURS_RATIO";;
    pending-wait) kvt=hybrid; KVDIR=$LOCAL_ROOT; NEED_REMOTE=1; EXTRA="$OURS $OURS_RATIO"
                  COND_KV_EXTRA='{"cufile_fs_pending_wait": true}';;
    cpu-landing)  kvt=hybrid; KVDIR=$LOCAL_ROOT; NEED_REMOTE=1; EXTRA="$OURS --kv-placement host_first";;
    *) log "unknown cond $cond"; continue;;
  esac
  # KV_EXTRA(캠페인 전체) + 조건별 COND_KV_EXTRA를 합쳐 --kv-extra로 넘긴다.
  kvextra=()
  MERGED=$(KV_EXTRA="${KV_EXTRA:-}" COND="${COND_KV_EXTRA:-}" python -c '
import json, os
a = json.loads(os.environ["KV_EXTRA"] or "{}"); a.update(json.loads(os.environ["COND"] or "{}"))
print(json.dumps(a) if a else "")')
  [ -n "$MERGED" ] && kvextra=(--kv-extra "$MERGED")
  COND_KV_EXTRA=""
  # NSYS=1 이면 lib/obs/run_nsys.sh로 감싸 stream(또는 phases의 cold_fill) 구간을 cudaProfilerApi 범위로 캡처.
  # NSYS_STEPS=0이면 phase 전체. 리포트는 $O/<cond>-nsys/, 분석은 nsys.done 뒤에만.
  wrap=(); nsysargs=()
  if [ "${NSYS:-0}" = 1 ]; then
    mkdir -p $O/$cond-nsys; wrap=(../../lib/obs/run_nsys.sh $O/$cond-nsys --); export NSYS_CAPTURE=cudaProfilerApi
    nsysargs=(--nsys-phase "${NSYS_PHASE:-$([ "$MODE" = stream ] && echo stream || echo cold_fill)}" --nsys-steps "${NSYS_STEPS:-0}")
  fi
  # KV 루트를 비우는 것부터 런이 끝날 때까지가 한 덩어리다. 캠페인이 여러 개 떠 있어도
  # 남의 런이 쓰는 원격 루트를 지우지 않도록 이 구간 전체를 GPU 잠금 안에서 돈다.
  run_cond(){
    clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
    mkdir -p "$LOCAL_ROOT"
    [ $NEED_REMOTE = 1 ] && mkdir -p "$REMOTE_ROOT"
    log "start $cond (kv-transport=$kvt, kv-root=$KVDIR$([ -n "$EXTRA" ] && echo ", $EXTRA")$([ ${#kvextra[@]} -gt 0 ] && echo ", --kv-extra ${kvextra[1]}"))"
    "${wrap[@]}" python run_obs.py --run-dir $D --model $MODEL $WORKLOAD --n-docs $NDOCS "${nsysargs[@]}" \
      --prompt-cap $CAP --decode-tokens $DECODE --max-model-len $MAXLEN --kv-batch $KVB --kv-block 64 \
      --gpu-util 0.9 --no-weight-offload --settle-sec 5 --final-settle-sec 5 --poll-sleep-ms 0 \
      --ssd-root $O/ssd --kv-root "$KVDIR" --kv-transport $kvt $EXTRA "${kvextra[@]}" > $O/$cond.log 2>&1
    rc=$?
    if [ -f $D/result.json ]; then log "done  $cond rc=$rc"; else
      log "FAILED $cond rc=$rc"; grep -aE 'Traceback|Error' $O/$cond.log | tail -2 | tee -a $O/campaign.log >/dev/null; fi
    clean_root "$LOCAL_ROOT"; clean_root "$REMOTE_ROOT"
  }
  # GPU는 다른 작업과 나눠 쓰므로 조건 하나씩 flock으로 줄 세운다(GPULOCK=""이면 끔).
  GL="${GPULOCK-/tmp/claude-gpu.lock}"
  if [ -n "$GL" ]; then ( flock 9; run_cond ) 9>"$GL"; else run_cond; fi
done
rm -rf $O/ssd
log "== phase3($MODE) 종료"
