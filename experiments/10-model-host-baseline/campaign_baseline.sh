#!/usr/bin/env bash
# 모델 크기 × host 비율 기준표. 실제 프롬프트(03-leval 문서 8개, 프리픽스 1,920 + 질문), KV는 SSD(expfs cuFile), 게이트 2.
# 조건: none(재계산) / cufile(1라운드 저장, 2라운드 적중). 상주 0층(전 층 스트리밍), GPU KV는 요청 2개분(배치 2), decode 8토큰.
# host 비율은 오프로드 가중치 대비(RAM 대비 아님). 1.0은 SSD 없이 GPU+CPU.
# 사용: ./campaign_baseline.sh [모델 목록]   기본 순서: opt-66b(캐시됨) → 6.7b → 13b → 30b(내려받고 앞 모델 캐시는 지움)
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
export CUFILE_ENV_PATH_JSON=$(cd ../08-cufile-bounce && pwd)/cufile-pb4096.json
O=$(cd ../../results/model-host-baseline && pwd); R=$(cd ../09-kv-policy && pwd); HF=~/.cache/huggingface/hub
DOCS=8; DECODE=8; KVB=2; MAXLEN=2048
declare -A LAYERS=([opt-2.7b]=32 [opt-6.7b]=32 [opt-13b]=40 [opt-30b]=48 [opt-66b]=64)
declare -A FRACS=([opt-2.7b]="1.0 0.7 0.5 0.3 0.1" [opt-6.7b]="1.0 0.7 0.5 0.3 0.1" [opt-13b]="1.0 0.7 0.5 0.3 0.1" [opt-30b]="1.0 0.7 0.5 0.3 0.1" [opt-66b]="0.7 0.5 0.3 0.1")
MODELS=${*:-"opt-66b opt-6.7b opt-13b opt-30b"}
log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }

run(){  # run <model> <frac> <kvt> <gate>
  local m=$1 f=$2 kvt=$3 gate=$4 tag="$1-h$2-$3" L=${LAYERS[$1]}
  [ -f $O/$tag.json ] && { log "skip $tag (있음)"; return 0; }
  rm -rf $O/ssd-$m $O/kv-$m
  ../07-combined/memguard.sh $tag $O/memguard.log & local guard=$!
  log "start $tag (layers $L, gate $gate)"
  VLLM_KV_LOAD_WAVE_GATE=$gate python $R/run_phase_66b.py --model facebook/$m --tag $tag --out-dir $O \
    --prompt-source leval --n-prompts $DOCS --rounds 2 --decode-tokens $DECODE --max-model-len $MAXLEN \
    --host-weight-fraction $f --group-size $L --num-in-group $L --kv-batch $KVB --kv-transport $kvt \
    --ssd-root $O/ssd-$m --kv-root $O/kv-$m > $O/$tag.log 2>&1
  local rc=$?; kill $guard 2>/dev/null; wait $guard 2>/dev/null
  if [ -f $O/$tag.json ]; then log "done  $tag rc=$rc"; else log "FAILED $tag rc=$rc"; grep -aE 'Error|Killed|out of memory|KILL|Traceback' $O/$tag.log $O/memguard.log | tail -3 >> $O/campaign.log; fi
  rm -rf $O/kv-$m
}

fetch(){ [ -d $HF/models--facebook--$1 ] && return 0; log "download $1"; huggingface-cli download facebook/$1 --exclude "*.h5" "*.msgpack" "*.ot" "*safetensors.index*" >> $O/download.log 2>&1 || python -c "from huggingface_hub import snapshot_download as s; s('facebook/$1', allow_patterns=['*.json','*.txt','*.bin','*.safetensors'])" >> $O/download.log 2>&1; log "download $1 done rc=$?"; }

log "== 기준표 캠페인 시작: $MODELS (docs $DOCS, decode $DECODE, KV 배치 $KVB)"
for m in $MODELS; do
  if [ "$m" = opt-30b ]; then for old in opt-6.7b opt-13b; do [ -d $HF/models--facebook--$old ] && { log "cache 삭제 $old (디스크)"; rm -rf $HF/models--facebook--$old; }; done; fi
  fetch $m || { log "FAILED download $m"; continue; }
  for f in ${FRACS[$m]}; do
    run $m $f none 0
    run $m $f cufile 2
  done
  rm -rf $O/ssd-$m
  log "== $m 완료: $(df -h / | awk 'NR==2{print $4}') 여유"
done
log "== 캠페인 종료"
