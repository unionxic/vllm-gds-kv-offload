#!/usr/bin/env bash
# Qwen2.5-72B-Instruct 기준선. host memory 비율(RAM 대비) RATIOS(기본 0.5) × 조건(none=재계산, cufile=SSD 적중, cufile-wb=SSD 적중 + write-behind 30 s).
# 입력 LongBench-v2 32건(data/longbench-v2-10k-32.jsonl), GPU KV·block_size는 vLLM 기본(--pure). VLLM_OFFLOAD_PIN_EXACT=1과 memguard는 안전장치.
# SRC=bailian 이면 02의 Bailian 트레이스 앞 NDOCS건(프리픽스 공유 구조 재현). CONDS에 cufile-lru|cufile-lfu(용량 CAPGB, 기본 12)|cufile-seen2(admission) 추가 가능. UTILS로 gpu_util 시도 순서.
# KVB=2.27 이면 --pure 대신 GPU KV를 요청 2.27개분(=기준 런의 자동 예산 7.0 GiB, 21k 토큰)으로 고정. prefetch 깊이 2·chunk 2048에서 자동 예산이 KV를 키워 샘플러가 OOM 나는 것을 막음.
# MODE=stream 이면 run_obs.py를 trace 순서 open-loop 스트림으로(--mode stream). TSCALE(기본 1)=도착 간격 배율, MAXCONC(기본 0=무제한)=동시 미완료 상한, AGAP=longbench 합성 간격.
#   태그에 -stream-ts<TSCALE>-c<MAXCONC>가 붙음. MODE 미지정이면 기존 phases 동작 그대로.
# NSYS=1 이면 lib/obs/run_nsys.sh로 감싸고 NSYS_PHASE(기본 cold_fill)의 앞 NSYS_STEPS(기본 40) forward만 캡처(cudaProfilerApi 구간).
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
unset CUFILE_ENV_PATH_JSON VLLM_OFFLOAD_SSD_REGISTER_MAX_MB; [ -n "${CUFILE_JSON_OVERRIDE:-}" ] && export CUFILE_ENV_PATH_JSON=$CUFILE_JSON_OVERRIDE; export VLLM_OFFLOAD_TIER_LAYOUT=${LAYOUT:-block}; [ -n "${GATE:-}" ] && export VLLM_KV_LOAD_WAVE_GATE=$GATE || unset VLLM_KV_LOAD_WAVE_GATE
MODEL=Qwen/Qwen2.5-72B-Instruct; O=$(mkdir -p ../../results/qwen72b && cd ../../results/qwen72b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
WB='{"cufile_fs_store_window": "host", "cufile_fs_store_window_max_s": 30}'
sargs=(); stag=""
if [ "${MODE:-phases}" = stream ]; then
  sargs=(--mode stream --time-scale ${TSCALE:-1} --max-concurrency ${MAXCONC:-0} --arrival-gap-s ${AGAP:-0})
  stag="-stream-ts${TSCALE:-1}-c${MAXCONC:-0}"
fi
log "== qwen72b: RAM ${RATIOS:-0.5} × (${CONDS:-none cufile cufile-wb}), docs ${NDOCS:-32}, decode ${DECODE:-8}, NSYS=${NSYS:-0}, MODE=${MODE:-phases}${stag}"
for f in ${RATIOS:-0.5}; do for cond in ${CONDS:-none cufile cufile-wb}; do tag=${SRC:+$SRC-}ram$f-$cond${LAYOUT:+-$LAYOUT}${PSTEP:+-p$PSTEP}${SPLIT:+-split}${stag}${TAGSUF:-}
  [ -f $O/$tag/result.json ] && { log "skip $tag"; continue; }
  CAP=${CAPGB:-12}
  case $cond in none) kvt=none; extra=();; cufile) kvt=cufile; extra=();; cufile-wb) kvt=cufile; extra=(--kv-extra "$WB");;
    cufile-lru) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_capacity_gb\": $CAP, \"cufile_fs_policy\": \"lru\"}");;
    cufile-lfu) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_capacity_gb\": $CAP, \"cufile_fs_policy\": \"lfu\"}");;
    cufile-seen2) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_admission\": \"seen_twice\"}");;
    cufile-wb-lru) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_store_window\": \"host\", \"cufile_fs_store_window_max_s\": 30, \"cufile_fs_capacity_gb\": $CAP, \"cufile_fs_policy\": \"lru\"}");;
    hybrid) kvt=hybrid; extra=(--kv-host-gb ${KVHOSTGB:-8} --kv-extra "{\"hybrid_host_policy\": \"${CPUPOL:-lru}\", \"hybrid_placement\": \"${PLACEMENT:-host_first}\"${PROFILE:+, \"hybrid_profile\": \"$PROFILE\"}}"); tag=$tag${KVHOSTGB:-8}gb-${CPUPOL:-lru}-${PLACEMENT:-host_first};;
    cpu) kvt=cpu; extra=(--kv-host-gb ${KVHOSTGB:-8} --kv-extra "{\"eviction_policy\": \"${CPUPOL:-lru}\"}"); tag=$tag${KVHOSTGB:-8}gb-${CPUPOL:-lru};;
    *) log "unknown cond $cond"; continue;; esac
  rm -rf $O/kv-72b; ../../lib/obs/memguard.sh $tag $O/memguard.log & g=$!
  wrap=(); [ "${NSYS:-0}" = 1 ] && { wrap=(../../lib/obs/run_nsys.sh $O/$tag-nsys --); export NSYS_CAPTURE=cudaProfilerApi; extra+=(--nsys-phase "${NSYS_PHASE:-cold_fill}" --nsys-steps "${NSYS_STEPS:-40}"); mkdir -p $O/$tag-nsys; }
  avail_gb=$(df -BG --output=avail $O | tail -1 | tr -dc 0-9); [ "${avail_gb:-0}" -lt 60 ] && { log "SKIP $tag: 디스크 여유 ${avail_gb} GB < 60 GB"; kill $g 2>/dev/null; continue; }
  log "start $tag"
  for util in ${UTILS:-0.9 0.85}; do
    "${wrap[@]}" python run_obs.py --run-dir $O/$tag --model $MODEL --prompt-source ${SRC:-longbench} --bailian-offset ${BOFF:-0} --n-docs ${NDOCS:-32} --decode-tokens ${DECODE:-8} --max-model-len ${MAXLEN:-12288} \
      --host-ram-fraction $f --prefetch-step ${PSTEP:-1} --max-num-batched-tokens ${MNBT:-0} --kv-split ${SPLIT:-off} $([ -n "${KVB:-}" ] && echo "--kv-batch $KVB --kv-block 64" || echo --pure) --poll-sleep-ms 0 --gpu-util $util --kv-transport $kvt --settle-sec 15 --final-settle-sec 15 \
      --profile-out $O/$tag-profile.json --ssd-root $O/ssd-72b --kv-root $O/kv-72b "${sargs[@]}" "${extra[@]}" > $O/$tag.log 2>&1
    rc=$?
    [ -f $O/$tag/result.json ] && break
    if grep -aq "OutOfMemoryError" $O/$tag.log; then log "OOM $tag (gpu_util $util) → 재시도"; rm -rf $O/$tag $O/kv-72b; continue; fi
    break
  done
  kill $g 2>/dev/null; wait $g 2>/dev/null; rm -rf $O/kv-72b
  [ -f $O/$tag/result.json ] && log "done  $tag rc=$rc (gpu_util $util)" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error|KILL' $O/$tag.log $O/memguard.log | tail -2 >> $O/campaign.log; }
done; done
log "== qwen72b 종료"
