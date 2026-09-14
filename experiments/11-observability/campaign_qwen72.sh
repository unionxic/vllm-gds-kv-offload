#!/usr/bin/env bash
# Qwen2.5-72B-Instruct 기준선. host memory 비율(RAM 대비) RATIOS(기본 0.5) × 조건(none=재계산, cufile=SSD 적중, cufile-wb=SSD 적중 + write-behind 30 s).
# 입력 LongBench-v2 32건(data/longbench-v2-10k-32.jsonl), GPU KV·block_size는 vLLM 기본(--pure). VLLM_OFFLOAD_PIN_EXACT=1과 memguard는 안전장치.
# SRC=bailian 이면 02의 Bailian 트레이스 앞 NDOCS건(프리픽스 공유 구조 재현). CONDS에 cufile-lru|cufile-lfu(용량 CAPGB, 기본 12)|cufile-seen2(admission) 추가 가능. UTILS로 gpu_util 시도 순서.
# NSYS=1 이면 lib/obs/run_nsys.sh로 감싸고 NSYS_PHASE(기본 cold_fill)의 앞 NSYS_STEPS(기본 40) forward만 캡처(cudaProfilerApi 구간).
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
unset CUFILE_ENV_PATH_JSON VLLM_OFFLOAD_SSD_REGISTER_MAX_MB; [ -n "${GATE:-}" ] && export VLLM_KV_LOAD_WAVE_GATE=$GATE || unset VLLM_KV_LOAD_WAVE_GATE
MODEL=Qwen/Qwen2.5-72B-Instruct; O=$(mkdir -p ../../results/qwen72b && cd ../../results/qwen72b && pwd); log(){ echo "[$(date +%m-%d\ %H:%M:%S)] $*" >> $O/campaign.log; }
WB='{"cufile_fs_store_window": "host", "cufile_fs_store_window_max_s": 30}'
log "== qwen72b: RAM ${RATIOS:-0.5} × (${CONDS:-none cufile cufile-wb}), docs ${NDOCS:-32}, decode ${DECODE:-8}, NSYS=${NSYS:-0}"
for f in ${RATIOS:-0.5}; do for cond in ${CONDS:-none cufile cufile-wb}; do tag=${SRC:+$SRC-}ram$f-$cond${TAGSUF:-}
  [ -f $O/$tag/result.json ] && { log "skip $tag"; continue; }
  CAP=${CAPGB:-12}
  case $cond in none) kvt=none; extra=();; cufile) kvt=cufile; extra=();; cufile-wb) kvt=cufile; extra=(--kv-extra "$WB");;
    cufile-lru) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_capacity_gb\": $CAP, \"cufile_fs_policy\": \"lru\"}");;
    cufile-lfu) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_capacity_gb\": $CAP, \"cufile_fs_policy\": \"lfu\"}");;
    cufile-seen2) kvt=cufile; extra=(--kv-extra "{\"cufile_fs_admission\": \"seen_twice\"}");;
    *) log "unknown cond $cond"; continue;; esac
  rm -rf $O/kv-72b; ../07-combined/memguard.sh $tag $O/memguard.log & g=$!
  wrap=(); [ "${NSYS:-0}" = 1 ] && { wrap=(../../lib/obs/run_nsys.sh $O/$tag-nsys --); export NSYS_CAPTURE=cudaProfilerApi; extra+=(--nsys-phase "${NSYS_PHASE:-cold_fill}" --nsys-steps "${NSYS_STEPS:-40}"); mkdir -p $O/$tag-nsys; }
  log "start $tag"
  for util in ${UTILS:-0.9 0.85}; do
    "${wrap[@]}" python run_obs.py --run-dir $O/$tag --model $MODEL --prompt-source ${SRC:-longbench} --bailian-offset ${BOFF:-0} --n-docs ${NDOCS:-32} --decode-tokens ${DECODE:-8} --max-model-len ${MAXLEN:-12288} \
      --host-ram-fraction $f --pure --poll-sleep-ms 0 --gpu-util $util --kv-transport $kvt --settle-sec 15 --final-settle-sec 15 \
      --profile-out $O/$tag-profile.json --ssd-root $O/ssd-72b --kv-root $O/kv-72b "${extra[@]}" > $O/$tag.log 2>&1
    rc=$?
    [ -f $O/$tag/result.json ] && break
    if grep -aq "OutOfMemoryError" $O/$tag.log; then log "OOM $tag (gpu_util $util) → 재시도"; rm -rf $O/$tag $O/kv-72b; continue; fi
    break
  done
  kill $g 2>/dev/null; wait $g 2>/dev/null; rm -rf $O/kv-72b
  [ -f $O/$tag/result.json ] && log "done  $tag rc=$rc (gpu_util $util)" || { log "FAILED $tag rc=$rc"; grep -aE 'Traceback|Error|KILL' $O/$tag.log $O/memguard.log | tail -2 >> $O/campaign.log; }
done; done
log "== qwen72b 종료"
