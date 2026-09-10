#!/usr/bin/env bash
# 09 2단계: 1단계와 같은 배치 구성(상주 2층, GPU KV 사다리 결과, 16프롬프트)에서 KV 정책 비교.
#   대조군은 1단계의 b-kv<KV>-cufile(= baseline: 끝난 블록 전부 저장, 적중 시 동기 로드).
#   staging 계열(staged-*)의 대조군은 staged block. 전송 구현이 CuFileTransport→StagedCuFileTransport 로 바뀌므로
#   admission 효과를 보려면 같은 staged 안에서 block 과 비교해야 한다.
#   제출 스케줄링 계열(read_priority, deferred_store)은 비staged cufile 위에서 baseline 과 직접 비교한다.
#   01~05 기록: deferred_store 는 p95 5배·CPU 12배 이득, read_priority 는 이득 0. 가중치 스트리밍과 공존할 때도
#   같은 순서가 유지되는지가 이번 질문.
set -u; cd "$(dirname "$0")"; source ../../env.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_OFFLOAD_PIN_EXACT=1
O=../../results/kv-policy; mkdir -p $O
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ # tag, extra args...
  local tag=$1; shift
  [ -f $O/$tag.json ] && { log "skip $tag (exists)"; return 0; }
  rm -rf ../../results/weight-offload/ssd-66b ../../results/combined/kv-66b
  local free_gb=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); [ "$free_gb" -lt 90 ] && { log "disk free ${free_gb}G < 90G, abort"; exit 1; }
  log "== $tag ($*)"
  ../07-combined/memguard.sh $tag $O/memguard.log & local guard=$!
  python run_policy_66b.py --host-fraction 0.85 --num-in-group 62 --kv-cache-gib $KV \
    --n-prompts 16 --out-dir $O $SLOTS "$@" --tag $tag > $O/$tag.log 2>&1
  kill $guard 2>/dev/null; wait $guard 2>/dev/null
  [ -f $O/$tag.json ] && { grep -a '^RESULT' $O/$tag.log | cut -c1-500; return 0; }
  log "FAILED $tag"; grep -aE 'Error|Killed|out of memory|KILL|Traceback' $O/$tag.log $O/memguard.log | tail -3; return 1
}
while pgrep -f 'campaign09[c]\.s[h]|run_policy_66b\.p[y]' >/dev/null; do sleep 30; done
KV=$(ls $O/b-kv*-cufile.json 2>/dev/null | head -1 | sed 's/.*b-kv\(.*\)-cufile\.json/\1/')
[ -z "$KV" ] && { log "1단계 결과가 없어 2단계 중단"; exit 1; }
log "2단계 시작 (KV=${KV}GiB)"
# staging ring 슬롯: 기본 2로 돌린 첫 런(p-staged-cpufb-slots2)은 slot_full 247회, 저장 시도 286건 중 183건 버림 →
# 2라운드 적중이 186만에서 2.4만 토큰으로 붕괴. 청크가 141MiB라 슬롯 2개(283MiB)로는 도착률을 못 따라간다.
# 슬롯 6개(848MiB) + writer 4개로 올린다. 슬롯2 런의 gpu_max 13.02GiB이므로 +565MiB 해도 예산 14.4GiB 안.
SLOTS="--staging-slots 6 --staging-writers 4 --cpu-fallback-slots 8"
run p-staged-cpufb   --kv-transport cufile_staged --staging-policy cpu_fallback
run p-staged-skip    --kv-transport cufile_staged --staging-policy skip
run p-staged-block   --kv-transport cufile_staged --staging-policy block
run p-deferred-store --kv-transport cufile --policy deferred_store
run p-read-priority  --kv-transport cufile --policy read_priority
FIX=blocking_event run p-blocking-event --kv-transport cufile
log "09 2단계 완료"
