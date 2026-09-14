#!/usr/bin/env bash
# nsys로 명령을 감싸고 끝나면 stats csv 4종. usage: run_nsys.sh RUN_DIR -- cmd args...
set -u; R="${1:?RUN_DIR}"; shift; [ "${1:-}" = "--" ] && shift
NSYS="${NSYS_BIN:-nsys}"; export TMPDIR="$R/nsys-tmp"; mkdir -p "$TMPDIR"
"$NSYS" --version >"$R/nsys_version.txt" 2>&1
"$NSYS" profile --output "$R/timeline" --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-memory-usage=true --storage-metrics=true --force-overwrite=true "$@"; rc=$?
[ -s "$R/timeline.nsys-rep" ] && "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum --format csv --output "$R/stats" "$R/timeline.nsys-rep" >/dev/null 2>&1
exit $rc
