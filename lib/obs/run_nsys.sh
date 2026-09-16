#!/usr/bin/env bash
# nsys로 명령을 감싸고 끝나면 stats csv 4종. usage: run_nsys.sh RUN_DIR -- cmd args...
set -u; R="${1:?RUN_DIR}"; shift; [ "${1:-}" = "--" ] && shift
# nsys 2025(gds 트레이스 지원)가 홈에 풀려 있으면 그것을, 아니면 PATH의 nsys(CUDA 12.8 동봉 2024.6)를 씀
NSYS="${NSYS_BIN:-$(ls $HOME/nsight-systems-*/opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | sort | tail -1)}"; NSYS="${NSYS:-nsys}"
export TMPDIR="$R/nsys-tmp"; mkdir -p "$TMPDIR"
"$NSYS" --version >"$R/nsys_version.txt" 2>&1
TRACE=${NSYS_TRACE:-cuda,nvtx,osrt}; "$NSYS" profile --help 2>&1 | grep -q "'gds'" && TRACE=$TRACE,gds
# NSYS_CAPTURE=cudaProfilerApi 이면 프로그램이 cudaProfilerStart/Stop 한 구간만 기록(run_obs.py --nsys-phase). 구간이 여러 개면 repeat로 이어 붙임
CAP=""; [ "${NSYS_CAPTURE:-}" = "cudaProfilerApi" ] && CAP="--capture-range=cudaProfilerApi --capture-range-end=repeat"
# nsys 임시 파일이 디스크를 채우면 런이 멈추므로(72B에서 24 GB) 시작 전 여유 40 GB 확인
avail_gb=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9); [ "${avail_gb:-0}" -lt 40 ] && { echo "run_nsys: 디스크 여유 ${avail_gb} GB < 40 GB, nsys 없이 실행" >&2; exec "$@"; }
"$NSYS" profile --output "$R/timeline" --trace=$TRACE --sample=none --cpuctxsw=none --cuda-memory-usage=true --storage-metrics=true --force-overwrite=true $CAP "$@"; rc=$?
[ -s "$R/timeline.nsys-rep" ] && "$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,nvtx_pushpop_trace --format csv --output "$R/stats" "$R/timeline.nsys-rep" >/dev/null 2>&1
rm -rf "$TMPDIR"
exit $rc
