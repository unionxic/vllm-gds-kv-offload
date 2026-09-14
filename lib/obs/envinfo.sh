#!/usr/bin/env bash
# 런 폴더에 environment.txt. usage: envinfo.sh RUN_DIR [INPUT_FILE...]
set -u; R="${1:?RUN_DIR}"; shift; REPO="$(cd "$(dirname "$0")/../.." && pwd)"
{
  date --iso-8601=seconds; echo "host=$(hostname) kernel=$(uname -r)"
  echo "repo=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$REPO" status --short 2>/dev/null | wc -l)"
  echo "vllm_fork=$(git -C "$HOME/vllm" rev-parse HEAD 2>/dev/null) branch=$(git -C "$HOME/vllm" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id --format=csv,noheader
  echo "cuda=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1) nsys=$(nsys --version 2>/dev/null | tail -1)"
  echo "nvidia_fs=$(modinfo nvidia_fs 2>/dev/null | awk '/^version/{print $2}') srcversion=$(modinfo nvidia_fs 2>/dev/null | awk '/^srcversion/{print $2}')"
  echo "cufile_json=${CUFILE_ENV_PATH_JSON:-default}"
  env | grep -E '^VLLM_|^PYTORCH_CUDA' | sort | tr '\n' ' '; echo
  for f in "$@"; do [ -f "$f" ] && echo "input=$f sha256=$(sha256sum "$f" | cut -c1-16) bytes=$(stat -c %s "$f")"; done
  findmnt -T "$R" -o TARGET,SOURCE,FSTYPE,OPTIONS -n 2>/dev/null
  df -h "$R" | tail -1
} >"$R/environment.txt" 2>&1
