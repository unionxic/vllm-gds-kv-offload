# source this: vllm-gds-kv 실험 공통 환경
# 파이썬 환경은 호스트마다 다름. rain은 conda gdsllm, conda가 없는 호스트(sunny)는 venv.
_GDSKV_CONDA=$HOME/miniconda3/envs/gdsllm
if [ -d "$_GDSKV_CONDA" ]; then
  export PATH=$_GDSKV_CONDA/bin:/usr/local/cuda/bin:$PATH
  export CUDA_HOME=/usr/local/cuda
  # Ubuntu 20.04 시스템 libstdc++가 conda libicui18n(CXXABI_1.3.15 요구)보다 낡아
  # vllm import가 죽음 → conda libstdc++ 선로드 필수
  export LD_PRELOAD=$_GDSKV_CONDA/lib/libstdc++.so.6
elif [ -d "$HOME/.venvs/gdsllm" ]; then
  # conda 없는 호스트. venv를 활성화하고, 시스템 libstdc++로 충분하므로 선로드 없음.
  # CUDA 툴킷도 없으면 CUDA_HOME을 두지 않음(torch 휠이 런타임을 들고 옴).
  . "$HOME/.venvs/gdsllm/bin/activate"
  [ -d /usr/local/cuda ] && { export PATH=/usr/local/cuda/bin:$PATH; export CUDA_HOME=/usr/local/cuda; }
else
  echo "env.sh: gdsllm 파이썬 환경을 찾지 못함(conda envs/gdsllm 또는 ~/.venvs/gdsllm)" >&2
fi
# fs 티어 파일명(콘텐츠 해시)의 프로세스 간 일치를 위해 고정
export PYTHONHASHSEED=0
# 재사용 모듈은 lib/, 공유 하네스는 harness/. spec_module_path="expfs" 등
# import 이름 로딩이 이 경로에 의존.
_GDSKV_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$_GDSKV_ROOT/lib:$_GDSKV_ROOT/harness:${PYTHONPATH:-}"

nvfs_stats() { grep -E '^(Reads|Writes)' /proc/driver/nvidia-fs/stats; }
