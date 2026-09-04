# source this: vllm-gds-kv 실험 공통 환경
export PATH=$HOME/miniconda3/envs/gdsllm/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
# Ubuntu 20.04 시스템 libstdc++가 conda libicui18n(CXXABI_1.3.15 요구)보다 낡아
# vllm import가 죽음 → conda libstdc++ 선로드 필수
export LD_PRELOAD=$HOME/miniconda3/envs/gdsllm/lib/libstdc++.so.6
# fs 티어 파일명(콘텐츠 해시)의 프로세스 간 일치를 위해 고정
export PYTHONHASHSEED=0
# 재사용 모듈은 lib/, 공유 하네스는 harness/. spec_module_path="expfs" 등
# import 이름 로딩이 이 경로에 의존.
_GDSKV_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$_GDSKV_ROOT/lib:$_GDSKV_ROOT/harness:$PYTHONPATH"

nvfs_stats() { grep -E '^(Reads|Writes)' /proc/driver/nvidia-fs/stats; }
