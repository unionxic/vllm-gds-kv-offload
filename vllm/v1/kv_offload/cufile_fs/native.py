# SPDX-License-Identifier: Apache-2.0
"""csrc/kv_offload/cufile_fs.cpp 를 torch C++ 확장으로 빌드·로드한다(최초 1회, 이후 캐시).
빌드에는 GCC 9 이상과 libcufile이 필요하다. CXX 미지정 시 g++-11, g++-10, g++-9 순으로 찾는다."""
import os
import shutil

from vllm.logger import init_logger

logger = init_logger(__name__)
_mod = None


def load():
    global _mod
    if _mod is not None:
        return _mod
    from torch.utils.cpp_extension import load as _load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, "..", "..", "..", "..", "csrc", "kv_offload", "cufile_fs.cpp"))
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.environ.get("CXX"):
        for cxx in ("g++-11", "g++-10", "g++-9"):
            if shutil.which(cxx):
                os.environ["CXX"] = cxx
                os.environ.setdefault("CC", cxx.replace("g++", "gcc"))
                break
    logger.info("cufile_fs: building native extension from %s (CXX=%s)", src, os.environ.get("CXX", "default"))
    cflags = ["-std=c++17", "-O2"]; ldflags = [f"-L{os.path.join(cuda, 'lib64')}", "-lcufile", "-lcudart"]
    if os.path.exists(os.path.join(cuda, "lib64", "libnvToolsExt.so")):  # NVTX 구간(kv_store_file, kv_load_file, kv_writes_pause/resume)
        cflags.append("-DCUFILE_FS_NVTX"); ldflags.append("-lnvToolsExt")
    _mod = _load(
        name="vllm_cufile_fs_C",
        sources=[src],
        extra_include_paths=[os.path.join(cuda, "include")],
        extra_ldflags=ldflags,
        extra_cflags=cflags,
        with_cuda=True,
        verbose=False,
    )
    return _mod
