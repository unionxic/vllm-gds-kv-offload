# 06-weight-offload 인수인계 (2026-09-07 작성)

## 이 세션 복원
```
claude --resume 529f68a0-167c-4c2c-bb29-8fd362cb59ae
```
대화 전체가 그대로 복원된다(로컬 transcript 기반, 재부팅과 무관).
안 되면 새 claude를 띄우고 "06-weight-offload HANDOFF.md 읽고 이어가" 라고 하면 된다.
메모리(~/.claude/projects/-home-unionxic/memory/)에도 상태가 기록돼 있다.

## 현재 상태
- vLLM: `~/vllm` 브랜치 `weight-ssd-offload`, 커밋 2fbceeb103 — 가중치 3단(GPU → pinned CPU 30% → SSD) 오프로드 구현 완료.
  새 인자: offload_ssd_path / offload_host_fraction(0.3) / offload_ssd_transport(cufile|posix) / offload_ssd_io_threads.
  V1 러너(VLLM_USE_V2_MODEL_RUNNER=0) + enforce_eager 필수.
- QA: `./run_qa.sh` (opt-2.7b 4 arm) 토큰 일치 전부 PASS. ssd-cufile native GDS 검사만 FAIL(아래 원인).
- 원인: 9/4 서버 정비로 MOFED 패치 nvme 제거 → cuFile compat(POSIX) 폴백. 복구 절차가 이 문서의 핵심.
- OPT-66B 다운로드: ~/.cache/huggingface/hub/models--facebook--opt-66b (132 GB 중 진행 중). 재부팅 시 중단됨 → 아래 명령으로 재개(이어받기 됨).

## 진행 기록
- 2026-09-07 17:20 MOFED 23.10 재설치 완료(mlnx-ofed-kernel-modules·mlnx-nvme-modules 5.15.0-97 ii, mlnx-en 24.10 제거됨).
  ucx-cuda만 실패(libnvidia-compute-535 의존, 드라이버 570이라 정상) → `sudo dpkg -P ucx-cuda`로 정리.
  /lib/modules/5.15.0-97-generic/updates/host/nvme-core.ko depends=mlx_compat 확인. 남은 것 = initramfs 재생성 + 재부팅.

## 재부팅 후 체크리스트 (순서대로)
1. GDS 스택 확인
   ```
   modinfo nvme_core | grep depends                     # mlx_compat 있어야 함
   /usr/local/cuda/gds/tools/gdscheck -p | grep 'NVMe '  # Supported 여야 함
   cat /proc/driver/nvidia-fs/stats | head -12
   ```
   실패면: MOFED 재설치가 안 된 것. `~/experiments/MLNX_OFED_LINUX-23.10-7.1.8.0-ubuntu20.04-x86_64/`에서
   `sudo ./mlnxofedinstall --with-nvmf --add-kernel-support --without-fw-update --force` → `sudo update-initramfs -u -k 5.15.0-97-generic` → 재부팅.
2. (선택) nvidia-fs IO 카운터 켜기: `echo 1 | sudo tee /sys/module/nvidia_fs/parameters/rw_stats_enabled`
3. OPT-66B 다운로드 재개 (tmux 안에서):
   ```
   source ~/experiments/vllm-gds-kv/env.sh
   hf download facebook/opt-66b --include "*.bin" --include "*.json" --include "*.txt"
   ```
4. GDS 복구 검증: `./run_qa.sh ssd-cufile` → "native GDS path" PASS(INTERNAL-BOUNCE 또는 DIRECT)여야 함.
5. 다운로드 완료 후 OPT-66B 실험:
   - 러너 V1, prefetch group 64 / num_in_group 61(GPU 상주 3층), offload_host_fraction 0.3 → CPU 19층, SSD 42층
   - arm: ssd-posix vs ssd-cufile, host_fraction 스윕(0.1/0.3/0.5), prefetch_step 1/2
   - 성능 런과 nsys 런 분리(`./run_qa.sh --nsys ...` 패턴 재사용)

## 함정
- 비밀번호 5회 실패 = 영구 잠금(pam_tally2). sudo는 신중히.
- 다른 사용자(susoon)가 콘솔에서 재부팅/정비하는 일이 잦음. 실험 전 `last -x reboot | head`로 확인.
- ~/vllm 커밋 77c4033078이 requirements/의 torch 라인을 지워 놓음(sed 사고). 런타임 무관, upstream diff 만들 때 주의.
