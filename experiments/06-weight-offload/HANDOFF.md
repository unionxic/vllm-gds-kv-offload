# 06-weight-offload 인수인계 (2026-09-07 작성)

## 이 세션 복원
```
claude --resume 529f68a0-167c-4c2c-bb29-8fd362cb59ae
```
대화 전체가 그대로 복원된다(로컬 transcript 기반, 재부팅과 무관).
안 되면 새 claude를 띄우고 "06-weight-offload HANDOFF.md 읽고 이어가" 라고 하면 된다.
메모리(~/.claude/projects/-home-unionxic/memory/)에도 상태가 기록돼 있다.

## 현재 상태 (2026-09-08 00:05 갱신)
- vLLM: `~/vllm` 브랜치 `weight-ssd-offload`, 커밋 3fc4433b62(경계 슬롯 충돌 수정 포함) — 가중치 3단(GPU → pinned CPU 30% → SSD) 오프로드 + 등록 GPU ring 모드(`offload_ssd_ring_mb`).
  새 인자: offload_ssd_path / offload_host_fraction(0.3) / offload_ssd_transport(cufile|posix) / offload_ssd_io_threads / offload_ssd_ring_mb.
  V1 러너(VLLM_USE_V2_MODEL_RUNNER=0) + enforce_eager 필수.
- GDS: 9/7 MOFED 재설치·재부팅으로 복구 완료. QA(`./run_qa.sh`, opt-2.7b) 4 arm 토큰 일치 + cufile DIRECT. ring QA(`smoke_ring.py 16 1 4`) PASS.
- OPT-66B: 다운로드 완료(~/.cache/huggingface/hub/models--facebook--opt-66b).
- **캠페인 실행 중**: `./relaunch.sh`(step1 회귀 QA → `./campaign.sh`) → phase A·B 완료, C(ring8+step2+thr8, gpu_util 0.75)부터 재개, 이후 D·E·F·ring16 r4.
- 2026-09-08 01:45 발견: upstream prefetch 버그(모듈 수 61이 step 2로 안 나뉘면 패스 경계에서 슬롯 충돌 → garbage). vllm 3fc4433b62로 수정, `repro_wrap.sh`(31모듈×step2)로 전/후 검증. 상세는 메모리 vllm-gds-kv-experiment.md. 로그 `results/weight-offload/opt66b/campaign.log`, 런당 14~20분.
  - 죽었으면 재개: `cd experiments/06-weight-offload && nohup ./campaign.sh > ../../results/weight-offload/opt66b/campaign.log 2>&1 &` (기존 json은 건너뜀).
  - 표: `python3 summarize_66b.py`
- phase A 결과(h0.3, step1, thr4, 3반복 중앙값): cuFile decode step 28.5s / prefill 31.1s, POSIX 68.0s / 72.0s → **cuFile 2.4배**. 둘 다 SSD 읽기 877GiB, 토큰 동일.

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
3. (완료됨, 참고용) OPT-66B 다운로드 재개 (tmux 안에서):
   ```
   source ~/experiments/vllm-gds-kv/env.sh
   hf download facebook/opt-66b --include "*.bin" --include "*.json" --include "*.txt"
   ```
4. GDS 복구 검증: `./run_qa.sh ssd-cufile` → "native GDS path" PASS(INTERNAL-BOUNCE 또는 DIRECT)여야 함.
5. OPT-66B 실험 (캠페인이 자동화함, 위 '현재 상태' 참조):
   - 러너 V1, prefetch group 64 / num_in_group 61(GPU 상주 3층), offload_host_fraction 0.3 → CPU 19층, SSD 42층
   - arm: ssd-posix vs ssd-cufile, host_fraction 스윕(0.1/0.3/0.5), prefetch_step 1/2
   - 성능 런과 nsys 런 분리(`./run_qa.sh --nsys ...` 패턴 재사용)

## 함정
- 비밀번호 5회 실패 = 영구 잠금(pam_tally2). sudo는 신중히.
- 다른 사용자(susoon)가 콘솔에서 재부팅/정비하는 일이 잦음. 실험 전 `last -x reboot | head`로 확인.
- ~/vllm 커밋 77c4033078이 requirements/의 torch 라인을 지워 놓음(sed 사고). 런타임 무관, upstream diff 만들 때 주의.
