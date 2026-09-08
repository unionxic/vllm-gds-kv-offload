# 07-combined: 가중치 3단 오프로드 + KV→SSD(expfs) 결합 — OPT-66B (2026-09-08)

## 1. 질문
주류 서빙 배치("가중치는 GPU 최대 + host fraction, 넘치면 SSD; KV는 활성 배치 몫만 GPU, 나머지는 SSD")에서
**KV를 SSD에서 가져오는 경로**(expfs cuFile vs POSIX bounce)의 비용/이득과, 가중치 스트리밍과의 디스크 간섭을 측정.

## 2. 구성
- 모델 OPT-66B fp16, GPU 상주 4층(group 64 / num_in_group 60), 가중치 prefetch 오프로더(step1, thr4, cuFile), KV expfs(block 64, 스레드 4).
- GPU KV 1.5 GiB(=480tok 프롬프트 1개 분량 → **8프롬프트가 batch=1로 순차 실행**), max_model_len 640, 프롬프트 8×(448 prefix+32 tail), decode 8tok.
- 라운드: r1 = cold(첫 generate가 KV 저장, 두 번째 generate가 저장분 적중) / r2 = 전부 SSD KV 적중. `run_combo_66b.py`, `campaign07.sh`, 표 `summarize_07.py`.
- 레짐 1: host 0.85 × KV {cufile, posix, none} / 레짐 2: host 0.3 × KV {cufile, posix}.

## 3. 결과 (results/combined/opt66b/*.json)
| arm | load | host / SSD 층 | r1 prefill | r1 decode-step | r2 prefill | r2 decode-step | r2 KV SSD 읽기 | 가중치 SSD 읽기 | avail(끝) |
|---|---|---|---|---|---|---|---|---|---|
| h0.85-kvnone | 205s | 106.3GiB/56L, 7.6GiB/4L | 108.2s | 96.8s | 108.0s | 96.9s | 0 | 1,109GiB | 12.3GiB |
| h0.85-kvcufile | 226s | 〃 | 113.4s | 95.8s | **100.0s** | 97.3s | 112회/15.8GiB | 1,109GiB | 12.6GiB |
| h0.85-kvposix | 226s | 〃 | 115.7s | 96.1s | 104.7s | 97.5s | 112회/15.8GiB | 1,109GiB | 10.6GiB |
| h0.3-kvcufile | 480s | 36.1GiB/19L, 77.9GiB/41L | 240.0s | 223.7s | **228.0s** | 225.3s | 112회/15.8GiB | 11,364GiB | 83.5GiB |
| h0.3-kvposix | 480s | 〃 | 248.0s | 227.9s | 236.2s | 229.4s | 112회/15.8GiB | 11,364GiB | 81.5GiB |

전 런 라운드 간 출력 토큰 동일. "decode-step"은 8프롬프트 순차 합계(= forward 8회): h0.85 forward ≈12s(host 106GiB H2D ≈8.6s @12.3GB/s + SSD 7.6GiB ≈2.6s), h0.3 ≈28s(06 실측 28.5s와 일치).

## 4. 판정
1. **SSD KV 적중은 재계산보다 빠르다**: h0.85 r2 prefill — 재계산(kvnone) 108.0s vs cuFile 100.0s(−8.0s) vs POSIX 104.7s(−3.3s). KV 저장 비용(r1)은 cuFile +5.2s, POSIX +7.5s.
2. **cuFile > POSIX**: KV 15.8GiB 읽기에서 cuFile이 4.7s(h0.85)·8.2s(h0.3) 빠름. 가중치 스트리밍이 SSD를 포화시키는 h0.3에서도 이득이 유지·확대됨(= 디스크 간섭 아래에서 bounce 직렬화가 더 손해).
3. **decode에는 무관**: 전 arm decode-step 동일(KV 경로는 prefill 적중에만 개입).
4. **비중**: 이 배치에서 prefill 시간의 대부분은 가중치 전송(PCIe/SSD)이라 KV 경로 차이는 총 시간의 4~8%. 가중치가 GPU에 다 올라가는 모델(주류 서빙)에서는 같은 절대값(수 초/16GiB)이 훨씬 큰 비율이 된다.

## 5. 함정과 수정 (이 캠페인에서 발견)
- **torch pinned 할당 2^n 올림**: CachingHostAllocator(`ATen/core/CachingHostAllocator.h:302` PowerOf2Ceil)가 OPT-66B 층 1.9→2.75GiB(1.45배; opt-2.7b 실측 1.405배). 06의 "오버헤드 16GiB"의 정체이며, 07 첫 시도(h0.92)에서 **머신 크래시**, 0.85~0.70 전부 워치독 KILL의 원인. pinned는 /dev/zero MAP_SHARED라 meminfo **Shmem**으로 잡힘.
  수정: `VLLM_OFFLOAD_PIN_EXACT=1` → `prefetch.py::_pinned_exact()`가 pageable 텐서를 `cudaHostRegister`(정확 크기)로 고정. 검증 `pin_exact_test.py`: Shmem 증가 5.77→0.08GiB, 토큰 동일, H2D 대역폭 동일(12.3GB/s, 07-combined-0c 세션 마이크로벤치). 함정: 로더가 `torch.device("cuda")` 컨텍스트라 `device="cpu"` 명시 필수; `Tensor.is_pinned()`는 스토리지 시작 주소로 판정하므로 정렬 오프셋이 아닌 스토리지 시작에서 등록.
- **campaign 스크립트 성공 판정**: `grep|cut` 파이프 종료코드(cut=0)로 판정해 실패 후 후퇴 사다리가 작동하지 않았음 → json 존재로 판정, `set -o pipefail`.
- **워치독 memguard.sh**: 1초 샘플링, MemAvailable<2GiB 또는 (avail<4GiB && PSI full>30%)면 프로세스만 kill. PSI 단독 10% 기준은 체크포인트 읽기 페이지캐시 회수로 오탐.
- 실행은 `setsid nohup`(tmux new-window는 사용자 화면을 바꿔 Ctrl-C 사고 유발).
- 0.85 정상상태: pinned 106.3GiB + RSS 오버헤드 ≈1.2GiB, avail 12.6GiB. 0.92는 올림 없이도 여유 ≈3GiB로 비권장.

## 6. 미해명·후속
- `matched` 카운터는 스케줄러 질의 누적이라 arm 간 값이 달라도 의미 없음(kv_read_n/bytes로 판단).
- h0.85 SSD 4층이 남아 "디스크=KV 전용" 레짐은 근사치. GPU 5층 상주 또는 host 0.88 시도 여지(여유 12.6GiB 중).
- 후속: 08-cufile-bounce(bounce 조각 크기), KV 블록 코얼레싱, batch>1 구성(kv 3GiB)에서 재측정.
