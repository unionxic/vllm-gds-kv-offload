---
name: compare-results
description: 새 실험 결과가 생겼을 때 이전 결과와 forward 고정비 모형에 먼저 대조하고, 어긋난 런을 설명한 뒤에야 원인 가설을 세우는 절차. 결과 보고·원인 분석·문서 반영 전에 반드시 거침. 정합성 검사와 캠페인 실행 규칙 포함.
---

### 언제

- 캠페인이 끝나 results/ 아래에 결과가 새로 생겼을 때(json 또는 run_obs의 result.json 폴더)
- 사용자가 "왜 느리냐", "원인이 뭐냐", "지난번과 다르다"를 물을 때
- README나 docs/detailed-log.md에 새 수치를 적기 전
- backend·스케줄러·오프로더 코드를 바꾼 뒤 첫 런

### 절차

1. 표부터 본다. 저장소 루트에서 실행.

```
python tools/compare_results.py --ref <같은 구성의 기준 런 tag>
python tools/compare_results.py --ref pure-ram0.5-none results/native-66b/*/result.json   # run_obs 런
```

   기준 런은 같은 host 비율, 상주 layer, GPU KV(1 GiB 단위), 프롬프트 수의 재계산(kv none) 런. 구성이 다르면 변화율이 붙지 않는다. 형식은 06·07·09·10 러너 json과 11 run_obs(result.json + steps.jsonl) 모두 지원. 새 러너가 형식을 바꾸면 extract()에 분기를 추가한 뒤에 보고한다.

2. 고정비를 먼저 놓는다. 가중치 스트리밍 조건에서 forward 하나는 가중치 이동이고 모형은 CPU 티어/12.3 GB/s + SSD 티어/3.44 GB/s(layer가 0.5 GiB 미만이면 2.9). 표의 model 열과 fwd_s 열이 15% 안이면 forward 자체는 정상이고, 손익은 forward 개수와 prefill 토큰 몫에서 찾는다. 벗어나면(!) 그 런을 먼저 설명한다. 알려진 이탈은 cuFile 1 MiB I/O 느린 모드(2~2.5배), POSIX 가중치 경로(모형 대상 아님), 06의 host 0.1 런(재현 안 됨).

3. 기준 대비 변화(*)를 forward 개수, prefill forward 길이, KV 읽기·쓰기 GiB, 배치당 로드 대기로 나눠 어느 항이 움직였는지 적는다. wall clock 차이 하나로 결론 내지 않는다. 지금까지 확인된 손해 자리는 셋. 저장 단계의 KV 쓰기와 가중치 SSD 읽기의 디스크 공유(prefill 직후 decode forward 하나가 늘어남), 게이트 없는 적중 단계의 forward 개수 증가, 이중 버퍼가 없을 때 prefill 계산이 전송 위에 얹히는 것.

3-1. 손익 표는 한 사이클 합계가 기본. 저장 라운드(쓰기 비용) + 적중 라운드(읽기 이득)를 재계산 같은 라운드 수와 비교한 순이익 열을 반드시 두고, 적중 라운드만의 변화율은 "적중 라운드"라고 열 이름에 적는다. 순이익이 host 비율이나 재사용 횟수에 따라 갈리면 손익분기 조건까지 적는다.

3-2. 설정 차이를 표에 적는다. 기본값에서 손댄 항목(게이트, cuFile I/O 크기, KV 예산, 상주 layer, gpu_util, 폴링 양보)을 런마다 명시한다. 기본값 런과 손본 런의 차이가 결과의 일부다(66B RAM 0.5에서 20%).

4. 분해로 설명되지 않는 몫이 남을 때만 경합, 스케줄링 같은 가설을 세우고, 가설마다 직접 측정(step 단위 기록, KV IO 타임라인, nvme 1초 샘플, nvidia-fs 카운터, nsys)으로 확인한 뒤에 원인이라고 쓴다. 측정 없이 "경합 때문"이라고 쓰지 않는다.

5. 지난 결과와 어긋나는 점은 그 자리에서 기록한다. detailed-log 해당 절에 한 줄, 미확립이면 미확립이라고. 나중에 몰아서 정리하지 않는다.

### 정합성 검사

- backend나 커넥터를 바꾼 뒤에는 같은 입력으로 재계산 런과 적중 런의 출력 토큰열(requests.jsonl의 ids)이 전부 같은지 먼저 확인한다. 다르면 성능 수치는 보고하지 않는다. 단 Qwen + TRITON_ATTN + chunked prefill에서는 재계산 런끼리도 GPU KV 예산에 따라 토큰열이 갈리므로(부동소수점 축약 순서) 이 검사가 성립하지 않는다. 그때는 재계산 런 두 개의 불일치 수를 먼저 재고 적중 런의 불일치가 그 안이면 통과로 본다.
- native backend는 result.json의 kv_io.errors가 0이어야 한다. registered_tensors로 등록 직접 DMA였는지 bounce였는지 적는다(이 카드는 항상 0, bounce).
- nvidia-fs 통계(rw_stats_enabled=1)의 읽기·쓰기 MiB가 KV IO 합계와 맞는지 tier_samples.jsonl로 대조한다.

### 캠페인 실행

- 새 실험은 experiments/11-observability/run_obs.py와 lib/obs로 돌려 같은 산출물 형식(environment.txt, capacity.json, events.jsonl, requests.jsonl, steps.jsonl, tier_samples.jsonl, host 지표, result.json, summary.csv)을 남긴다.
- 캠페인은 setsid nohup으로 띄우고 성공 판정은 result.json 존재로 한다. OOM이면 gpu_util을 한 단계 낮춰 재시도하고 그 사실을 campaign.log와 결과에 남긴다.
- 백그라운드 런을 멈출 때는 kill 명령을 별도 호출로 하고, 같은 명령줄에 대상 이름을 쓰지 않는다(pgrep 자기 매칭으로 셸이 죽는다). 대상은 대괄호 패턴('campaign_pure6[6]')이나 PID로만 지정한다.
- 66B 가중치 스트리밍은 VLLM_OFFLOAD_PIN_EXACT=1과 memguard가 항상 필요하다(없으면 RAM 0.85 이상에서 머신이 죽음). 이 둘은 정책이 아니라 안전장치로 취급한다.

### 산출

- 사용자 보고: 표 요약(기준 대비 변화, 모형 대비 이탈), 분해 결과, 설정 차이, 남은 미확립 항목
- 문서: README 측정 표에 수치 한 줄, detailed-log에 분해와 근거. 오류 수정 서사(실패 시도, 폴링 같은 것)는 문서에 남기지 않고 결과만 남긴다
- 새 러너가 json 형식을 바꾸면 tools/compare_results.py의 extract()에 분기를 추가
