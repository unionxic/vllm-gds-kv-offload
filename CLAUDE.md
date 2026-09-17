### 결과 해석

- 새 결과가 생기면 먼저 `python tools/compare_results.py --ref <기준 런>`으로 이전 결과와 forward 고정비 모형에 대조. 절차는 .claude/skills/compare-results/SKILL.md.
- 병목은 고정비 분해에서 시작. 가중치 스트리밍에서 forward는 CPU 티어/12.3 GB/s + SSD 티어/3.44 GB/s(작은 layer는 2.9). 분해로 설명되지 않는 몫이 있을 때만 경합·스케줄링을 가설로 세우고, 직접 측정 없이 원인을 확정하지 않음.
- 손익은 한 사이클 합계로 적음. KV 오프로드는 저장 라운드 손해 + 적중 라운드 이득을 재계산 두 라운드와 비교한 순이익이 기준이고, 적중 라운드만의 변화율을 wall clock 변화처럼 쓰지 않음. 부분 구간 수치를 낼 때는 어느 구간인지 열 이름에 명시.
- backend·커넥터를 바꾸면 성능보다 먼저 출력 토큰열이 재계산과 같은지 확인. 기본값에서 손댄 설정은 런마다 명시.
- 지난 결과와 어긋나는 점은 발견 즉시 docs/detailed-log.md에 기록. 미확립이면 미확립이라고 씀. 오류 수정 서사는 남기지 않고 결과만 남김.

### 실험 실행

- 새 실험은 experiments/11-observability/run_obs.py + lib/obs로 돌려 같은 산출물 형식을 남김. 이전 세대(OPT, expfs, 01~10)는 git 태그 archive/opt-era-2026-09-18에만 있음. KV 전송은 포크 안 CuFileFsSpec(C++), 외부 파이썬 전송 코드를 쓰지 않음.
- 캠페인은 setsid nohup으로, 성공 판정은 result.json 존재로. nsys 런의 분석은 result.json이 아니라 캠페인 로그의 done 줄 또는 <run>-nsys/nsys.done 표식 뒤에만 시작(리포트 후처리 중 건드리면 유실). 분석 스크립트는 다른 프로세스가 쓰는 폴더(nsys-tmp 등)를 절대 지우지 않고 자기 mktemp 폴더만 쓴다. 백그라운드 런을 멈출 때는 kill을 별도 호출로 하고 같은 명령줄에 대상 이름을 쓰지 않음.
- 72B는 VLLM_OFFLOAD_PIN_EXACT=1과 lib/obs/memguard.sh 필수(안전장치). 캠페인은 experiments/11-observability/campaign_qwen72.sh.

### 문서와 용어

- md 문서: 볼드·절번호·날짜 금지, 대주제 h3·소주제 h4, 개조식 명사 종결. 요약은 README, 상세는 docs/detailed-log.md. 실험 폴더에 보고서를 따로 두지 않음.
- 설명용 조어를 만들지 않음. 풀어서 쓰거나 영어 용어 그대로(wall clock은 wall clock). 실험 조건은 "조건"으로 부름.
