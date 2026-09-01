### scheduling 레짐 재현 실험 계획

목표는 튜닝이 아니라 판정이다. V2-D-b64에서 한 번 관찰된 빠른 레짐을 만드는 조건이 존재하는지, 명시적 설정으로 반복 재현되는지, 그것이 cuFile 고유인지 expfs 전체의 개선인지(동일 조건 E paired control) 분리한다. 재현되지 않으면 기존 판정(GDS 우위 미확립)을 유지한다.

#### 사전 분석에서 확정한 사실

발산은 요청 0부터 전 구간 균일하다. 원본 600요청을 50요청 창으로 잘라 보면 모든 창에서 slow의 p95가 fast의 1.9~3.0배이고 첫 창(0~49)부터 그렇다. 레짐은 런 시작 시점에 결정되며 진행 중 전이가 아니다. storage hit 인덱스는 fast와 slow에서 147/147 동일해 워크로드 자체는 결정적이다.

#### 축소 trace 선정

앞 150요청을 그대로 쓴다(순서 보존, 별도 warmup 불필요, hit와 miss 공존, store와 foreground 겹침). 창 기준값은 원본 run의 해당 창을 재집계해 정한다.

| 창 0~149 기준 | fast(1차) | slow(재측정/재검증) |
|---|---|---|
| p95 | 약 2.0 | 4.4~5.3 |

빠른 레짐 판정: 창 p95가 2.5 이하. 느린 레짐: 4.0 이상. 그 사이는 미확정으로 기록.

#### 기록된 환경 차이 (스윕 전 최소 재현 대상)

1차 fast 런은 /dev/shm이 이전 크래시 누출로 63GB 만석(가용 RAM 약 65GB)인 상태에서 돌았고, 모든 slow 런은 shm 청정(가용 약 128GB)이었다. 로그 타임라인으로 확정했다(누출 파일들의 mtime이 전부 fast 런 이전, 직전 v2-C가 그 만석 때문에 EFAULT로 사망). 그 외 차이는 replay.py의 관찰 훅 추가(요청당 비용 미미)와 vLLM shm 패치(expfs는 해당 모듈 미사용)다.

최소 재현: n=150에서 D/E × shm 60GiB 채움/비움 4런(r01~r04). 채움은 전용 파일 sched_fill.bin으로 하고 그 파일만 제거한다.

#### 실험 순서

1. shm-fill 최소 재현 (위)
2. switch interval sweep: 0.1/0.25/0.5/1/2/5(기본)/10 ms, 각 1회 screening, D와 E paired, 실행 순서 균형(D→E와 E→D 교차). 유망 후보(기본 대비 D p95 15% 이상 개선 + 처리량 10% 이상 + 반복 2회 이상 같은 레짐)만 3회 반복.
3. write worker 수: 1/2/4/8/16 (read 8 고정), D/E paired.
4. store overlap/backpressure: load-only, drain 후 load, deferred store, concurrent(현행), concurrent+제한.
5. affinity: 앞 단계에서 유망 설정이 나온 경우에만.
6. 유망 설정 반복 검증(축소 5+5회, 순서 교차) 후에만 전체 600요청 진출.

nsys와 perf는 일반 실행으로 후보를 재현한 뒤 별도 diagnostic run에서만 쓰고, 그 수치는 성능 표에 섞지 않는다.

#### 판정 규칙 요약

단일 최솟값은 성공이 아니다. 모든 run은 run ID로 보존한다(runs/ 아래 raw.csv와 meta.json). 실패나 느린 실행도 manifest에 남긴다. 변수는 한 번에 하나만 바꾼다. D 단독 개선이면 cuFile과 스케줄링의 상호작용, D/E 동반 개선이면 expfs control plane 문제, 무개선이면 해당 가설 기각.

#### run manifest

| run_id | design | 조건 | 목적 |
|---|---|---|---|
| r01-D-empty | D | shm 청정 | slow 기준선 |
| r02-D-shmfill | D | shm 60GiB 채움 | 1차 fast 조건 재현 |
| r03-E-shmfill | E | shm 60GiB 채움 | paired control |
| r04-E-empty | E | shm 청정 | paired control |

중간 판정: r01(D 청정) p95 4.867, r02(D 60GiB 채움) p95 5.090, r03(E 채움) p95 5.435 —
shm 만석 가설 기각. 기록된 환경 차이는 빠른 레짐의 원인이 아니다.

#### 문헌 기반 스케줄링 정책 실험 (step 1)

가설 출처: Tutti(foreground read와 background store 동시 실행이 tail을 악화),
GIDS(고정 worker 수보다 outstanding IO 제어가 중요), cuFile Batch API(다중 span의
호출 수 축소). Batch API는 시스템 libcufile에 심볼 존재 확인됨(cuFileBatchIOSetUp 등 5종).

정책 4종을 policies.py로 주입(phase2/expfs.py 무수정): baseline / read_priority
(load 대기 중 store 제출 보류) / deferred_store(요청 gap에서만 store 제출·완료 대기)
/ phase_sep(load와 store task 상호배제). 공통 안전장치: 커넥터의 wait(job_ids)
fence 시 해당 보류 job 강제 제출(forced로 계수) — GPU 블록 정합성 유지.

| run 그룹 | 구성 |
|---|---|
| s1-baseline-r1..5-{D,E} | 현행 동시 실행, 반복 변동계수 산출용 |
| s1-read_priority-r1..5-{D,E} | load 우선 |
| s1-deferred_store-r1..5-{D,E} | gap 전용 store (배치 B) |
| s1-phase_sep-r1..5-{D,E} | 상호배제 (배치 B) |

실행 순서는 rep 홀수 D→E, 짝수 E→D로 교차. 계측: p50/p95/p99, wall, cpu,
max outstanding store/load, forced/gap flush 수. 판정은 반복 통계(평균·CV·레짐
분류 FAST≤2.5 / SLOW≥4.0)와 D/E paired 비교로만 한다.
