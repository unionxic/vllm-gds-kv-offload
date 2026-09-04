### Gate 0: expfs I/O 스택 감사

#### 프로세스/스레드 구조 (정적 확인, 소스 기준)

오프라인 하네스는 VLLM_ENABLE_V1_MULTIPROCESSING=0으로 돌므로 scheduler(EngineCore),
GPU worker, expfs IO 스레드가 전부 한 프로세스에 있다. 서빙 모드라면 EngineCore
프로세스(스케줄러 쪽 FilesystemManager)와 worker 프로세스(FilesystemWorker와 IO
스레드)로 갈린다. 실측 PID/TID 맵은 각 run의 meta.json audit_threads에 기록된다.

- read worker: DualQueueThreadPool의 load-priority 스레드 8개(expfs_l0..7)
- write worker: store-priority 스레드 8개(expfs_s0..7)
- 전용 completion/polling 스레드: 없음. 완료는 IO 스레드가 파이썬 deque에 append하고,
  EngineCore의 스케줄 스텝(엔진 busy loop, 파이썬)에서 get_finished()가 drain한다.
  즉 polling loop의 실체는 엔진 자체의 파이썬 루프다.
- queue 우선순위: load 큐 우선(load-priority 스레드가 load 먼저, store 스레드는 반대,
  서로의 큐로 fallback).

#### cuFile 호출 경로와 GIL

- libcufile 로드: phase1/gdslib.py가 ctypes.CDLL(절대경로, RTLD_GLOBAL)로 시스템
  /usr/local/cuda/.../libcufile.so.0만 단일 로드. PyDLL 아님. pip wheel libcufile은
  로드하지 않는다(이중 로드 segfault는 이전 단계에서 확인되어 회피).
- restype/argtypes만 지정, errcheck/callback 없음 → native 실행 중 파이썬 재진입 없음.
- 따라서 cuFileRead/Write/BatchIOGetStatus의 native 구간에서 GIL은 해제된다(CDLL 규약).
- 동기 경로: chunk당 open→cuFileHandleRegister→span별 동기 cuFileRead/Write→
  deregister→close. V2 b64 기준 chunk당 span 수 = 레이어 수(32) ÷ 병합 정도
  (연속 GPU 블록이면 텐서당 1개로 병합 — 실측은 run별 cufile_sync_calls로 기록).
- Batch API: cuFileBatchIO* 5심볼이 시스템 libcufile(1.13)에 존재(nm 확인).
  ABI는 /usr/local/cuda/include/cufile.h에서 추출해 w2_leval/cufile_batch.py로 바인딩.

#### "GIL 문제"의 실측 계획

추정이 아니라 다음 ns 카운터로 구분해 기록한다(scheduler.py):
queue_wait_read/write(스레드풀 enqueue→task 시작), python_prepare(파일 open/등록/
IOParams 구성), native_submit(BatchIOSubmit), native_wait(동기 cuFile 호출 또는
completion Event 대기), python_completion(검증·rename·해제), scheduler_report
(task 종료→get_finished drain 시점, job 단위 근사). NVTX: CUFILE_SUBMIT/CUFILE_WAIT/
READ_QUEUE_WAIT/WRITE_QUEUE_WAIT/STORE_DEFERRED_RELEASE.

주의: 동기 경로의 native_wait에는 얇은 파이썬 래퍼(open/register)가 포함된 근사값이다.
실측 수치는 Gate 1/2 run의 meta.json과 본 문서의 갱신분에 기록한다.

#### 기존 단계에서 이월되는 사실

- 미등록 cuFile write는 nvidia-fs 내부 바운스 경로(TRACE 분류), read는 direct.
- 직전 sched 실험: 순수 read-priority 제출 정책(S1 상당)은 축소 윈도에서 무효과였고,
  store를 요청 gap으로 미루는 deferral은 tail을 약 5배 개선(반복 5회 일관). 이는
  read/write 제출 순서가 아니라 foreground 실행과 store의 동시성 자체가 지배 변수라는
  신호로, W2의 S2(엄격 위상 분리)와 S4(slack-aware) 해석에 선행 증거가 된다.
