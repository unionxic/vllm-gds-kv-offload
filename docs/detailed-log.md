### 상세 기록

README의 요약을 뒷받침하는 설계 근거, 전체 측정표, 발견과 정정의 기록이다. 진행 순서대로 쓴다.

이 문서 본문은 구 디렉터리 이름(phase0~3, sched, w1, w2_leval, admission)으로 쓰였다. 저장소는 이후 대분류로 재구성됐다. 아래 대응표로 읽으면 된다.

| 구 경로 | 신 경로 |
| --- | --- |
| phase0 | experiments/01-feasibility/bringup |
| phase05 | experiments/01-feasibility/prefix-gate |
| phase1 | experiments/01-feasibility/cufile-microbench (gdslib·path_classify는 lib) |
| phase2 | experiments/01-feasibility/expfs-smoke (expfs.py는 lib) |
| phase3 | experiments/01-feasibility/abcde-matrix |
| w1 | experiments/02-bailian/replay600 |
| sched | experiments/02-bailian/window150 (run_bench.py는 harness, 공유 모듈은 lib, runs·prof는 results/bailian) |
| w2_leval | experiments/03-leval (scheduler·cufile_batch는 lib) |
| admission | experiments/04-admission (value_admission.py는 lib) |
| upstream_check | experiments/05-upstream/checks |
| results/w2_leval | results/leval |
| results/w2_openloop | results/leval-openloop |

### 연구 질문의 구조

질문은 셋으로 분리해 각각 따로 답한다. transport feasibility(SSD hit가 재계산보다 빠른가), workload legitimacy(실제 워크로드에서 filesystem hit가 자연 발생하는가), GDS benefit(SSD→CPU→GPU보다 SSD→GPU가 낫는가). 세 번째는 기존 경로 C와 GDS 경로 D의 직접 비교로만 답할 수 있다. C와 B(CPU hit)의 차이를 GDS 개선 상한으로 읽으면 안 되는데, 그 차이에는 NVMe read가 포함되고 그 비용은 GDS도 지불하기 때문이다.

워크로드 조사는 GDS 구현의 허가 조건이 아니다. production trace에 장기 재사용 KV가 존재하고, Mooncake가 그 cold tail을 위한 SSD 티어를 실제로 구현했으며, Tutti가 vLLM 위에서 SSD 기반 prefix KV와 GDS 계열을 직접 평가했고, vLLM 자체가 filesystem 티어와 persistent 공유를 공식 지원하며, RFC 48504가 worker 쪽 filesystem/GDS 백엔드를 제안하고 있다. 리플레이 워크로드는 허가 게이트가 아니라 구현된 경로를 평가할 대표 워크로드다.

Mooncake의 위치. Mooncake는 production 유래 워크로드에서 SSD 기반 KV 티어의 필요성을 제시하고 실제 SSD 오프로드를 구현했지만, SSD에서 GPU로의 load는 아직 DRAM을 경유한다. Mooncake가 후속 과제로 남긴 GDS 직행 경로를 이 실험에서 vLLM native OffloadingConnector 위에 구현해 손익을 검증한다. GDS 미구현은 결격 사유가 아니라 우리가 측정할 미구현 구간이다.

### 워크로드 정당성 조사

#### vLLM 오프로드의 의미론

v0.26의 OffloadingConnector는 요청 간 prefix 재사용(A 계열)이다. 공식 문서가 스스로를 prefix cache의 확장이라 정의하고, offload_prompt_only 기본값이 true라 기본은 prefill 블록만 저장한다. store는 매 스케줄 스텝에서 계산이 끝난 chunk를 증분 저장하고, load는 요청 admission 시의 prefix lookup에서만 일어난다. 디코드 중 활성 KV를 내렸다 올리는 경로는 없으므로 FlexGen류 active KV swapping(B 계열) 논문은 이 서브시스템의 근거가 될 수 없다. 재시작 후 재사용과 다중 인스턴스 공유는 공식 intended use이며 조건은 모든 인스턴스의 PYTHONHASHSEED 고정이다.

#### 후보 비교와 판정

근거 수준: 1 production trace와 실제 SSD 구현 / 2 production trace이나 스토리지 구조가 다름 / 3 공개 실데이터 시스템 평가 / 4 공식 합성 벤치 / 5 마이크로벤치나 인위적 압박.

| 후보 | 근거 | 계열 | 로컬 NVMe | vLLM fs 적합성 | 주요 한계 |
|---|---|---|---|---|---|
| Qwen-Bailian trace | 2 | A | 무관 | 16토큰 블록 해시가 vLLM과 1:1, 공식 replayer가 OpenAI API로 리플레이 | 원논문은 to-B에 대해 SSD 회의적 |
| Mooncake | 1 | A | 혼합 | FilePerKeyBackend가 vLLM fs와 구조 동일, GDS 도입이 명시된 후속 과제 | GDS 미구현, E2E 수치는 벤치 기반 |
| LMCache bench | 4 | A | 해당 없음 | LMCache 없이 vLLM에 리플레이 가능 | 합성, thrash 노브는 인위적 압박 |
| Tutti | 3 | A | 예 | LMCache-GDS baseline이 우리 설계와 같은 슬롯, 256GB DRAM에서도 SSD가 hit를 올림 | 도착이 Poisson 합성 |
| DualPath | 2 | A | 아니오(3FS) | 의미론 일치, 아키텍처 무관 | trace 미공개 |
| HyMCache | 3 | A | 아니오(CXL) | 접근 패턴 논제만 이식 | 파일시스템 아님 |
| HiFC | 3~4 | B | 예 | 부적합(preemption swap 서브시스템) | 데이터 경로 참고용 |
| DUAL-BLADE | 5 | B | 예 | 부적합 | 인위적 host 제한 |
| vLLM persistence 문서 | 공식 | A | 예 | 정의상 완벽 | 운영성 워크로드 |

production 타당성이 가장 높은 것은 Mooncake와 Bailian trace의 조합(Mooncake 블로그가 같은 trace를 분석하므로 사실상 한 몸)이고, 데이터 경로가 가장 가까운 선행은 Tutti의 LMCache-GDS baseline이며, 재현이 가장 쉬운 것은 재시작 warm-start다. 세 기준의 답이 다르므로 억지로 합치지 않는다. HiFC와 DUAL-BLADE는 계열이 달라 정당성으로 쓰지 않되 데이터 경로 증거(KV 텐서를 GDS 버퍼로 등록, 64토큰 블록이 유리, 동시성이 지연을 은폐하면 SSD 티어가 DRAM 티어와 e2e 동률)는 차용한다. Mooncake 분석에서 검증한 수치: coding trace에서 재사용 블록의 69.2%가 10분 내 재접근, 10.3%가 30분 후 cold 재사용.

대표 워크로드. W1은 Bailian coder trace 리플레이(도착 순서와 재사용 거리 보존, CPU 티어는 임의 배수가 아니라 실제 운영 예산), W2는 LEval/LooGLE 실데이터 long-doc(인위적 오버플로 노브 금지), W3는 재시작 warm-start.

### 설계

#### 활성화 경로와 구조적 제약

활성화 경로는 두 가지다. kv-offloading-size 플래그는 native CPU 오프로드를 자동 구성하는 편의 경로이고, 티어링이나 out-of-tree 스펙은 kv-transfer-config에서 connector와 spec_name을 직접 지정한다. 후자에는 편의 플래그가 필요 없다.

```bash
PYTHONHASHSEED=0 vllm serve <model> --kv-transfer-config '{
  "kv_connector": "OffloadingConnector", "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 10737418240, "block_size": 16,
    "secondary_tiers": [{"type": "fs", "root_dir": "/path/kv_cache"}]}}'
```

GDS를 2차 티어로 넣을 수 없는 이유가 셋 있다. 2차 티어는 스케줄러 프로세스에서 실행되어 CUDA 컨텍스트와 GPU 텐서가 없고, 티어가 받는 유일한 데이터 핸들은 /dev/shm mmap 위의 host memoryview이며, job의 블록 id는 CPU 슬롯 인덱스라 GPU 블록을 지칭할 방법이 없다. 삽입 지점은 한 단계 위인 OffloadingSpec이다. worker 쪽 OffloadingWorker는 GPU KV 텐서를 (블록 수, 페이지 바이트) 형태로 직접 받고, spec_module_path라는 공식 이음새로 vLLM 수정 없이 로드된다.

오프로드 단위는 chunk다. blocks_per_chunk개의 GPU 블록에 해당하는 전 레이어 데이터를 연결한 플랫 버퍼 하나가 파일 하나가 되고 페이지 크기로 정렬된다.

#### GPU 배치는 러너에 따라 갈린다

이 빌드에는 model runner가 두 벌 공존한다. 기본인 V2 러너는 cross-layer 배치가 미구현이라 게이트 조건이 전부 참이어도 레이어별 분산 텐서를 쓰고, VLLM_USE_V2_MODEL_RUNNER=0으로 고르는 V1 러너는 전 레이어를 한 스토리지에 넣어 블록당 단일 연속 span을 만든다. opt-125m 실측으로 확인했다. V2에서는 chunk가 12개의 48KiB 조각, V1에서는 576KiB 연속 span 하나다. 러너 선택이 1급 실험 변수이며, 러너 차이는 연산에도 영향을 주므로 비교는 항상 같은 러너 안에서만 한다.

핵심 가설. 현재 runtime의 실제 GPU span 기하에서, pinned CPU staging이 제공하는 비동기성과 수명 분리, 그리고 GDS가 제공하는 CPU 우회의 손익을 비교한다. CPU 홉의 역할은 코얼레싱만이 아니라 GPU 블록을 빨리 놓아주는 수명 분리이기도 하다.

#### expfs 구현

ExperimentalFilesystemSpec은 셋으로 나뉜다. 스케줄러 쪽 FilesystemManager는 해시에서 파일명으로의 매핑(기존 FileMapper 재사용)과 lookup, 완료 의미를 맡고 용량 정책은 없다. worker 쪽 FilesystemWorker는 GPU 블록을 span으로 해석하고 CUDA fence와 job 수명을 관리하며 기존 DualQueueThreadPool을 재사용한다. transport는 CuFileTransport(GDS 직행)와 PosixBounceTransport(같은 control plane에 전송만 pinned bounce, 비교군 E)로 교체 가능하다.

범위 제한: TP=1, 단일 KV group, uniform full-attention 전용의 성능 검증 prototype이다. 정확성 요건은 store 전 compute 완료 fence, load 완료 전 compute 차단, store 중 GPU 블록 재사용 방지, 임시 파일 후 원자적 rename, 4KiB 정렬과 short IO 검증, 출력 토큰 일치 검증이다.

RFC 48504는 같은 방향의 공식 설계 제안(open)이다. 이 RFC는 GDS가 빠르다는 증거가 아니라 부재 원인 중 하나가 인터페이스 경계였음을 확인해 주는 문서이고, 성능 질문은 남겨 두었다. rain 실험이 그 질문에 실측으로 답한다.

#### 비교군

A는 오프로드 없는 재계산, B는 CPU hit, C는 기존 Tiering FS(SSD→CPU→GPU), D는 expfs+GDS(SSD→GPU), E는 expfs+posix bounce(control plane을 D와 고정한 transport 대조). A 대 C/D가 SSD 자체의 정당성, C 대 D가 핵심 판정, E 대 D가 transport 단독 비교다. store와 load는 별도 결론을 낸다.

### 진행 기록

#### 기능 개통

opt-125m으로 TieringOffloadingSpec과 fs 티어가 돈다. chunk 파일 크기 589,824바이트가 손계산(2 x 16토큰 x 12헤드 x 64차원 x 2바이트 x 12레이어)과 일치한다.

함정: Ubuntu 20.04의 시스템 libstdc++가 conda의 libicui18n 요구를 못 채워 vllm import가 죽는다. env.sh가 conda libstdc++를 선로드한다.

FS hit는 CPU hit와 분리해 증명했다. 완전히 새 프로세스(CPU 티어가 빈 상태)에서 같은 프롬프트를 넣으면 저장된 37개 chunk 전량에 파일 read가 발생하고 출력 토큰이 완전히 일치한다. 관측 기법은 in-process 엔진(VLLM_ENABLE_V1_MULTIPROCESSING=0)에 엔진 생성 전 몽키패치 계수를 다는 것.

레버 둘을 소스에서 확인했다. reset_prefix_cache()는 GPU prefix cache만 비워 CPU hit를 강제하고, reset_prefix_cache(reset_connector=True)는 CPU 티어까지 비우되 fs 티어는 의도적으로 보존해 같은 프로세스에서 SSD hit를 강제한다. fs load가 O_DIRECT라 페이지캐시 혼입이 없다.

#### SSD 정당성 게이트

한 프로세스에서 prefix별로 재계산, CPU hit, SSD hit를 차례로 재고 몽키패치 계수로 경로를 검증한다. 함정: store 캐스케이드는 스케줄러 스텝에서만 진행되어 max_tokens=1 요청이 끝나면 store가 다음 generate까지 정체된다. 16토큰 미만 더미 요청으로 스텝을 공급하는 nudge-drain으로 해결.

opt-2.7b 결과. prefix 2032에서 재계산 1.155초 대 SSD hit 0.483초로 2.4배 유리하고 1024에서도 성립하며 512는 경계다. store 비용은 재사용 한두 번이면 회수된다.

#### cuFile 마이크로벤치

함정 둘. cuda-python의 cufile 바인딩은 pip 휠 libcufile을 dlopen해 시스템 라이브러리와 이중 로드되고 비결정 segfault를 낸다. 해법은 시스템 libcufile만 ctypes로 단일 로드하는 gdslib.py. 또 rain의 nvidia-fs는 IO 통계가 꺼져 있어 per-IO 카운터가 0에 고정된다. 대체 증거로 Bar1 매핑 카운터와 cufile.log TRACE 분류기를 쓴다. 분류 결과 등록 IO는 direct, 미등록 write는 nvidia-fs 내부 바운스, 미등록 read는 direct, compat POSIX 폴백은 전무.

행렬 결과(27개 기하 x 4개 transport x 읽기/쓰기, 체크섬 전건 통과). 등록 실패 없음. op별로 필요한 쪽만 등록하면 BAR1 256MB에서 128MiB span까지 등록된다. 단일 span 기하에서 등록 GDS가 posix 대비 쓰기 14%, 읽기 22% 빠르고(최대 3.3GiB/s), 작은 조각의 다중 span에서는 posix 코얼레싱이 이긴다. crossover는 span 조각 1MiB 부근. CPU 사용률은 GDS 쪽이 3할가량 낮고 host 메모리 왕복이 없다. staging 경유는 전 구간 최하위. 게이트 통과.

#### expfs 검증

vLLM 소스 무수정, spec_module_path로 로드. cuFile store 후 새 프로세스 load에서 37개 chunk 전량이 SSD에서 GPU로 직행하고 토큰이 일치한다. posix transport도 동일 통과. V1 러너에서는 canonical 텐서가 하나가 되어 chunk가 단일 span으로 동작한다. 파일은 기존 fs 티어와 바이트 단위로 동일하다. registered_tensors 모드는 KV 풀 전체 등록이 BAR1 초과로 실패하면 경고 후 미등록으로 폴백한다. TRACE 분류로 load 전건 direct 확인.

미시험: preemption이 실제로 유발하는 wait 경로, IO 실패 주입, 멀티프로세스 서빙의 pickle 왕복. V1과 V2 러너는 canonical 구성이 달라 파일이 상호 호환되지 않으므로 root_dir를 분리한다.

#### 비교군 실측

opt-2.7b, prefix 1024와 2032, 동일 조건, 러너 안에서만 비교. prefix 2032 TTFT 중앙값(초):

| 러너 | A 재계산 | B CPU hit | C tiering | D gds | E posix |
|---|---|---|---|---|---|
| V2(기본) | 1.134 | 0.110 | 0.449 | 0.494 | 0.904 |
| V1(cross-layer) | 1.152 | 0.103 | 1.765 | 0.867 | 0.321 |

8개 prefix 동시 load에서는 두 러너 모두 C가 최선(재계산 약 3초 대비 1.3~1.4초). store는 전 구성 동률이고 warm TTFT 영향은 3% 이하.

해석. V2에서 C와 D는 사실상 동률이고, V1에서 D가 C를 2배 이기지만 E 대조군이 그 우세를 분해한다. 같은 control plane의 posix bounce가 D보다 2.7배 빨랐으므로 D의 우세는 가벼운 standalone control plane 덕이 대부분이고 cuFile transport 자체는 오히려 짐이었다. DRAM 1차 티어(B)는 전 조건을 지배해 vLLM의 CPU staging 설계가 실측으로 정당화됐다.

#### nsys 진단: GIL 콘보이

in-engine D는 chunk당 6.8ms인데 동일 조건 standalone은 1.5ms다. 파일 핸들 수명주기(실측 0.3ms), 스레드 경합(8스레드 정상 스케일), cold 목적지 피닝(fresh 주소도 3.2GiB/s)을 모두 기각한 뒤 결정적 관측을 얻었다. nsys를 붙이면 빨라진다. D는 1.22초에서 0.27초로, V1의 C는 2.53초에서 0.65초로. sys.setswitchinterval을 0.5ms로 줄이는 것만으로 절반이 준다.

원인은 GIL 콘보이다. 엔진 busy loop이 메인 스레드에서 GIL을 스위치 인터벌(기본 5ms) 단위로 독점하고 IO 스레드는 chunk당 여러 파이썬 구간마다 재획득에서 굶는다. nsys의 osrt 개입이 시스템콜을 끼워 스위치를 유발해 병목을 치료한 것이다. V1에서 C가 V2보다 4배 느리던 anomaly도 같은 병인. 처방은 바이트당 GIL 크로싱 수를 줄이는 chunk 확대.

#### 재대결: bucketing

expfs에 blocks_per_chunk 지원과 인접 GPU 블록 span 병합을 넣었다. 블록 id가 연속이면 chunk당 IO가 한 번으로 준다.

정정. 처음에 현행 tiering이 큰 block_size에서 vLLM 자체 크래시를 낸다고 기록했으나 오판이었다. 진범은 크래시한 이전 런들이 /dev/shm에 누적시킨 mmap 파일 누출로 tmpfs가 고갈된 것이었고, 깨끗한 상태에서는 정상 동작한다.

prefix 2032 단일 TTFT 중앙값(초):

| 러너 | 구성 | C tiering | D gds | E posix |
|---|---|---|---|---|
| V1 | block 16 | 1.765 | 0.867 | 0.321 |
| V1 | block 256 | 측정 예정 | 0.513 | 0.582 |
| V2 | block 16 | 0.449 | 0.494 | 0.904 |
| V2 | block 64 | 측정 예정 | 0.382 | 0.570 |

bucketing이 판을 뒤집었다. block 16에서 E가 D를 2.7배 이기던 것이, V1 block 256에서는 D가 단일 스트림과 8동시 모두 최선이 되고 V2 block 64에서도 D가 E를 1.5배 이긴다. 진단대로 GIL 크로싱 축소가 작동했다.

#### W1: production trace 리플레이

프레임: 순수 transport 비교가 아니라 각 설계의 최선 구성 간 end-to-end 비교다. V1은 D(block 256) 대 C(block 16), V2는 D(block 64) 대 C(block 16), 러너 간 수치는 비교하지 않는다.

방법. Qwen-Bailian coder trace 600요청을 도착 순서 그대로 closed-loop으로 리플레이한다. 프롬프트는 hash id마다 결정적 16토큰 블록을 생성해 재구성하므로 원 trace의 prefix hit/miss 패턴이 보존된다(공식 replayer와 같은 원리). rain 제약으로 2032토큰에서 절단하고 max_tokens=1로 TTFT를 잰다. 한계: 디코드 부하 부재, 순차 리플레이라 동시성 없음, C는 CPU 8GiB 티어를 추가로 가진 구성.

trace 사전 계산. chain-hash 기준 재사용률은 block 16에서 35.3%, 64에서 34.2%, 256에서 32.6%로 chunk 크기에 거의 둔감하다(재사용이 multi-turn의 full-prefix 연장이라 chunk 정렬을 자연히 따른다). 600요청의 unique KV는 opt-2.7b 기준 어느 granularity에서든 약 200GB로, GPU 약 6GB와 CPU 8GiB를 자연스럽게 수십 배 초과한다. 운영 요건: 런마다 약 200GB가 쌓이므로 kvroot는 런 종료 시 삭제하고 디스크 여유 가드를 둔다.

결과(600요청, 동일 순서):

| | TTFT p50/p95/p99 | tok/s | storage hit | SSD read (IO 수) | CPU 초/요청 | host 왕복 |
|---|---|---|---|---|---|---|
| V1 C(b16) | 1.165 / 3.641 / 5.080 | 1140 | 20.9% | 59.6GiB (12212) | 0.94 | 496GiB |
| V1 D(b256) | 1.880 / 4.133 / 5.828 | 830 | 18.0% | 54.2GiB (713) | 4.44 | 0 |
| V2 C(b16) | 1.160 / 3.107 / 4.043 | 1392 | 21.2% | 56.1GiB (11488) | 0.67 | 498GiB |
| V2 D(b64) | 1.142 / 2.090 / 2.205 | 1596 | 20.5% | 61.5GiB (3191) | 8.23 | 0 |

V2에서는 D가 이겼다. p50은 동률, p95는 1.49배, p99는 1.83배 좋고 처리량 15% 우위, IO 수 3.6분의 1, host DRAM 왕복 0. 대가는 요청당 CPU 12배(미등록 cuFile write의 내부 바운스).

V1에서는 C가 이겼고 synthetic 승자가 뒤집혔다. 세 가지가 겹쳤다. block 256의 거친 매칭이 storage hit를 깎아 재계산이 늘었고, 실전은 store 178GB가 서빙과 동시에 흐르는데 80MiB 미등록 write의 CPU와 GIL 간섭이 miss 요청까지 오염시켰으며, synthetic은 store를 비운 뒤 load만 재는 구조라 이 간섭이 보이지 않았다.

부수 발견. V1 러너와 tiering 조합에서 이 trace로 요청 80 부근에 재현되는 vLLM 크래시가 있다. 스케줄러가 종료된 요청까지 store 준비를 순회하는 동안 매니저가 요청 상태를 먼저 지우는 종료 시점 레이스다. 문서화된 가드(모르는 요청의 store 준비를 건너뜀, 발화 10회로 영향 미미)로 측정을 확보했다.

관찰. C의 storage-hit 요청 TTFT가 miss보다 느리다. hit의 손익은 같은 요청을 재계산했을 때와의 비교로만 판정 가능하므로 절대값으로 단정하지 않는다.

W2 진출 설계는 V2 + D(block 64)다. V1의 D는 store 간섭 처방(등록 또는 staging write, write 스레드 조절, 백프레셔) 전까지 보류.

#### causal-closure와 재현성 검증

V2에서 C(block 64), E(block 64), D(block 64 재측정)를 동일 trace로 연속 측정했다. 결과는 w1/w1_causal.csv에 있다.

| | p50 | p95 | p99 | tok/s | storage hit | CPU 초/요청 |
|---|---|---|---|---|---|---|
| C b64 | 1.155 | 5.950 | 7.946 | 962 | 20.5% | 1.17 |
| E b64 | 1.208 | 4.642 | 5.480 | 840 | 20.5% | 8.49 |
| D b64 재측정 | 1.171 | 4.902 | 5.697 | 837 | 20.5% | 7.53 |
| D b64 단독 재검증 | 1.169 | 4.586 | 5.315 | 837 상당 | 20.5% | 7.51 |

여기서 심각한 사실이 드러났다. D의 1차 측정(p95 2.090, 1596 tok/s, wall 617초)이 재현되지 않는다. 연속 3번째 순서라는 교란을 의심해 디스크가 빈 상태에서 단독으로 다시 돌렸지만 결과는 연속 측정과 같았다(wall 1134초, p95 4.586). 즉 순서와 디스크 상태는 원인이 아니고, 1차 측정이 우연히 빠른 레짐에 있었던 것이다. p50은 전 측정에서 일치하고 tail만 두 배로 갈리는 양상은 앞서 진단한 GIL 콘보이의 timing 민감성과 부합한다. 원인(스케줄링 레짐, 온도/클럭 등)은 미규명으로 남긴다.

이에 따라 W1 판정을 정직하게 수정한다. V2에서 D가 tail과 처리량으로 이긴다는 결론은 1차 측정 레짐에서만 관측된 것이고, 재현 시도 세 번은 모두 C(block 16)의 tail(p95 3.1)보다 나빴다. 현재 근거로는 V2에서도 현행 tiering이 더 안정적인 선택이다. D의 우위 주장은 레짐 변동의 원인이 규명되고 재현 조건이 특정되기 전까지 보류한다.

causal 사슬 자체는 동일 레짐 내 비교로 유효하다. cuFile 단독 효과(E 대 D)는 중립이었다. 같은 control plane에서 transport만 바꿔도 p50, p95, 처리량이 5% 안에서 같다. control plane 비교(C 대 E, 같은 block 64)에서는 tiering이 처리량(962 대 840)과 CPU(1.17 대 8.49)로 이기고 tail은 expfs 쪽이 낫다. bucketing이 tiering에 주는 효과(C b16 대 C b64)는 tail 악화다(p95 3.19 대 5.95). 20MiB chunk를 shm 스테이징과 단일 POSIX write로 미는 현행 경로는 큰 블록에서 오히려 손해를 본다.

종합하면 W1의 최종 그림은 이렇다. SSD 티어 자체는 유효하고(재계산 대비 명확한 이득), 현행 CPU staging 설계(block 16)가 전 조건에서 가장 안정적이며, GDS 직행의 우위는 어느 러너에서도 재현 가능한 형태로 확립되지 않았다. GDS가 이기는 순간이 존재한다는 것(1차 측정)과 그것이 공학적으로 신뢰할 수 있는 우위라는 것 사이의 간극이 이 실험의 마지막 발견이다.

부기. e2e 시험이 같은 태그로 돌며 C b16의 totals 파일을 덮어쓴 사고가 있었고, 로그의 원본 수치와 행 단위 재집계로 복구했다(wall 706.8초, SSD read 60.2GB로 원값 일치). 이후 하네스는 시험용 실행에 별도 출력 경로를 쓴다.

### /dev/shm mmap 누출의 근본 수정

vLLM의 SharedOffloadRegion은 정상 종료의 cleanup에서만 파일을 지우므로 SIGKILL, OOM, init 중 예외, 일부 인터프리터 종료 경로에서 /dev/shm 파일이 누적되고, tmpfs가 차면 다음 엔진이 EFAULT로 죽는 2차 장애가 난다.

수정은 flock 기반이다. 모든 참여자가 파일에 shared lock을 프로세스 수명 동안 쥔다. 커널이 어떤 죽음에도 락을 풀므로, 시작하는 엔진이 exclusive lock을 잡을 수 있는 파일은 고아임이 증명되어 안전하게 회수한다. 살아 있는 다른 엔진의 파일은 락이 잡혀 있어 건드리지 않는다. 같은 engine_id 재시작은 stale을 회수하고 새 creator가 되며, init 도중 실패도 파일을 남기지 않는다.

독립 diff 리뷰가 10건을 지적했고 8건을 반영했다. 모든 unlink에 inode 재검증(경로 대 inode TOCTOU), blocking shared lock(락 실패 삼킴 제거), 합류 후 inode 재확인(재시작 split-brain 차단), 자기 경로의 0바이트 고아는 짧은 유예 후 회수(engine_id 영구 고착 방지). 잔여 위험: 미패치 vLLM 인스턴스가 같은 호스트에 공존하면 그 파일에는 락이 없어 회수될 수 있다(flock 계열 방식의 태생적 한계). EC connector 쪽 동일 계열 결함은 수정 범위 밖으로 기록만 했다.

검증: 회귀 포함 34개 테스트가 4연속 통과, 실제 엔진을 SIGKILL한 뒤 8GiB stale이 다음 시작에서 자동 회수되는 E2E 통과. harness는 삭제 대신 관찰만 한다. 런 전 목록과 용량을 기록하고, 회수는 엔진의 Reclaimed 로그로, 정상 종료는 잔재 0으로 확인한다.

### W2: LEval 실제 텍스트 워크로드와 I/O 스케줄링 연구

증거 축을 분리한다. synthetic은 통제된 기전 확인, Bailian coder는 production 유래 KV 접근 위상, LEval W2는 실제 텍스트 content-validity 교차 검증, 스케줄링 연구는 read/write admission과 제출·완료 방식의 인과 분해다. LEval 워크로드의 정확한 명칭은 real-text synthetic-interleaving long-document workload이며 production이라 부르지 않는다.

#### 선행: 축소 윈도 스케줄링 정책 실험

Bailian 앞 150요청 윈도에서 정책 4종을 각 5회, D/E paired로 측정했다. read-priority 제출(무효), read/write 상호배제(무효), /dev/shm 상태(무효)를 기각하고, store를 요청 사이 gap으로 미루는 deferral만이 FAST 레짐을 10/10, 변동계수 0으로 재현했다. storage 동작은 baseline과 완전히 동일했고(read 317건, store 2,737건, matched 19,936 일치) 효과는 cuFile과 POSIX에서 같았다. 결론: 병목은 read 대 write의 중재가 아니라 store 실행과 foreground 엔진 구간의 시간 중첩이며, 필요한 것은 foreground-aware store admission이다. NVMe 온도는 slow 레짐의 일중 악화만 설명하고 fast 레짐과 무관했다.

#### 준비와 감사

LEval(HF L4NLP/LEval, revision 43b9dbf)에서 문서당 질문 2개 이상, OPT 토크나이저 기준 1,920토큰 이상인 실제 문서 64개를 14개 도메인에서 고정 시드로 선정했다. prefix는 고정 system instruction과 문서를 정확히 1,920토큰(블록 16과 64의 공배수)으로 절단해 만들고, 절단 후 중복되는 prefix와 질문은 제외했다. 검증: 같은 문서의 모든 요청은 첫 1,920토큰이 동일, 질문 suffix와 문서 prefix는 서로 다름, 총 입력 2,032토큰 이하.

스택 감사 결과, libcufile은 ctypes.CDLL 단일 로드(PyDLL 아님, errcheck 없음)로 native 구간에서 GIL이 해제되고, 전용 completion 스레드는 없으며 완료 polling의 실체는 엔진 자체의 파이썬 루프다. cuFile Batch API 5심볼은 시스템 libcufile에 존재하고 ABI는 cufile.h에서 확정했다.

#### Batch API

standalone 1-chunk 파일럿은 전 케이스(미등록/등록 버퍼, 단일/다중 entry, read/write) 완료 이벤트와 체크섬이 정상으로 usable 판정. 그러나 엔진 통합에서는 두 번 연속 행이 발생했다(공유 핸들+완료 스레드 구성, 스레드-로컬 핸들 구성 모두). 판정: integration-blocked. 다중 IO 스레드 환경과의 상호작용이 미해결이며 격리 디버그는 별도 과제로 남긴다.

#### Gate 2 (28문서 기능·용량 파일럿)

28문서(unique KV 16.4GiB, C군 GPU+CPU의 1.22배)에서 자연 filesystem hit 확보(R2 26/28). read-priority(DS1)와 strict phase(DS2)는 동시 실행 baseline과 동률로 무효 재확인. store 간섭은 R1(cold+store)에서 극적으로 드러났다(p50 3.8초, deferral 시 1.11초).

#### Gate 3 최종 (64문서, 반복 측정)

D1과 E1은 5회, 나머지는 3회, 실행 순서 교차. matched와 store IO는 D/E군 전체에서 동일(생략 없음 검증), backlog 0.

| arm | n | R2 p50 평균 | R2 p95 평균 (CV) | R1 p95 평균 | CPU s/런 |
|---|---|---|---|---|---|
| A 재계산 | 3 | 1.090 | 1.161 (0.001) | 1.141 | 5 |
| C tiering b16 | 3 | 0.977 | 4.908 (0.023) | 4.474 | 144 |
| D0 cuFile 동시 | 3 | 0.661 | 0.850 (0.044) | 4.380 | 1,180 |
| E0 posix 동시 | 3 | 0.884 | 1.246 (0.048) | 4.254 | 1,183 |
| D1 cuFile deferred | 5 | 0.637 | 0.786 (0.046) | 1.169 | 114 |
| E1 posix deferred | 5 | 0.920 | 1.206 (0.056) | 1.168 | 123 |

판정.

- cuFile transport 우위가 처음으로 반복 확립됐다. 간섭을 제거한 D1 대 E1에서 R2 p95 10% 이상 우세가 5/5회, 평균 격차 35%다. load 지배적인 실텍스트 재사용 경로에서 cuFile 직행 read가 posix bounce의 2-hop을 이긴다. phase1 마이크로벤치의 read 우위와 방향이 일치한다.
- deferral(foreground-aware store admission)은 R1을 3.7~3.8배, CPU를 10배 개선하며 이 효과는 transport와 무관하다.
- 시스템 수준에서 D1은 현행 C를 전 지표에서 이긴다(R2 p95 6.2배, R1 p95 3.8배, CPU 유사). C의 R2 tail(약 4.9초)은 promotion 경로의 구조적 문제이고, C는 이 워크로드에서 fs hit 수 자체도 런마다 흔들렸다(48~58/64).
- 종료-레이스 버그가 V2 러너에서도 재현됐다(64문서에서 C 3런 전부 크래시, 가드로 우회, 발화 4~14회). W1의 V1 한정 관찰을 넘어 일반 버그로 상향.

한계와 남은 것. W1(Bailian)에서는 현행 C-b16이 최선이었다는 결론과 공존하며, 우열은 워크로드 의존적이다(Bailian은 store가 지속 유입되고 재사용 거리가 길다; LEval W2a는 load 지배적 재사용). Batch API 엔진 통합, py-spy GIL 직접 측정은 미완이다.

#### W2b (decode 포함, 출력 16~128토큰, ignore_eos, 각 3회)

핵심 질문이었던 backlog 안정성은 확인됐다. deferred store는 decode가 길어져도 요청 사이 gap마다 정상 배출되어 backlog 최대 1, 종료 잔량 0, 최대 보류 시간 약 4.5초(한 요청 길이 이내)로 starvation이 없다.

| 구성 | e2e 재사용 p50 | e2e 재사용 p95 (CV) | cold p95 | out tok/s | backlog 최대 |
|---|---|---|---|---|---|
| 기존 tiering | 1.155 | 4.329 (0.28) | 5.654 | 14.8 | 0 |
| cuFile 지연 store | 1.219 | 3.052 (0.01) | 4.487 | 11.6 | 1 |
| POSIX 지연 store | 1.669 | 3.365 (0.01) | 4.492 | 11.1 | 1 |
| cuFile slack-aware+40MB 제한 | 1.249 | 3.119 (0.02) | 7.393 | 11.5 | 5 |

경계 하나를 명시한다. backlog 안정성은 closed-loop 순차 요청에서 확인된 것이다. 이 방식은 요청 경계의 gap이 반드시 존재하지만, 동시 요청이 계속 들어오는 open-loop 서버에서는 전역 foreground gap이 사라져 store가 굶을 수 있다. open-loop/동시성 검증이 다음 우선 실험이다.

판정. decode가 tail을 지배하면서 cuFile 대 POSIX의 e2e p95 격차는 9.3%로 문턱(10%) 아래로 희석됐고 paired 기준 0/3이다. 다만 p50에서는 27% 우위가 유지된다. slack-aware admission(decode 중 배출 허용)은 이 구현에서는 역효과였다. cold 구간 간섭이 재유입되어 cold p95가 최악(7.4초)이고 backlog와 보류 시간도 더 길다. gap 전용 배출이 옳다. 기존 tiering은 처리량(14.8 tok/s)에서 앞서고 tail 안정성(CV 0.28 대 0.01)에서 크게 뒤진다. 종합하면 transport 우위의 실용 가치는 TTFT가 중요한 짧은 출력 서빙에서 크고, 긴 decode에서는 스케줄링 안정성(지연 store의 CV 0.01)이 남는 이득이다.

### 교훈

복사를 없앤다고 빨라지지 않는다는 오래된 교훈이 세 번 확인됐다. 소조각 IO에서, control plane 분해(E 대조군)에서, production trace의 store 동시성에서. 표준 프로파일러가 관측 대상을 바꿔 버리는 경우(GIL 콘보이)에는 프로파일러의 개입 자체가 진단 단서가 된다. 그리고 synthetic 벤치의 승자는 production trace 앞에서 겸손해야 한다.

#### upstream 회귀 검증 (최신 main)

우리가 특성화한 두 버그가 upstream에서 이미 처리됐는지, pinned base가 아닌 최신 main에서 재현 시험으로 확인했다. 검증 환경은 별도 worktree(~/vllm-main, commit 1f1f628859, 0.26.1rc1.dev1488)와 별도 venv(python 3.12, precompiled wheel)로, 실험용 pinned 환경(568afb3a13)은 건드리지 않았다.

종료 시점 _req_state race. 수정 PR #49671(Defer request finalization until final store, 2026-07-25 병합)은 우리 base(07-26 커밋)에 포함되지 않은 것으로 git 조상 검사에서 확인됐다 — 우리가 이 race를 맞은 이유가 설명된다. 최신 main에서 pinned base가 가드 없이 3/3 크래시하던 동일 조건(LEval 64문서 2라운드, tiering b16)을 가드 없이 돌린 결과 128요청 완주, KeyError 0. 대신 "cannot store chunks" WARNING이 다수 찍히는데 이것이 #49671이 도입한 우아한 거부 경로다. 판정: race는 upstream에서 해결됐고 회귀가 아니다. 우리 하네스의 prepare_store 가드는 구버전(v0.26.0) workaround로 기록을 유지하며, 새 issue는 내지 않는다.

/dev/shm 누출. #52596(unlink-after-barrier, 2026-08-31 병합) 적용 상태에서 spec별로 엔진 기동 → SIGKILL → 잔재 검사를 수행했다. CPUOffloadingSpec은 실행 중에 이미 파일이 unlink되어 있고(barrier 발동 확인) SIGKILL 후 잔재 없음. TieringOffloadingSpec은 실행 중에도 파일이 링크된 채였고 SIGKILL 후 그대로 남았다 — 누출 재현. 정적 원인도 확인했다. cpu/spec.py는 barrier=_all_workers_barrier를 넘기지만 tiering/spec.py의 SharedOffloadRegion 생성부 두 곳(스케줄러 쪽 rank=None, worker 쪽)은 barrier 인자를 넘기지 않는다. 단순 배선 누락만도 아닌 것이, tiering의 스케줄러 쪽 opener는 worker collective 밖에 있어 worker barrier 후 unlink하면 나중에 경로로 여는 스케줄러가 실패한다. 구조적으로 (a) 스케줄러를 포함한 rendezvous 또는 (b) unlink 순서에 무관한 liveness(flock, #54124 계열 = 우리 로컬 패치와 같은 접근)가 필요하다. 판정: 후속 버그로 성립. issue 초안은 upstream_check/issue_tiering_shm.md에 있고 제출은 사용자 확인 후 진행한다. 우리 flock 패치는 별도 PR로 내지 않고 이 issue와 #54124 논의에 증거로 연결하는 것이 우선이다.

재현 스크립트는 upstream_check/에 보존한다(race_repro.py, shm_repro.py, run_shm_check.py).

#### W3 open-loop/동시성 검증

W2b의 경계 그대로를 시험했다. closed-loop 순차 요청이 보장하던 요청 경계 gap이 동시성과 지속 도착에서 사라질 때 지연 store가 어떻게 되는가. 하네스는 w2_leval/openloop.py. AsyncLLM은 항상 별도 프로세스(make_async_mp_client)라 계수기·스케줄러 몽키패치가 무효하므로, sync LLMEngine의 add_request와 step을 직접 구동해 continuous batching을 유지한 채 in-process로 쟀다. step 스트리밍 덕에 W2b에서 불가능했던 요청별 실측 TTFT(도착~첫 토큰, 큐 대기 포함)가 나온다. 워크로드는 LEval 64문서 2라운드, 출력 16~128토큰. closed는 동시 스트림 N이 완료 즉시 다음 요청을 넣고, Poisson은 도착률 λ의 지수 간격으로 단일 스트림이 들어온다. λ는 실측 서비스율(D1 약 0.39 req/s, C 약 0.61 req/s) 기준으로 0.3(D1 용량의 약 0.8배)과 0.55(D1 초과·C 근접)를 골랐다.

backlog와 starvation. 전 구성에서 backlog는 유한했다(최대 10, 1Hz 시계열의 최소자승 기울기 분당 ±1 이내, 종료 잔량 0). 무한 굶주림은 없다. 그러나 그 이유가 설계 의도가 아니다. conc≥2에서 전역 gap이 실제로 소멸해(gap 배출 0~1회) 보류 store 전량이 블록 재사용 fence의 강제 배출(76~81회)로 나갔다. 최대 보류 시간은 conc=2에서 10초, Poisson 0.3에서 14초까지 늘었다. 즉 gap 전용 배출 설계는 동시성 2부터 이미 무의미해지고, fence가 안전판이자 사실상의 배출 경로가 된다. 강제 배출은 foreground 실행 중에 일어나므로 deferral의 원 목적(간섭 회피)은 붕괴한다.

closed-loop 동시성 (R2 TTFT 초, 3회 중앙값):

| 구성 | conc 1 p50/p95 | conc 2 p50/p95 | conc 4 p50/p95 (n=3) | conc 8 p50/p95 | conc 4 tok/s |
|---|---|---|---|---|---|
| 기존 tiering | 0.63 / 0.70 | 0.81 / 6.78 | 1.67 / 12.26 | 4.66 / 11.80 | 19.6 |
| cuFile 지연 store | 0.70 / 0.88 | 0.68 / 1.14 | 0.85 / 3.97 | 1.60 / 6.04 | 14.1 |

conc 4는 3회 반복 전 쌍에서 지연 store가 p50·p95 모두 우위(3/3). 기존 tiering은 동시성이 붙는 순간 TTFT tail이 무너지고(0.70→12초), 처리량은 계속 우위(conc 8에서 21.8 대 13.9 tok/s). 주의: 기존 tiering 런은 종료 race 가드가 동시성에서 다발(최대 31/128 요청) — pinned base 한정 caveat.

Poisson open-loop (R2 TTFT 초, 3회 중앙값, λ=0.3):

| 구성 | p50 | p95 | 최대 보류 | gap/강제 배출 |
|---|---|---|---|---|
| 기존 tiering | 0.79 | 8.64 | - | - |
| cuFile 지연 store | 0.87 | 26.85 | 14.2s | 20 / 44 |
| cuFile slack-aware+40MB | 1.25 | 38.42 | 10.8s | 0 / 80 |

λ=0.55(단일 런, 구조 확인용): 기존 tiering p50 13.6초로 버티는 부하에서 지연 store는 p50 132초로 발산 — 도착률이 서비스율을 넘어선 포화.

판정. open-loop 지속 부하에서 관계가 역전된다. 지연 store 계열의 tail이 기존 tiering보다 3~9배 나쁘고(3/3), 원인은 backlog 폭주가 아니라 서비스율 격차다. tok/s가 낮은 쪽은 버스트가 만든 큐를 느리게 비우고, TTFT에 큐 대기가 편입되어 tail이 증폭된다. closed 동시성에서 지연 store가 이기는 이유(요청별 지연 우위)와 open-loop에서 지는 이유(처리량 열위)가 같은 트레이드오프의 양면이다. slack-aware(S4)는 open-loop에서도 지연 store를 구하지 못했다(p95 오히려 열위). 종합하면 gap 전용 배출은 단일 스트림 closed-loop 전용 설계이고, 서빙 레짐에서 GDS 경로가 서려면 store의 처리량 개선(예: 배출 병렬화, Batch API의 원래 자리) 또는 부하 인지형 admission이 선결이다. W2b의 트레이드오프 결론(중앙 지연·안정성 ↔ 처리량)이 open-loop에서는 처리량 쪽이 지연까지 지배하는 형태로 확장된다.

원자료는 results/w2_openloop/(런별 raw.csv, 1Hz timeseries.csv, meta.json, 집계는 analyze_openloop.py).

#### foreground-store 간섭의 함수 수준 원인 규명

W2 이래 이 실험의 최대 미해명 사항 — expfs store가 추론과 동시에 실행되면 CPU가 12배 뛰고 tail이 무너지는데 store를 요청 사이로 미루면 정상화되는 이유 — 를 함수·대기 구간 수준까지 내려가 확정했다. 프로토콜은 5단계: 계측 없는 기준 재현, 관찰 전용 계측, nsys와 py-spy, 원인 분해 격리, 원인 제거 검증. 워크로드는 Bailian 앞 150요청(opt-2.7b, V2, b64).

기준 재현(각 3회). GDS-동시저장 SLOW(p95 5.5~6.5초, CPU 약 1,074초), POSIX-동시저장도 SLOW(CPU 약 1,270초) — cuFile 고유 아님이 재확립. GDS-지연저장 FAST(p95 1.196초, CV 0.0004, CPU 약 90초). 기존-tiering은 tail 높고(p95 약 5.1초) CPU 낮음(약 146초).

계측과 프로파일. 관찰 전용 NVTX·ns 계측(store 동작 무수정)에서 store의 CUDA event 대기가 평균 358ms 대 0.7ms(500배), 총합 980초로 CPU 초과분(1,003초)과 산술적으로 일치했다. cuFile write 자체는 양쪽 평균 15.5ms로 동일 — native 쓰기는 느려지지 않는다. nsys python-gil trace에서는 CUDA 동기화 레코드 건수가 동일(254,200)한데 총 시간이 1,021초 대 96초로 10.6배 갈렸고, py-spy CPU-활성 캡처에서 활성 샘플의 87.9%(1,568 스레드-초)가 ev.synchronize 스택이었다 — 대기가 sleep이 아니라 스핀임의 직접 증거.

원인 분해(격리, 각 3회 반복). store 경로에서 한 요소씩만 남겼다. 결과는 이중 해리다.

| 변형 | 남긴 동작 | p95 | CPU |
|---|---|---|---|
| 제어부만 | queue·rename·완료 처리 | 1.163~1.168 | 19~20초 |
| CUDA만 | event sync만, IO 제거 | 1.164~1.165 | 1,504~1,507초 |
| 네이티브대기 | GIL 놓는 1.28초 sleep | 4.334~4.356 | 31~32초 |
| POSIX만 | pwrite만 | 4.337 | 57초 |
| cuFile만 | cuFile write만, event 제거 | 4.312 | 110초 |

CPU 폭증은 CUDA event sync 단독으로 최대 재현되고 그때 tail은 완전히 정상이다. tail은 순수 sleep만으로 완전 재현되고 그때 CPU는 정상이다. 두 증상은 원인이 다르다.

원인 제거 검증(3회). torch.cuda.Event(blocking=True) 한 줄 차이로 전체 GDS-동시저장 경로의 CPU가 1,074초에서 118~123초로 9배 정상화됐고(3/3), tail은 예측대로 잔존했다(p95 5.1~5.5). LEval 8문서 store workload 교차 확인에서도 CPU 119.9→5.0초(24배), tail 잔존 — Bailian 특이 아님.

확정 결론.

첫째, CPU 12배 폭증의 원인은 expfs 구현의 CUDA event 스핀 대기다. torch.cuda.Event() 기본 플래그는 cudaEventSynchronize를 busy-wait로 돌리고, store 스레드 8~16개가 foreground compute가 event 완료를 늦추는 동안 스핀한다. 이것은 Python expfs 구현의 문제이지 GDS 자체의 약점이 아니다 — blocking 플래그 한 줄로 제거된다.

둘째, TTFT tail의 원인은 store 작업의 시간 점유가 요청 경계를 침범하는 것 자체다. transport 종류·CPU 사용량·GIL과 무관하며(순수 sleep으로 재현), store가 진행 중인 동안 블록 delay-free가 길어져 다음 요청 admission이 지연되고 엔진이 빈 step을 공회전한다(요청당 830 step, 정상의 17.7배). 제거 방법이 곧 지연 store다 — gap에서 store를 완결시켜 경계 침범을 없앤다. 이는 구현 언어를 바꿔도 남는 store scheduling 구조 문제다.

셋째, 기각된 가설: cuFile·nvidia-fs 고유 문제(POSIX도 SLOW, cuda_only에서 tail 정상, write 속도 불변), CUDA driver/context 경합(cuda_only에서 tail 정상), Python 제어부·GIL 단독(제어부만 FAST). sudo perf 단계는 rootless 증거로 원인이 분리되어 불필요했다.

과거 서사의 정정. 이전에 "GIL 콘보이"로 불렀던 병명은 이번 분석으로 두 요인으로 분해된다. 과거의 간접 증거(nsys osrt 개입 4.5배, switchinterval 2.3배 가속)는 load-side 단독 측정 레짐의 것이고, store 동시실행 레짐의 병인은 event 스핀과 경계 침범이다. 엔진의 GIL 보유 2.5배와 빈 step 공회전은 실재하지만 격리에서 tail을 단독으로 만들지 못했다.

일반화 범위. event 스핀은 CUDA의 일반 동작이라 rain 한정이 아니며(코어 수와 GPU 세대에 따라 정도 차이는 있을 수 있음, 타 서버 재현은 미실시), store 경계 침범은 vLLM 블록 수명 구조의 일반 성질이다. "vLLM production이 GDS를 기본 경로로 쓰지 않는 이유"에 대한 이 실험의 증거 기반 답: GDS transport 자체는 결격이 아니고(load에서 우위 반복 확립, write 속도 동일), 결격은 worker 쪽 store 실행이 추론과 자원·블록 수명을 공유하는 구조에서 온다. GDS를 서빙에 넣으려면 transport가 아니라 store의 실행 시점(admission)과 event 대기 방식, 블록 수명 분리를 함께 설계해야 한다.

산출물: sched/prof/analysis_notes.md(전체 수치), sched/prof_instrument.py(계측), sched/prof_isolate.py(격리), sched/prof_fix.py(제거 검증), nsys 리포트와 py-spy speedscope는 용량 문제로 로컬 보존(sched/prof/).

#### GPU staging ring과 비차단 저장 정책

원인 규명이 지목한 tail 원인(store의 GPU 블록 수명이 SSD 쓰기에 결합)에 대한 설계 응답. KV 블록을 GPU staging ring으로 D2D 복사한 뒤 그 슬롯에서 cuFile로 쓴다. store job 완료를 복사 완료 시점으로 정의해 원본 블록을 즉시 반환하고, SSD 쓰기는 writer 스레드가 비동기로 수행한다. tiering이 CPU 복사로 얻던 수명 분리를 GPU 안에서 재현하는 셈이다.

기능 검증: 저장 파일이 기존 cuFile 산출과 바이트 동일, 새 프로세스 재로드에서 토큰 일치, ring 등록 direct 경로(TRACE에서 bounce 0), 원본 블록 반환 후 즉시 덮어써도 checksum 유지. 완료 보고와 SSD commit이 분리됨을 계측으로 확인(completed_before_file = 전 chunk).

Bailian 150(V2 b64, 3회)에서 CPU는 동시저장 1,074초에서 staging 97초로 완전히 정상화됐으나 tail은 p95 5.4초로 남았다. 계측이 원인을 규명했다. JOB_HOLD(GPU 블록 점유)가 4.8초로 여전히 컸고, 원인은 ring slot 고갈이다. 동시 실행 중 cuFile write가 chunk당 177밀리초(마이크로벤치의 30배, foreground와 경합)이고 슬롯이 유한하니, 블록이 복사 완료가 아니라 슬롯이 빌 때까지(=이전 SSD write까지) 잡힌다. slot을 6에서 12로 늘려도 JOB_HOLD가 4.8초에서 4.2초로 거의 안 줄어 slot-depth가 아니라 write-throughput 병목임이 확인됐다.

해결은 비차단 admission이다. 포화 시 원본 블록을 붙잡지 않고 즉시 처리한다. skip은 저장을 생략(원본 즉시 해제, 파일 없음=miss), cpu_fallback은 D2H로 CPU pinned 버퍼에 복사(원본 해제)한 뒤 CPU에서 쓴다. 어느 경로든 블록은 유한 복사로만 해제되고 슬롯을 기다리지 않는다.

Bailian 150·600, 양 러너에서 3회씩. skip과 cpu_fallback이 JOB_HOLD를 block의 5,312밀리초에서 517~986밀리초로 낮췄고(7~10배), 슬롯 대기 재폴링이 843만 회에서 2.5천 회로 사라졌다. tail은 block·tiering의 5~6.6초에서 1.3초로, CPU는 최저로 떨어졌다. 대가는 storage hit 손실(skip 47~60%, cpu_fallback 34~35%)로, cpu_fallback이 tail은 skip과 동급이면서 hit를 더 지키는 균형점이다.

정확성 요건은 QA 에이전트 적대적 리뷰로 검증했다. 발견된 ship-blocker 2건(하네스 계수 래퍼의 인자 불일치로 write가 조용히 실패, block 정책 종료 시 미admit job 누수)을 수정 후 재검증했다. 600요청 확장과 V1 b256에서도 결론이 재현됐다. V1 b256 재검증은 과거 V1+b256 패배가 cuFile transport가 아니라 event spin과 블록 수명 결합 탓임을 분리 확인했다(blocking event만으로 tiering과 대등, 비차단으로 3배 우위). posix staging도 cuFile staging과 tail이 동일해, 이득의 원천이 transport가 아님을 재확증했다.

#### Prefix Value Admission — 무엇을 저장할지

비차단 정책의 admission이 도착 순서(random skip)라, 버리는 저장이 어떤 KV인지 고려하지 않는다. 저장 가치가 높은 KV를 골라 저장하면 같은 tail에서 hit 손실을 줄일 수 있다. scheduler가 store 후보 chunk의 관측 이력(빈도, 재사용 거리)으로 value를 산출해 worker에 admission hint(GDS_RING/CPU_FALLBACK/DROP + priority)를 내리고, worker가 압박과 결합하는 구조를 구현했다.

오프라인 시뮬레이션 게이트: Bailian 600 trace에서 value/reuse-distance 신호가 arrival-order random 대비 useful hit/GiB 효율 2배(V1 633→1,183, V2 477→1,174), wasted write 84%→53%. future-reuse oracle이 상한을 형성. 신호 예측력이 확인돼 구현으로 진행했다.

Bailian 150·600, 양 러너: seen-twice 빈도 필터가 random을 이겼다. 150 V2에서 hit +29%(10,976 대 8,480)를 write 63% 절감(2.7 대 7.3GiB)으로 달성, 같은 tail. hit/GiB 4,014로 오프라인 oracle 효율에 근접. random이 일회성 chunk에 슬롯을 낭비하고 포화 시 재사용 chunk를 맹목 drop하는 반면 seen-twice는 재등장 chunk만 저장하기 때문이다. 600에서는 hit 격차가 좁혀지고(+5~10%) write 효율 격차가 넓어졌다(2배). reuse-distance 정교화(value_density)는 Bailian에서 seen-twice와 동일했다. 워킹셋 200GiB가 GPU+CPU 13.5GiB를 크게 넘어 짧은 거리 재사용도 evict되므로, 요청-거리 기반 evict 예측이 변별력을 못 가졌다.

혼합 workload가 이 결론을 조건부화했다. real-text synthetic-interleaving admission workload를 만들어(one_shot·near_reuse·far_reuse·repeated를 의도적으로 섞음) 카테고리별 useful hit을 쟀다. seen-twice는 far_reuse(먼 거리 1회 재사용)에서 useful hit이 0이었다. 2번째 등장에서 저장하는데 far_reuse는 2회만 등장해 저장이 도움될 3번째가 없기 때문이다. 이 far_reuse가 Mooncake가 SSD의 표적으로 지목한 10.3% cold 재사용에 해당한다. 반면 1번째 등장에서 저장하는 random은 far_reuse의 2번째를 잡았고(V1 far_hit 2,928, V2 2,880), 총 useful hit도 random이 seen-twice를 앞섰다. Bailian과 정반대다. 양 러너에서 동일하게 재현됐다.

결론: 최적 admission 정책은 재사용 구조에 의존한다. 반복형 재사용에서는 빈도 필터가 낭비를 최소화하며 이기고, cold-tail 재사용에서는 1번째 등장 저장에 one_shot만 예측 drop하는 정책이 필요하다(빈도 필터와 정반대 방향). 재사용 구조를 감지해 전환하는 하이브리드가 이상적이며, 1번째 등장에서 cold-tail을 예측하는 신호(재사용 거리는 사후에만 알 수 있어 못 씀) 설계가 향후 과제다. 이로써 이 실험의 결정 축이 하나 더 확장된다. transport(무관) → store 실행 구조(수명 분리·비차단) → admission(무엇을 저장, workload 의존).
