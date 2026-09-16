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

### 가중치 스트리밍과 KV 오프로드의 결합 (OPT-66B)

앞선 실험(01~05)은 모델이 GPU에 통째로 올라가고 CPU 티어를 인위적으로 제한해 SSD를 필요하게 만든 설계. SSD의 당위성은 가중치가 GPU와 RAM을 넘을 때 생기므로, OPT-66B fp16 132 GB를 Quadro RTX 5000 16 GB 한 장과 RAM 125 GiB에서 돌리는 조건을 만들고 그 위에 KV 오프로드를 올렸다. 실험 폴더는 06-weight-offload(가중치 경로), 07-combined(결합), 08-cufile-bounce(cuFile 조각 크기), 09-kv-policy(배치 구성, 원인 규명, 스케줄러 수정, 저장 정책, 양자화).

#### 구성과 정적 버퍼

vLLM v0.26의 가중치 오프로드는 UVA와 Prefetch 두 백엔드뿐이고 SSD 티어가 없음. Prefetch 백엔드에 세 번째 티어를 넣어 층을 GPU 상주, pinned host, SSD 파일로 나누고 매 forward 정적 GPU 버퍼 풀로 층을 차례로 올린다(포크 ~/vllm 브랜치 weight-ssd-offload, offload_ssd_path와 offload_host_fraction 등 다섯 설정). host 비율은 MemTotal 대비이며 층 하나가 1.9 GiB.

| host 비율 | 예산 | host 층 | SSD 층 | forward당 SSD 읽기 |
|---|---|---|---|---|
| 0.1 | 12.6 GiB | 6 | 55 | 104.4 GiB |
| 0.3 | 37.6 GiB | 19 | 42 | 79.7 GiB |
| 0.5 | 62.7 GiB | 33 | 28 | 53.2 GiB |
| 0.85 | 106.7 GiB | 56 | 4~6 | 7.6~11.4 GiB |

층의 큰 행렬 네 개(fc1 648 MiB, fc2 648 MiB, qkv 486 MiB, out_proj 162 MiB)에 맞춘 정적 버퍼 네 개를 매 층 재사용. NVMe가 GPU 메모리에 직접 DMA하려면 목적지가 BAR1 창에 매핑돼야 하는데 이 카드의 창이 256 MiB라 out_proj만 등록되고 셋은 창보다 커서 거절. 등록 실패는 POSIX로 떨어지지 않고 cuFile이 내부 캐시(128 MiB, 1 MiB 조각)로 DMA한 뒤 GPU 안에서 D2D로 옮기는 두 홉 경로를 탄다. forward당 8만 회. 이 경로의 병목은 wall clock가 아니라 CPU와 스레드 경합에 나타남(아래 조각 크기 절). VRAM 전체가 창인 데이터센터 GPU에서는 없는 문제.

forward 하나는 host에서 GPU로 106 GiB(h0.85)와 SSD에서 층 몇 개를 옮기는 고정 비용이라 토큰 수와 거의 무관. h0.85에서 약 13초, h0.3에서 약 28초. 이 고정 비용이 이후 모든 결론의 전제.

#### 가중치 경로 비교

기본 구성 host 0.3, 프롬프트 4개 × 256토큰, decode 8토큰. 요청 4개 합계 1,056토큰이 자동 배분된 GPU KV 2,512토큰(5.55 GiB)에 한 배치로 들어가 forward 시간이 배치 분할이나 선점 없이 전송 비용만 반영. 두 경로 모두 매 forward 같은 877 GiB를 읽고 생성 토큰이 전 런에서 동일. 반복 3회 중앙값, load 기준 8% 이상 느린 이상치 2런 제외.

| 조건 | prefill | decode step | CPU | nvidia-fs 읽기 | 평균 IO |
|---|---|---|---|---|---|
| cuFile, 스레드 4 | 31.1 s | 28.5 s | 159 s | 826,056 | 1.1 MiB |
| cuFile, 스레드 4, ring 16 MiB | 31.4 s | 29.8 s | 133 s | 112,728 | 8.2 MiB |
| cuFile, 2-layer prefetch, 스레드 8 | 29.9 s | 29.7 s | 158 s | 863,016 | 1.1 MiB |
| cuFile, 2-layer prefetch, 스레드 8, ring 8 MiB | 27.4 s | 26.7 s | 137 s | 112,728 | 8.2 MiB |
| POSIX, 스레드 4 | 72.0 s | 68.0 s | 146 s | 0 | |
| POSIX, 2-layer prefetch, 스레드 8 | 70.0 s | 68.2 s | 142 s | 0 | |

- cuFile은 디스크 한계. decode 중 NVMe 2.96 GB/s, 점유 90%로 80 GiB를 27초에 읽는 값과 일치. POSIX는 점유 94%인데 1.2 GB/s. 스레드마다 pread, H2D, 동기 대기가 직렬화되어 디스크 큐가 빔.
- 경로 증명은 셋. cuFile TRACE 분류 DIRECT, nvidia-fs 카운터(cuFile만 877 GiB 증가), nsys memcpy 집계(cuFile은 SSD 몫이 D2D 863 GB, POSIX는 H2D 941 GB 추가).
- 2-layer prefetch 단독은 오히려 느림. 스레드 8개가 cuFile 내부 1 MiB 캐시를 두고 경합. ring이 그 경합을 8 MiB 직접 DMA로 치환할 때만 6% 이득. ring 슬롯 총량은 BAR1 안이어야 하며 16 MiB × 2 × 8은 13개까지만 등록되어 8 MiB로.
- host 비율 스윕(스레드 4): decode는 SSD 층 수에 거의 비례(0.5, 0.3, 0.1에서 22.8, 28.5, 56.5초)하고 그 위에서 cuFile 배수 2.2~2.4배 유지. 0.1은 디스크 85% 점유로 1.6배.
- 부산물로 upstream Prefetch 오프로더 버그를 발견해 수정(포크 3fc4433b62). 모듈 수가 prefetch step으로 나뉘지 않으면 패스 경계에서 슬롯이 충돌해 garbage 토큰. opt-2.7b 31모듈에서 재현.

#### pinned 할당 올림과 host 상한

결합 실험은 가중치를 최대한 RAM에 두려고 host 0.92로 시작했고 머신이 통째로 죽었다(재부팅). 원인은 torch CachingHostAllocator가 pinned 블록을 2의 거듭제곱으로 올리는 것. fc1과 fc2 679 MB가 각각 1 GiB를 차지해 층당 실제 점유가 예산의 1.45배. 06의 "오버헤드 16 GiB"도 대부분 이것. 0.85부터 0.70까지 워치독이 전부 종료.

수정은 VLLM_OFFLOAD_PIN_EXACT. pageable 버퍼를 정확한 크기로 잡아 cudaHostRegister로 고정(포크 2f050f7fd5). opt-2.7b 검증에서 host 티어 4.10 GiB의 Shmem 증가가 5.77 GiB에서 0.08 GiB로, 출력 토큰 동일, H2D 대역폭 동일(12.3 GB/s). 함정 둘. 로더가 cuda device 컨텍스트 안이라 버퍼에 device를 명시해야 하고, is_pinned 판정이 스토리지 시작 주소를 보므로 정렬 오프셋이 아닌 스토리지 시작에서 등록해야 함. 이후 0.85 성립(host 56층 106.3 GiB, SSD 4층, 실행 후 여유 12.6 GiB).

워치독 memguard.sh는 1초 샘플링으로 MemAvailable 2 GiB 미만이면 즉시, PSI full은 MemAvailable 4 GiB 미만일 때 30% 초과면 프로세스만 종료. PSI 단독 임계는 체크포인트 읽기의 페이지캐시 회수를 오탐. 캠페인 스크립트의 성공 판정은 파이프 종료코드가 아니라 결과 json 존재로. 실행은 setsid nohup으로 분리(tmux new-window는 사용자 화면을 바꿔 Ctrl-C 사고를 유발).

#### 결합 실험

가중치 3단 위에 KV를 expfs 스펙으로 SSD에 저장하고 프리픽스 적중 시 읽는다. 프롬프트 8개 × (448 + 32)토큰, decode 8, GPU KV 1.5 GiB(480토큰 하나 분량이라 batch 1 순차), 2라운드(cold 저장, 적중 로드). 레짐 1은 host 0.85 × KV cufile, posix, 저장 안 함. 레짐 2는 host 0.3 × cufile, posix. 전 런 출력 토큰 일치.

| 조건 | 1라운드 prefill | 2라운드 prefill | decode step | 가중치 SSD 읽기 |
|---|---|---|---|---|
| h0.85, 저장 안 함 | 108.2 s | 108.0 s | 96.8 s | 1.1 TB |
| h0.85, cuFile | 113.4 s | 100.0 s | 97.3 s | 1.1 TB |
| h0.85, POSIX | 115.7 s | 104.7 s | 97.5 s | 1.1 TB |
| h0.3, cuFile | 240.0 s | 228.0 s | 225.3 s | 11.4 TB |
| h0.3, POSIX | 248.0 s | 236.2 s | 229.4 s | 11.4 TB |

- SSD KV 적중이 재계산보다 빠름. h0.85에서 재계산 108.0초 대 cuFile 100.0초. 가중치가 디스크를 포화시키는 h0.3에서도 cuFile 우위가 8.2초로 유지.
- decode step은 전 조건 동일. KV 경로는 prefill 적중에만 개입.
- 다만 이 배치에서 KV 트래픽은 가중치의 0.3~3%라 총 시간 기여가 4~8%. decode step 96초는 8프롬프트 순차 forward 8회의 합.

#### cuFile 조각 크기

등록 안 된 정적 버퍼로 가는 읽기는 cuFile 내부 캐시의 조각 크기(cufile.json per_buffer_cache_size_kb, 기본 1 MiB)로 쪼개진다. 06의 ring 대신 이 값만 키워도 되는지를 06 기본 구성에서 확인. 제약은 max_device_cache_size_kb를 조각 크기로 나눈 값이 io_batchsize 이상이어야 한다는 것이며, 아니면 cuFile이 조각을 1 MiB로 되돌린다. 동기 cuFileRead만 쓰므로 io_batchsize를 32, 16, 8로 낮춤. 런마다 64 MiB 검증 읽기의 TRACE 로그로 적용을 확인.

| 조각 | step/스레드 | prefill | decode step | CPU |
|---|---|---|---|---|
| 1 MiB (06 기준 3회) | 1/4 | 30.8~31.8 s | 28.1~29.1 s | 159~178 s |
| 4 MiB | 1/4 | 30.0 s | 27.7 s | 138 s |
| 8 MiB (2회) | 1/4 | 30.6~30.8 s | 28.4~28.6 s | 114 s |
| 16 MiB | 1/4 | 실패, BAR1 부족 | | |
| 8 MiB | 2/8 | 실패, BAR1 부족 | | |
| 4 MiB | 2/8 | 27.3 s | 27.0 s | 150 s |

- 조각을 키워도 wall clock는 디스크 한계에 붙고 CPU만 8 MiB에서 28% 감소. ring 단독(17%)보다 큼.
- 4 MiB에 2-layer와 스레드 8을 얹으면 27.0초로 06의 ring8 조건(26.7초)과 동급. ring 코드 없이 설정 파일로 대체 가능. 06의 "8 스레드가 1 MiB 캐시를 두고 경합" 해석이 맞음.
- 실용 상한은 step 1에서 8 MiB, step 2에서 4 MiB. KV 쓰기까지 더해지면 8 MiB도 BAR1 부족으로 cuFileWrite 실패(dmesg no space for BAR1 mappings).
- 1 MiB 경로가 간헐적으로 3배 느려지는 모드가 있음. 캠페인 당일 1 MiB 런 4회가 모두 66초였다가 다음 날 28초로 복귀. GPU 클럭, pinned 방식, 상주 층 수, shadow 버퍼, cuFile 경로, CPU 배치, 드라이버 버전 모두 배제. 저장장치 일시 상태가 남은 후보이며 미확립. 4 MiB 이상은 흔들리지 않아 이후 실험은 CUFILE_ENV_PATH_JSON으로 4 MiB 고정.

#### 배치가 있는 구성에서의 손실

07은 KV 1.5 GiB라 batch 1이었고 KV 트래픽이 가중치의 0.3%라 정책이 관여할 몫이 없었다. GPU 상주를 4층에서 2층으로 줄이고 그 자리를 KV에 줘(5.5 GiB) 요청 5개가 한 배치로 올라가게 했다. 프롬프트 16개, 프리픽스 448 + 32, host 0.85, 2라운드.

| 조건 | 1라운드 prefill | 2라운드 prefill | 두 라운드 CPU |
|---|---|---|---|
| 저장 안 함 | 72.8 s | 72.2 s | 935 s |
| cuFile | 80.2 s | 96.7 s | 1,081 s |
| POSIX | 88.7 s | 116.1 s | 2,386 s |

프리픽스 적중이 재계산보다 24.5초, 34% 느림. 프리픽스를 1700토큰으로 늘리면(상주 0층, KV 10 GiB, 프롬프트 6개) 9.4% 손해로 줄지만 방향은 같음. 비용 분해로 세 가설을 기각. 프롬프트를 절반으로 줄이면 손해가 34%만 감소(바이트 비례 아님), 블록을 64에서 256으로 키우면 손해 증가(IO 건당 고정 비용 아님), host 0.75로 가중치 SSD 트래픽을 2.2배로 하면 손해 9%만 증가(경합 아님).

#### forward 비용의 분해

이 조건의 병목은 forward마다 GPU 밖에서 들여오는 가중치 이동이고, 그 시간은 host 티어와 SSD 티어의 크기를 각 경로의 실측 대역폭으로 나눈 합으로 설명된다. host에서 GPU는 pinned H2D 12.3 GB/s(PCIe 3.0 x16), SSD에서 GPU는 cuFile 3.2 GiB/s(gdsio, 06 decode 중 NVMe 점유 90%). KV는 계산 중 GPU 안에 있어 이 합에 들어가지 않는다.

| 구성 | host 티어 | SSD 티어 | 모형 | 실측 decode step | 출처 |
|---|---|---|---|---|---|
| host 0.85, 상주 2층 (09) | 106.3 GiB, 9.3 s | 11.4 GiB, 3.6 s | 12.8 s | 13.2~13.3 s | ab-none, ph-none |
| host 0.85, 상주 4층 (07) | 106.3 GiB, 9.3 s | 7.6 GiB, 2.4 s | 11.7 s | 12.1 s | h0.85-kv* |
| host 0.5 (06) | 62.7 GiB, 5.5 s | 53.2 GiB, 16.6 s | 22.1 s | 22.8 s | c-h0.5-r1 |
| host 0.3 (06, 07) | 36.1 GiB, 3.1 s | 78~80 GiB, 24.3~24.9 s | 27.4~28.0 s | 28.1~28.5 s | c-h0.3-r1~3, h0.3-kv* |
| host 0.1 (06) | 11.4 GiB, 1.0 s | 104.4 GiB, 32.6 s | 33.6 s | 56.5 s | c-h0.1-r1 |

- decode step은 출력이 있는 step 길이의 중앙값. 러너의 phase 라벨은 첫 토큰을 기다리는 요청이 남아 있으면 decode forward도 prefill로 적으므로 그 평균(11.3초)은 쓰지 않음.
- host 0.3에서 0.85까지 cuFile 런은 모형과 5% 안에서 맞음. 0.85에서는 PCIe 몫이 7할, SSD 몫이 3할이라 SSD를 무한히 빠르게 해도 9.3초가 남고, 이 조건에서 SSD 대역폭은 forward의 3할만 좌우. host 비율을 내릴수록 SSD 몫이 커져 0.3에서는 9할.
- 06의 host 0.1은 모형보다 68% 느렸으나 4 MiB 조각으로 다시 잰 기준표 절의 host 0.1은 1% 안. 06은 1 MiB 조각이라 느린 모드로 봄.
- 1 MiB cuFile 조각 구성은 간헐적으로 2~2.5배 느려져 모형에서 벗어남. 4 MiB 조각에서는 재현되지 않음.
- 09의 손익은 이 고정비 위에서 정해짐. 적중이 아끼는 상한은 prefill forward의 토큰 몫 5초(18.7초에서 13.6초)이고, 배치가 쪼개져 forward가 2개 늘면 27초를 잃음. 이 분해를 먼저 놓았으면 경합이나 폴링 가설 이전에 손해의 크기와 상한이 정해졌을 것. 초기에 그 순서를 거꾸로 밟은 것이 09의 시행착오.
- tools/compare_results.py가 06, 07, 09의 결과 json 전체를 이 모형과 대조하고 15% 이상 벗어난 런과 기준 런 대비 10% 이상 움직인 런을 표시. 새 결과는 이 표부터 확인.

#### 구간 분리와 원인

wall clock 차이로는 갈리지 않아 엔진 step을 직접 돌리는 계측기(run_phase_66b.py)를 만들었다. KV 읽기와 쓰기 건마다 시각과 outstanding, step마다 가중치와 KV 바이트, 요청마다 제출과 첫 토큰과 완료 시각을 기록하고, decode step과 KV 읽기의 실제 중첩을 IO 구간 교차로 계산. 초기의 generate 두 번 방식은 두 번째 호출의 prefill이 decode 몫으로 섞여 "decode 손해"라는 잘못된 결론을 냈고 폐기.

| 2라운드 | 저장 안 함 | cuFile |
|---|---|---|
| wall | 449.4 s | 507.9 s |
| decode step에 겹친 KV 읽기 | 해당 없음 | 0.0초 |
| forward 수 | 32 | 38 |
| forward별 출력 요청 수 | 0, 5, 5, 5, 5, 5, 5, 5 | 0, 1, 1, 5, 5, 5, 5, 5, 5, 4 |
| forward 평균 | 13.5 s | 13.3 s |
| 폴링 step | 4개 | 1,176개, 합계 0.4 s |
| 라운드 경계 잔여 store | 0 | 0 |

- decode 감속은 없음. forward 길이가 기준선과 같은 13.3초. nsys 리포트를 다시 보면 KV 읽기 112건(점유 34.3초)이 2라운드의 13.3초짜리 forward들과 31.5초 겹치는데, 게이트 없는 이 런에서 그 forward들은 먼저 도착한 요청 하나가 혼자 도는 prefill이고 길이는 늘지 않았다. 처음 적었던 "중첩 0"은 러너가 마지막 배치만 decode로 라벨한 범위의 값이라 정정. 게이트를 켠 기준표 런에서는 읽기가 배치 사이 빈 시간에만 일어나 forward 28개 전부와 겹침 0.
- 쓰기만 있는 1라운드는 기준선과 같은 444.0초. 저장은 공짜.
- 손해는 전부 forward 개수. GPU KV에 5개가 함께 올라가는데, KV가 먼저 도착한 요청 하나가 두 forward를 혼자 돌고 나머지 넷이 뒤늦게 합류하며 먼저 나간 요청이 먼저 끝나 마지막에 4개짜리 forward가 남는다. 배치 한 번에 forward 8개가 10개로. 배치 3번이면 forward 6개 약 80초, 쓰기가 없어져 prefill forward가 18.7초에서 13.6초로 짧아진 이득 20초를 빼면 관측된 58.5초와 일치.
- 읽기가 디스크를 점유한 시간은 다 합쳐 34.6초, 최대 동시 8개. 디스크 포화 아님.
- vLLM 스케줄러는 로드가 끝난 요청을 바로 승격한다. forward가 수십 밀리초인 보통 서빙에서는 맞는 설계이고 상류 이슈 41784와 RFC 43702가 의도된 설계임을 확인. 가중치 스트리밍이 forward를 13초로 고정하면 그 설계의 비용 모델이 뒤집힌다. 이 조합을 다룬 탑티어 논문은 조사 범위에서 없음.

#### 스케줄러 게이트

포크 스케줄러에 승격 검사 하나를 추가(VLLM_KV_LOAD_WAVE_GATE, 기본 꺼짐, 커밋 2998fcca0b와 b576070a77). 수준 1은 자기 로드가 끝난 요청을 함께 올라온 다른 요청의 로드가 모두 끝날 때까지 승격하지 않음. 수준 2는 로드가 없는 신규 요청도 동료가 로드 중이면 계산을 미룸. 상한 VLLM_KV_LOAD_WAVE_WAIT_S 기본 30초. 상류 PR 55724는 토큰 예산 소진 시점을 고치는 것이라 지점이 다름.

같은 시스템 상태에서 연달아 측정, cuFile 4 MiB 조각, 전부 반복 워크로드.

| 2라운드 | 재계산 | cuFile 게이트 없음 | cuFile 게이트 1 |
|---|---|---|---|
| wall | 439.5 s | 503.7 s | 428.4 s |
| forward 수 | 32 | 38 | 32 |
| 마지막 요청 첫 토큰 | 360.7 s | 424.9 s | 349.5 s |
| 게이트 대기 합계 | | | 4.6 s |

재사용 4개와 1회성 12개가 섞인 워크로드, 3라운드.

| 라운드 | 재계산 | 게이트 0 | 게이트 1 | 게이트 2 | forward |
|---|---|---|---|---|---|
| 2 | 440.5 s | 477.2 s | 477.4 s | 437.8 s | 32 / 35 / 35 / 32 |
| 3 | 440.6 s | 477.1 s | 477.3 s | 437.9 s | 32 / 35 / 35 / 32 |
| 합계 | 1,321.9 s | 1,395.4 s | 1,396.3 s | 1,317.7 s | |

반복 요청 첫 토큰 중앙값 32초(재계산), 55초(게이트 없음), 29초(게이트 2). 손해가 이득으로 전환. 예측값 430초와 일치. 수준 1은 재사용과 1회성이 섞인 배치에서 무효(1회성 요청은 로드가 없어 대상이 아님). 수준 2는 커넥터 적중 조회 뒤에 검사해야 하며(앞에 두면 로드가 하나씩만 시작됨), 로드 중 여부는 큐 위치가 아니라 요청 ID 집합으로 추적.

#### 저장 정책과 양자화

문헌(Marconi, HotPrefix, Baleen, S3-FIFO)의 공통 출발점인 한 번만 보이는 항목 배제를 적용. 두 번째 관측부터 저장하는 seen_twice(lib/value_admission). 게이트 없음, 섞인 워크로드 3라운드.

| 라운드 | 재계산 | 전부 저장 | seen_twice | forward |
|---|---|---|---|---|
| 1 | 449.8 s | 447.6 s | 445.6 s | 32 / 32 / 32 |
| 2 | 449.0 s | 482.5 s | 445.4 s | 32 / 35 / 32 |
| 3 | 449.1 s | 482.6 s | 481.8 s | 32 / 35 / 35 |
| 합계 | 1,347.9 s | 1,412.7 s | 1,372.8 s | |

- seen_twice는 라운드 1에 저장을 안 해 라운드 2가 재계산과 같고 실제로 읽는 라운드 3에서 같은 33초 손해. 손해가 절반이 된 것은 읽기를 피한 덕분이지 원인을 고친 것이 아님. 디스크는 39.4 GiB 대 3.9 GiB로 10분의 1이지만 쓰기가 공짜라 wall clock에 안 나타남.
- 게이트로 로드가 이득이 된 뒤에는 첫 적중을 포기하는 만큼 seen_twice가 손해. 저장 정책의 가치는 스케줄러 상태에 종속.
- staged 경로에서는 링 슬롯 6개와 CPU 경유 8개로도 저장 대상 청크의 절반이 슬롯 압박으로 버려짐. 전송 구현을 staged로 바꿔 전부 저장해도 손해 33초로 동일.
- 어텐션 기반 토큰 선별(H2O, SnapKV, PyramidKV)은 저장 필터로 쓸 수 없음. SCBench와 KVzip이 재사용 시 붕괴를 실증. 쿼리 무관 선별(KVzip 정적 헤드 모드, DuoAttention)과 양자화만 저장 시점에 안전.

양자화는 저장 계층에만 넣었다(lib/expfs.py CuFileQ8Transport, 설정 cufile_q8). 토큰 하나의 K 또는 V 벡터 9216개마다 absmax 스케일 하나로 int8 저장. GPU 여유가 200 MiB 남짓이라 256행 타일로 처리하며 버퍼는 스레드마다 한 번만 잡음. 단독 왕복 테스트에서 파일 크기 비율 0.500, 복원 오차 absmax 대비 0.4%, 추가 GPU 할당 11 MiB.

| 라운드 | 재계산 | fp16 + 게이트 2 | int8 + 게이트 2 |
|---|---|---|---|
| 2 | 440.5 s | 437.8 s | 439.2 s |
| 3 | 440.6 s | 437.9 s | 439.2 s |

디스크는 39.4 GiB에서 19.7 GiB로 절반이 됐지만 wall clock는 라운드당 1.4초 느려짐. 게이트 뒤 남은 로드 대기가 배치당 1초 남짓이라 바이트를 줄여 얻을 상한이 그 정도였고 변환 비용이 상쇄. 출력 품질은 미측정(입력이 난수 토큰). 양자화는 디스크 용량이 제약일 때의 수단이지 이 환경의 지연 개선 수단은 아님.

#### 문헌과의 관계

- Bidaw(FAST 2026): 느린 계층 I/O가 끝나지 않은 요청이 준비된 요청을 막는 문제를 dual queue로 분리. 문제 정의가 우리 관측과 같음. 가중치는 GPU 상주 가정.
- Strata(OSDI 2026): 캐시 로딩 지연을 모르는 스케줄러가 시스템을 loading-bound로 만든다는 문제 정의. balanced batch와 bubble filling. 게이트 수준 2가 balanced batch의 가장 단순한 형태.
- Revisiting Pipeline Parallelism for LLM Serving(OSDI 2026): 파이프라인 병렬 주제라 다르지만 배치를 어긋나게 만드는 요청을 보류하고 보류 중 KV를 할당하지 않는 delay scheduling이 같은 지렛대.
- AttentionStore(ATC 2024)의 층 단위 선반입은 프리픽스 적중에서 새 입력이 0이라 필요 버퍼가 KV 전체로 발산해 적용 불가. Bidaw가 같은 결론을 실측, Tutti는 SSD에서 층 단위가 버블을 오히려 늘린다고 보고.
- IMPRESS(FAST 2025)는 디스크의 프리픽스 KV를 읽는 것이 항상 TTFT를 줄이지 않는다고 문제 정의. Baleen(FAST 2024)은 적중률 최적화가 종단 성능을 해쳐 목표를 디스크 점유 시간으로 바꿈. 우리가 슬롯을 늘려 적중을 올리자 느려진 것과 같은 실패.
- Cake(ICML 2025)의 양방향 로드와 재계산은 계산과 I/O가 독립 자원이라는 전제가 우리 환경에서 깨짐.
- 가중치 경로 선행: FlexGen(디스크에서 CPU 경유), DeepSpeed DeepNVMe와 Endor와 TERAIO(GDS 사용), CHEOPS 2025 I/O 특성 연구(bounce buffer 병목 진단). 새로운 부분은 vLLM prefetch 오프로더 위에 host 비율이 파라미터인 SSD 티어를 넣고 같은 코드에서 POSIX와 cuFile을 대조한 것.
- vLLM 이슈 41784가 디스크에서 KV를 읽을 때 GPU가 유휴가 되는 같은 증상. 미해결. RFC 43702는 폴링 구조가 의도된 설계임을 명문화.
- 가중치 스트리밍과 KV 로딩이 스케줄러를 통해 서로를 악화시키는 조합을 정면으로 다룬 탑티어 논문은 없음. 이 빈틈이 기여의 자리. 다만 실측 이득이 2.5%로 작고 하드웨어가 한 장이라 일반성 주장은 약함.

#### 워크로드와 측정 방법

입력은 난수 토큰 ID 열(맨 앞 토큰 2, 나머지 4~49,999 난수). 재는 것이 I/O와 스케줄링 비용이라 문장 내용은 무관하고, 프롬프트끼리 겹치면 GPU 프리픽스 캐시가 먼저 맞아 SSD 경로를 안 타므로 시드를 다르게. 정확성은 라운드 간 출력 토큰 일치로 확인. 품질 평가는 이 입력으로 불가.

| 항목 | 06 | 07 | 09 |
|---|---|---|---|
| 프롬프트 | 4개 × 256 | 8개 × (448 + 32) | 16개 × (448 + 32) |
| GPU KV | 자동 배분 5.55 GiB | 1.5 GiB | 5.5 GiB |
| 동시 요청 | 4 (한 배치) | 1 | 5 |
| GPU 상주 층 | 3 | 4 | 2 |
| decode | 8 | 8 | 8 |
| 라운드 | 1 | 2 | 2~3 |

448과 32는 64토큰 블록 정렬 때문(꽉 찬 7블록만 저장). 480토큰은 요청당 KV가 약 1 GiB가 되어 5.5 GiB에 5개가 한 배치가 되는 길이. 1,700토큰이면 요청당 3.6 GiB라 상주 층을 다 내려 KV를 10 GiB로 키워도 2~3개뿐이어서 배치 분할을 보기 어려움. admission 실험은 16개 중 4개만 라운드마다 반복하고 12개는 매번 새 토큰열로 재사용을 치우침.

시간은 단조 시계로 wall clock, getrusage로 CPU 시간. 엔진 step을 직접 돌려 step마다 출력 요청 수로 prefill과 decode를 분류하고 1초 미만 step은 폴링으로 분리. nsys는 확인용으로 한 번(NVTX 표식으로 step, 라운드, KV 읽기와 쓰기), 기록 170 MB. 경로 확인은 cuFile TRACE 로그, nvidia-fs 통계, dmesg.

#### 파일과 재현

| 경로 | 내용 |
|---|---|
| experiments/06-weight-offload | run_66b.py와 run_66b.sh(가중치 경로 매트릭스), campaign.sh, run_qa.sh와 smoke_*.py(opt-2.7b QA), repro_wrap.sh(prefetch 버그 재현), summarize_66b.py |
| experiments/07-combined | run_combo_66b.py(결합 러너, generate 두 번 방식이라 decode 지표 불신), campaign07.sh, memguard.sh, pin_exact_test.py, summarize_07.py |
| experiments/08-cufile-bounce | cufile-pb{4096,8192,16384}.json, check_props.py(적용된 조각 크기 확인), bench_bounce.py(읽기 패턴 마이크로벤치), campaign08.sh와 b~e, summarize_08.py |
| experiments/09-kv-policy | run_phase_66b.py(구간 계측 러너, staged와 q8 전송, 반복과 1회성 워크로드), run_policy_66b.py(초기 러너), campaign09.sh와 09b, 09c(배치 구성, 정책, 프리픽스 1700), campaign10.sh(비용 분해), 11과 12(구간 분리), 13(admission), 14~16(게이트), 17(양자화) |
| results/weight-offload/opt66b | 06 결과 json과 로그, nsys csv |
| results/combined | 07 결과 |
| results/cufile-bounce | 08 결과, verify 로그 |
| results/kv-policy | 09 결과, campaign09.log, nsys 기록 |
| tools/compare_results.py | 06, 07, 09 결과 json 일람, forward 고정비 모형 대조, 기준 런 대비 변화율 |
| experiments/10-model-host-baseline | campaign_baseline.sh(모델 × host 비율, 실제 문서, 배치 2). 러너는 09의 run_phase_66b.py에 --prompt-source leval, --host-weight-fraction, --kv-batch, --poll-sleep-ms 추가 |
| results/model-host-baseline | 기준표 결과 json과 로그, campaign.log, RAM 0.1 점(ram0.1), 이중 버퍼 비교(b1-step1, b1-step2) |
| tools/baseline_table.py | 기준표 결과를 모델 × host 표로(적중 라운드, 저장 추가, 두 라운드 합계) |
| lib/expfs.py | CuFileQ8Transport 추가 |
| ~/vllm weight-ssd-offload | SSD 티어(2fbceeb103, 1c86373b60, fbfc637cb0), prefetch 경계 수정(3fc4433b62), 정확 등록(2f050f7fd5), 스케줄러 게이트(2998fcca0b, b576070a77), 등록 총량 상한(929df037b9) |

재현 환경: env.sh 소싱, VLLM_USE_V2_MODEL_RUNNER=0, VLLM_ENABLE_V1_MULTIPROCESSING=0, host 0.85 이상은 VLLM_OFFLOAD_PIN_EXACT=1, cuFile 조각은 CUFILE_ENV_PATH_JSON으로 4 MiB 설정, 게이트는 VLLM_KV_LOAD_WAVE_GATE=2. 런마다 SSD 티어와 KV 저장소를 지우고 다시 만들므로 디스크 여유 90 GB 이상 필요. 캠페인은 setsid nohup으로 띄우고 성공 판정은 결과 json 존재로.

교훈. 1 MiB cuFile 조각 구성은 단독 측정으로 결론 내지 말 것. 실험 폴더에 보고서를 따로 두지 말고 이 문서와 README에만 적을 것.

### 모델 크기와 host 비율 기준표 (실제 문서, KV는 SSD)

앞 절까지의 66B 결과는 전부 난수 토큰이었고 host 비율도 0.85 한 점이 중심이었다. 기준을 세우려고 실제 문서 프롬프트로 바꾸고 모델 크기와 host 비율을 축으로 forward 고정비, 적중 이득, 저장 비용을 한 표에 놓았다. 실험 폴더 experiments/10-model-host-baseline, 결과 results/model-host-baseline.

#### 설계

- 가중치는 GPU 상주 0층, pinned CPU 티어, 나머지 SSD 티어. host 비율은 오프로드되는 가중치 대비(RAM 대비가 아님). 오프로더의 offload_host_fraction은 RAM 전체 대비라 작은 모델은 어떤 값을 줘도 전부 CPU에 들어가므로, 러너가 모델 크기(12 d² × 층 수 × 2바이트)로 환산해 넘긴다.
- KV는 GPU 작업 공간 밖으로 전부 SSD(expfs cuFile, 4 MiB 조각). GPU KV 예산은 요청 2개분(max_model_len 2,048 토큰 × 1.15)으로 모델마다 계산해 배치 2로 고정. 예산이 크면 문서 8개의 KV가 GPU에 남아 2라운드가 SSD를 읽지 않는다(opt-2.7b 시험 실행에서 확인).
- 프롬프트는 03-leval 실제 문서 8개. 프리픽스 1,920토큰 + 구분자 + 질문. 1라운드는 질문 1로 저장, 2라운드는 질문 2로 프리픽스 적중. 재계산 조건은 저장 없음. decode 8토큰, 스케줄러 게이트 2.
- 측정은 run_phase_66b.py 그대로. forward 시간은 출력이 있는 step 길이의 중앙값(러너의 phase 라벨은 첫 토큰을 기다리는 요청이 남아 있으면 decode forward도 prefill로 적으므로 평균을 쓰지 않음). prefill forward는 출력 0인 첫 step. 로드 대기는 실제 step 사이의 빈 시간.

#### OPT-66B, host 0.7에서 0.1

| host | CPU / SSD 티어 | forward 실측 / 모형 | prefill 재계산 / 적중 | 배치당 로드 대기 | 적중 라운드 wall clock, 재계산 / 적중 | 저장 라운드 추가 | 두 라운드 합계, 재계산 / 저장+적중 |
|---|---|---|---|---|---|---|---|
| 0.7 | 44층 83.5 GiB / 20층 38.0 GiB | 19.37 / 19.14 s | 34.97 / 19.73 s | 2.7 s | 683.1 / 633.4 s (-7.3%) | +6.2 s | 1365 / 1323 s (-3.1%) |
| 0.5 | 31층 58.9 GiB / 33층 62.7 GiB | 24.70 / 24.70 s | 40.28 / 25.08 s | 3.4 s | 853.6 / 807.4 s (-5.4%) | +25.1 s | 1707 / 1686 s (-1.2%) |
| 0.3 | 19층 36.1 GiB / 45층 85.4 GiB | 29.76 / 29.82 s | 45.30 / 30.28 s | 2.9 s | 1015.5 / 970.9 s (-4.4%) | +45.3 s | 2030 / 2032 s (+0.1%) |
| 0.1 | 6층 11.4 GiB / 58층 110.1 GiB | 35.68 / 35.37 s | 51.16 / 35.96 s | 3.0 s | 1204.5 / 1153.2 s (-4.3%) | +66.2 s | 2409 / 2424 s (+0.6%) |

- forward 고정비는 네 비율 모두 모형(CPU 티어/12.3 GB/s + SSD 티어/3.44 GB/s)과 1% 안. 06에서 host 0.1이 모형보다 68% 느렸던 것은 이번에 재현되지 않았고, 06은 1 MiB 조각이었으므로 그 느린 모드로 본다.
- 적중이 아끼는 것은 prefill forward의 토큰 계산 몫이고 세 비율 모두 15초(3,940토큰). host 비율은 forward 고정비만 바꾸므로 이득의 절대량은 같고 비율만 줄어든다. 448토큰 난수 프롬프트에서 5초였던 몫이 1,970토큰 실제 문서에서 15초. 프리픽스가 길수록 SSD 적중이 유리하다는 01의 결론이 스트리밍 조건에서도 성립.
- 배치당 로드 대기 2.7~3.4초는 비율과 무관. 8.4 GiB를 배치 경계에서 읽고 앞 배치의 decode와 겹치지 않는다. 앞 배치 중에 로드를 시작하면 이득이 토큰 계산 몫 전부로 커진다. 미구현.
- 저장 라운드의 추가 시간은 위치가 잡혔다. 배치마다 prefill 직후 첫 decode forward 하나만 늘고 나머지 step은 재계산과 같다. 그 step 동안 방금 prefill한 프리픽스 8.4 GiB가 SSD에 쓰이고 같은 forward가 SSD에서 가중치를 읽는다. 늘어난 시간이 host 0.7에서 1.2초, 0.3에서 10초로 SSD 가중치 몫에 비례. KV 쓰기와 가중치 읽기의 디스크 공유가 step 단위로 관측된 첫 사례이고, 09에서 decode 구간 겹침 0초로 기각한 것은 읽기 쪽이었다. 저장을 decode 뒤나 CPU 구간으로 미루는 deferred_store(04)가 대응책이며 66B에서는 미측정.
- 저장까지 넣은 두 라운드 합계가 순이익. host 0.7에서 −3.1%, 0.5에서 −1.2%, 0.3과 0.1에서는 0. 재사용이 한 번이면 host 0.3 이하에서 저장 손해와 적중 이득이 상쇄되고 두 번 이상 재사용돼야 남는다. 가중치의 SSD 몫이 클수록 KV 오프로드의 순이익이 줄고 원인은 디스크 공유.

#### opt-6.7b, 13b, 30b, host 1.0에서 0.1

같은 설계로 작은 모델 세 개. 13b와 30b는 정적 버퍼 등록 상한 100 MiB(아래 BAR1 항목). host 1.0은 SSD 없이 GPU+CPU.

| 모델 | host | CPU / SSD layer (GiB) | forward 실측 / 모형 | prefill 재계산 / 적중 | 배치당 로드 대기 | 적중 라운드 wall clock, 재계산 / 적중 | 저장 라운드 추가 | 두 라운드 합계, 재계산 / 저장+적중 |
|---|---|---|---|---|---|---|---|---|
| 6.7b | 1.0 | 32 (12.0) / 0 | 1.12 / 1.05 s | 3.52 / 1.14 s | 0.8 s | 45.6 / 39.3 s (-14.0%) | +0.6 s | 91 / 86 s (-6.5%) |
| 6.7b | 0.7 | 22 (8.2) / 10 (3.8) | 2.23 / 1.89 s | 4.50 / 2.20 s | 5.6 s | 80.7 / 92.5 s (+14.6%) | +0.3 s | 161 / 174 s (+8.0%) |
| 6.7b | 0.5 | 15 (5.6) / 17 (6.4) | 2.99 / 2.48 s | 5.25 / 3.01 s | 4.2 s | 104.8 / 112.3 s (+7.1%) | +5.1 s | 211 / 222 s (+5.1%) |
| 6.7b | 0.3 | 9 (3.4) / 23 (8.6) | 3.62 / 2.99 s | 5.80 / 3.59 s | 4.9 s | 125.0 / 133.0 s (+6.5%) | +7.8 s | 248 / 266 s (+7.3%) |
| 6.7b | 0.1 | 3 (1.1) / 29 (10.9) | 4.21 / 3.49 s | 6.35 / 4.19 s | 2.3 s | 143.1 / 142.6 s (-0.3%) | +44.3 s | 287 / 330 s (+15.1%) |
| 13b | 1.0 | 40 (23.4) / 0 | 2.18 / 2.05 s | 6.36 / 2.24 s | 4.6 s | 86.6 / 88.1 s (+1.7%) | +1.6 s | 175 / 176 s (+0.5%) |
| 13b | 0.7 | 28 (16.4) / 12 (7.0) | 3.73 / 3.63 s | 7.82 / 3.83 s | 1.3 s | 136.0 / 125.5 s (-7.8%) | +1.0 s | 272 / 263 s (-3.4%) |
| 13b | 0.5 | 19 (11.1) / 21 (12.3) | 4.92 / 4.81 s | 9.01 / 5.04 s | 2.3 s | 174.2 / 168.0 s (-3.5%) | +8.2 s | 348 / 350 s (+0.6%) |
| 13b | 0.3 | 11 (6.5) / 29 (17.0) | 5.96 / 5.87 s | 10.03 / 6.02 s | 1.8 s | 207.2 / 197.0 s (-4.9%) | +18.6 s | 414 / 423 s (+2.1%) |
| 13b | 0.1 | 4 (2.3) / 36 (21.1) | 6.87 / 6.79 s | 10.91 / 6.99 s | 2.2 s | 236.4 / 229.8 s (-2.8%) | +71.6 s | 473 / 538 s (+13.8%) |
| 30b | 1.0 | 48 (55.1) / 0 | 5.08 / 4.81 s | 13.08 / 5.20 s | 2.9 s | 194.9 / 174.5 s (-10.4%) | +4.7 s | 395 / 374 s (-5.2%) |
| 30b | 0.7 | 33 (37.9) / 15 (17.2) | 8.92 / 8.69 s | 16.74 / 9.01 s | 2.8 s | 317.3 / 295.6 s (-6.9%) | +3.1 s | 634 / 616 s (-2.9%) |
| 30b | 0.5 | 23 (26.4) / 25 (28.7) | 11.37 / 11.27 s | 19.18 / 11.54 s | 4.3 s | 395.7 / 382.7 s (-3.3%) | +11.2 s | 791 / 790 s (-0.2%) |
| 30b | 0.3 | 14 (16.1) / 34 (39.0) | 13.71 / 13.59 s | 21.53 / 14.09 s | 2.4 s | 470.7 / 455.8 s (-3.2%) | +32.8 s | 941 / 959 s (+1.9%) |
| 30b | 0.1 | 4 (4.6) / 44 (50.5) | 16.65 / 16.18 s | 24.43 / 16.78 s | 3.0 s | 564.9 / 543.3 s (-3.8%) | +51.6 s | 1154 / 1160 s (+0.5%) |

- forward는 13b 이상에서 모형과 1~6% 안. 6.7b는 7~20% 벗어나고 SSD layer가 많을수록 커짐. layer가 0.38 GiB로 작으면 SSD 읽기가 유효 2.9 GB/s에 그침. 모형의 SSD 대역폭은 layer 크기에 따라 달리 잡아야 함.
- 적중이 아끼는 prefill 몫은 모델 크기에 비례하고 host 비율과 무관. 6.7b 2.3초, 13b 4초, 30b 8초, 66B 15초. 같은 절대 이득을 host 비율이 높을수록 짧은 라운드에서 나누므로 wall clock 이득의 비율은 host 비율이 높을수록 큼.
- 저장 라운드 추가 시간은 SSD 티어가 클수록 커져 host 0.1에서 44~72초. 66B와 같은 디스크 공유. 두 라운드 합계로는 30b가 host 1.0과 0.7에서 −5.2%와 −2.9%, 13b는 0.7에서 −3.4%만 이득이고 host 0.3 이하는 전 모델에서 손해. 6.7b는 host 1.0 외에는 손해.

#### host memory 비율로 다시 본 표

위 표의 host는 오프로드 가중치 대비였다. host memory(RAM 125.5 GiB) 대비 비율로 보면 예산이 0.1에서 12.5 GiB, 0.3에서 37.6 GiB, 0.5에서 62.7 GiB, 0.7에서 87.8 GiB이고, 그 예산을 넘는 layer가 SSD로 간다. 30b 이하는 0.3 또는 0.5부터 전부 CPU라 상한이 의미 없고, 66B는 121.5 GiB라 어떤 비율에서도 SSD 티어가 생긴다. 비는 칸은 RAM 0.1 점을 따로 측정(가중치 0.227과 0.533)했고, 나머지는 layer 수가 같은 기존 런을 대응시켰다.

| 모델 | RAM 비율 | CPU / SSD layer (GiB) | forward 실측 / 모형 | 적중 라운드 | 저장 라운드 추가 | 두 라운드 합계 | 대응 런 |
|---|---|---|---|---|---|---|---|
| 6.7b | 0.1 | 32 (12.0) / 0 (0.0) | 1.12 / 1.05 s | 46 → 39 s (-14.0%) | +1 s | 91 → 86 s (-6.5%) | 전부 CPU |
| 6.7b | 0.3 | 32 (12.0) / 0 (0.0) | 1.12 / 1.05 s | 46 → 39 s (-14.0%) | +1 s | 91 → 86 s (-6.5%) | 전부 CPU |
| 6.7b | 0.5 | 32 (12.0) / 0 (0.0) | 1.12 / 1.05 s | 46 → 39 s (-14.0%) | +1 s | 91 → 86 s (-6.5%) | 전부 CPU |
| 6.7b | 0.7 | 32 (12.0) / 0 (0.0) | 1.12 / 1.05 s | 46 → 39 s (-14.0%) | +1 s | 91 → 86 s (-6.5%) | 전부 CPU |
| 13b | 0.1 | 21 (12.3) / 19 (11.1) | 4.76 / 4.55 s | 169 → 166 s (-2.0%) | +15 s | 337 → 350 s (+3.6%) |  |
| 13b | 0.3 | 40 (23.4) / 0 (0.0) | 2.18 / 2.05 s | 87 → 88 s (+1.7%) | +2 s | 175 → 176 s (+0.5%) | 전부 CPU |
| 13b | 0.5 | 40 (23.4) / 0 (0.0) | 2.18 / 2.05 s | 87 → 88 s (+1.7%) | +2 s | 175 → 176 s (+0.5%) | 전부 CPU |
| 13b | 0.7 | 40 (23.4) / 0 (0.0) | 2.18 / 2.05 s | 87 → 88 s (+1.7%) | +2 s | 175 → 176 s (+0.5%) | 전부 CPU |
| 30b | 0.1 | 10 (11.5) / 38 (43.6) | 15.17 / 14.63 s | 517 → 496 s (-4.0%) | +38 s | 1034 → 1051 s (+1.7%) |  |
| 30b | 0.3 | 33 (37.9) / 15 (17.2) | 8.92 / 8.69 s | 317 → 296 s (-6.9%) | +3 s | 634 → 616 s (-2.9%) | 가중치 0.7 런, 예산보다 layer 하나 많음 |
| 30b | 0.5 | 48 (55.1) / 0 (0.0) | 5.08 / 4.81 s | 195 → 175 s (-10.4%) | +5 s | 395 → 374 s (-5.2%) | 전부 CPU |
| 30b | 0.7 | 48 (55.1) / 0 (0.0) | 5.08 / 4.81 s | 195 → 175 s (-10.4%) | +5 s | 395 → 374 s (-5.2%) | 전부 CPU |
| 66B | 0.1 | 6 (11.4) / 58 (110.1) | 35.68 / 35.37 s | 1205 → 1153 s (-4.3%) | +66 s | 2409 → 2424 s (+0.6%) | 가중치 0.1 런, layer 수 동일 |
| 66B | 0.3 | 19 (36.1) / 45 (85.4) | 29.76 / 29.82 s | 1015 → 971 s (-4.4%) | +45 s | 2030 → 2032 s (+0.1%) | 가중치 0.3 런, layer 수 동일 |
| 66B | 0.5 | 33 (62.7) / 31 (58.9) | 23.86 / 23.84 s | 827 → 776 s (-6.1%) | +28 s | 1653 → 1631 s (-1.3%) | 별도 측정(가중치 0.516) |
| 66B | 0.6 | 39 (74.0) / 25 (47.5) | 21.40 / 21.28 s | 748 → 696 s (-6.9%) | +15 s | 1496 → 1460 s (-2.4%) | 별도 측정(가중치 0.620) |
| 66B | 0.7 | 46 (87.3) / 18 (34.2) | 18.54 / 18.29 s | 657 → 605 s (-7.9%) | +24 s | 1312 → 1285 s (-2.1%) | 별도 측정(가중치 0.723) |
| 66B | 0.8 | 52 (98.7) / 12 (22.8) | 16.07 / 15.73 s | 578 → 527 s (-8.8%) | +14 s | 1155 → 1118 s (-3.2%) | 별도 측정(가중치 0.826) |

- 66B RAM 0.5, 0.6, 0.7, 0.8은 가중치 비율로 환산(0.516, 0.620, 0.723, 0.826)해 별도 측정. 위 표. forward는 네 점 모두 모형과 2% 안. 적중 라운드 이득이 −6.1%에서 −8.8%로, 저장 손해가 28초에서 14초로 줄어 한 사이클 순이익이 −1.3%에서 −3.2%로 커짐. RAM 비율이 높을수록 KV 오프로드가 유리하고, 그래도 3% 안.
- 0.7 적중 런의 저장 라운드에서 nvme0n1 1초 샘플(/proc/diskstats)로 디스크 공유를 직접 확인. 배치마다 KV 8.3 GiB 쓰기가 10~12초 동안 712~855 MB/s로 나가고, 그 초들에서 가중치 읽기가 3,213 MB/s에서 223~383 MB/s로 떨어지며 읽기 지연이 6.3 ms에서 13~17 ms로 오름. 쓰기가 없는 초는 읽기 3.2 GB/s, 지연 6.3 ms.
#### 전송과 계산의 겹침: prefetch_step 2

오프로더의 정적 버퍼는 slot 수가 prefetch_step과 같다. 기본 1이면 한 세트라 layer l의 forward가 끝나야 l+1의 전송이 시작되어 prefill이 전송 + 계산의 합이 된다(66B host 0.7에서 19.4 + 15.6 = 35 s). 2로 두면 두 세트(66B 4.08 GB)가 되어 layer l 계산 중 l+1 전송이 돈다.

16 GB 카드에서는 버퍼 두 세트(4.08 GB)와 요청 2개분 KV를 같이 넣을 수 없어 배치 1(KV 5.2 GiB), 재계산만, 한 라운드로 비교.

| 조건 (66B RAM 0.7, 배치 1, 요청당 1,970토큰) | prefill forward | decode forward | 라운드 wall clock (8요청) |
|---|---|---|---|
| prefetch_step 1 | 25.5 s | 18.3 s | 1,233 s |
| prefetch_step 2 | 17.8 s | 17.6 s | 1,128 s (−8.5%) |

- step 1의 prefill 25.5 s는 전송 18.3 s + 1,970토큰 계산 7.2 s. step 2에서는 prefill이 decode와 같은 17.8 s로 계산이 전송 아래 전부 숨음. decode도 layer마다의 동기화 틈이 사라져 0.7 s 짧아짐.
- 따라서 이 조건에서 재계산의 추가 비용은 이중 버퍼만 켜면 0에 가깝고, SSD KV 적중이 아낄 몫도 함께 사라진다. 기준표의 적중 이득(−4~−9%)은 오프로더가 전송과 계산을 직렬로 돌린 상태에서만 성립한 값. 저장 손해(SSD 몫에 비례)는 그대로이므로 이중 버퍼 조건에서는 KV 오프로드가 순손해가 될 가능성이 크며, 배치 2에서 GPU 메모리가 허락하는 카드에서 다시 재야 확정.
- 결론 순서. 가중치 스트리밍 조건에서는 오프로더의 이중 버퍼가 KV 오프로드보다 먼저이고, 대가는 GPU 메모리 layer 한 세트(66B 1.9 GiB). 그 뒤에 남는 KV 오프로드의 역할은 이중 버퍼를 켤 GPU 메모리가 없는 경우로 좁혀진다.


#### 기본값 런과 native backend, LMCache 비교 시도

관측 계층(lib/obs)과 in-tree CuFileFsSpec(C++)으로 66B RAM 0.5를 손대지 않은 설정(게이트 없음, cuFile 기본 json 1 MiB, KV 예산 vLLM 자동, 폴링 양보 없음)에서 재계산과 SSD 적중을 비교. 워크로드는 LEval 문서 8개, cold_fill → settle 15초 → reverse_retrieve(역순, 다른 질문), decode 8. 결과 results/native-66b/pure-ram0.5-*.

| 단계 | 재계산 | native SSD 적중 |
|---|---|---|
| cold_fill(저장) wall clock | 846 s | 1,061 s (+25%) |
| cold_fill prefill forward | 32.3 s | 41.5 s |
| reverse_retrieve(적중) wall clock | 828 s | 931 s (+12%) |
| reverse_retrieve forward 수 | 32 | 38 |
| reverse_retrieve prefill forward | 30.2 s | 24.6 s |
| 두 단계 합계 | 1,674 s | 1,992 s (+19%) |

- 출력 토큰은 16요청 전부 재계산과 동일, native 오류 0. vLLM 자동 KV는 10.4 GiB(동시 2요청)이며 gpu_util 0.9에서는 첫 prefill이 OOM이라 0.85로 재시도한 값(캠페인 스크립트가 자동 재시도하고 기록).
- 손해 자리는 앞 절과 같음. 저장 단계는 KV 쓰기와 가중치 SSD 읽기의 디스크 공유(1 MiB 조각이라 4 MiB 때보다 큼), 적중 단계는 게이트가 없어 forward 6개 증가. 같은 조건에 게이트와 4 MiB 조각만 넣으면 순이익 −1.3%였으므로 기본값과 손본 설정의 차이가 20%.
- 기본값에서는 prefill이 요청 2개를 한 forward로 묶지 않고 요청마다 따로 돌아(기본 배치 토큰 상한) prefill forward가 32초짜리 둘.
- write-behind(cufile_fs_store_window=host: 가중치 오프로더가 SSD 티어 layer를 읽는 동안 KV 쓰기 스레드를 멈추고 host 티어 layer 구간에 재개, 상한 초과 시 강제 재개)만 켠 같은 조건. 상한 10 s는 RAM 0.5의 SSD 구간 18 s(layer 33~63 연속)보다 짧아 forward마다 강제 재개가 남았고, 상한 30 s에서 forward 안에서는 안 풀림.

| 조건 | cold_fill(저장) | 저장 단계 prefill 평균 / decode 최대 | reverse(적중) / forward 수 | 두 단계 합계 |
|---|---|---|---|---|
| 재계산 | 846 s | 32.3 / 24.7 s | 828 s / 32 | 1,674 s |
| SSD 적중, write-behind 없음 | 1,061 s | 41.5 / 53.3 s | 931 s / 38 | 1,992 s (+19.0%) |
| write-behind 상한 10 s | 906 s | 39.4 / 24.7 s | 940 s / 38 | 1,846 s (+10.3%) |
| write-behind 상한 30 s | 892 s | 36.5 / 24.6 s | 935 s / 38 | 1,827 s (+9.1%) |

- 쓰기가 어느 forward에 떨어졌는지(step별 KV 쓰기 바이트). 저장 단계의 한 배치는 요청 A prefill 40 s → 요청 B prefill 24.5 s → decode 7회 → KV 8.4 GiB 쓰기 제출. write-behind 없음: 다음 배치의 B prefill(24.5 → 29.5 s, 안에서 8.5 GiB) 과 decode(24.7 → 53.3 s)에 떨어짐. write-behind 30 s: B prefill과 decode는 재계산과 같고(0.1 GiB), 다음 배치의 A prefill(40.5 → 50.5 s, 안에서 7.3 GiB)에 떨어짐. 세 배치 모두 50.4, 50.9, 50.6 s. 즉 규칙은 쓰기를 decode forward에서 빼내 다음 배치의 긴 prefill forward로 옮긴 것이고 배치당 손해가 약 30 s에서 10 s. 출력 토큰열 세 런 모두 재계산과 동일, 오류 0.
- 한 배치의 쓰기가 RAM 구간(forward당 5.5 s) 하나에 못 들어가는 이유는 쓰기 속도. 스레드당 111~188 MB/s(4스레드 합 0.5~0.7 GB/s)라 8.4 GiB에 스레드 시간 15 s가 들고, 같은 구조의 읽기는 스레드당 844 MB/s(합 3.3 GB/s). 원인은 아래 절.
- 적중 단계의 forward 6개 증가는 게이트 몫이며 write-behind와 게이트를 같이 켠 조합은 측정하지 않음(66B 종료).
- LMCache 0.5.5의 GDS L1(--gds-l1-path, DRAM 층 없이 cuFile로 GPU↔NVMe)을 같은 조건의 비교 상대로 시도. 이 카드에서 성립하지 않음. (1) LMCache가 등록하는 GPU staging 버퍼가 chunk KV × 4라 BAR1 256 MiB 안에 들어가려면 opt-2.7b는 chunk 64토큰, 66B는 16토큰 이하여야 함(기본 256에서 cuFileBufRegister 5036). (2) chunk를 줄이고 오프로더를 끈 opt-2.7b 격리 시험에서 저장은 정상(GDS 쓰기 4.9 GB, 출력 토큰 재계산과 동일)이나 적중 시 서버의 cuFileReadAsync 읽기가 첫 요청에서 멈춰 vLLM이 서버를 불량으로 판정하고 재계산으로 우회. LMCache는 cuFile 1.15와 CUDA 13에서 검증된 async 경로를 쓰고 우리는 cuFile 1.13(CUDA 12.8). (3) 66B에서는 가중치 오프로더가 BAR1을 같이 써야 하므로 chunk 16으로도 여유 없음. 결론은 LMCache GDS L1 비교는 CUDA 13과 BAR1이 VRAM 전체인 카드(양태규 서버)에서 해야 한다는 것.

#### KV 쓰기 속도와 SSD 쓰기 상한

backend(csrc/kv_offload/cufile_fs.cpp)의 파일 1개 처리 순서는 CUDA 이벤트 대기 → 임시 파일 O_DIRECT open과 cuFileHandleRegister → layer마다 cuFileWrite(66B는 576 KiB × 64회, 파일 36 MiB) → Deregister, close, rename. stats에 구간별 스레드 시간 합과 호출 수를 추가(포크 f3bb7ad92e)하고 vLLM 없이 GPU 가짜 KV 4.5 GiB를 파일 128개로 store/load 하는 단독 벤치로 분해.

| 구간 | store (4스레드) | load (4스레드) |
|---|---|---|
| 이벤트 대기 | 0 s | 없음 |
| open + HandleRegister | 0.1 s | 0.0 s |
| cuFile 호출 합 | 30.9 s | 5.5 s |
| rename, close | 0.0 s | 0.0 s |
| 처리량 | 0.62 GB/s | 3.5 GB/s |

- 시간은 전부 cuFileWrite 안. 스레드 8개(0.38 GB/s), 파일당 4블록으로 호출 2.3 MiB(0.34 GB/s), fallocate 선할당 모두 개선 없음. 이벤트를 forward 뒤에 기록하면 대기가 그대로 쓰기 시간에 더해짐(busy 이벤트 23.9 s).
- gdsio 쓰기(-I 1, bounce, 4 worker): 1 MiB 0.28 GiB/s, 4 MiB 0.70 GiB/s. 08의 3.28 GiB/s는 읽기(-I 0)였음.
- cuFile을 빼고 host에서 dd O_DIRECT 1 MiB 4병렬로 4.5 GiB씩 연속 쓰기. 디스크 94% 사용(여유 31 GB) 상태: 1.59, 0.33, 0.32 GB/s. 40초 쉬고 1회: 1.67 GB/s. 처음 약 4 GB만 빠르고 그 뒤 0.33 GB/s로 떨어지며 쉬면 회복. 970 EVO의 SLC 쓰기 캐시가 차면 TLC 직접 쓰기로 넘어가는 동작이며 온도 48°C라 스로틀 아님. ext4는 discard 없이 마운트(주 1회 fstrim.timer).
- LMCache 실패 런의 slab 40 GB와 OPT-66B(HF 캐시 124 GB, SSD 티어 59 GB)를 지우고 fstrim 뒤 36% 사용(여유 279 GB): 1.89, 0.70, 0.70, 0.70 GB/s. 지속 쓰기 상한이 0.33에서 0.70 GB/s로 올라감. 여유 96 GB, trim 직후: 1.04, 0.45, 0.43 GB/s(trim과 겹침).
- 결론. KV 쓰기 속도는 backend 코드가 아니라 이 SSD의 지속 쓰기 상한(SLC 캐시 약 4 GB 뒤 0.3~0.7 GB/s, 빈 공간에 좌우)이 정함. 66B 한 배치의 KV 8.4 GiB는 캐시보다 커서 앞 4 GB만 빠르게 나감. 따라서 write-behind가 한 번에 내보내는 양은 4 GB 이하로 끊고 사이에 쉬는 시간을 둬야 캐시 안에서 처리되며, 근본 해결은 쓰는 양 자체를 줄이는 것(GQA 모델로 토큰당 KV 7분의 1, 저장 admission). 디스크는 20% 이상 비워 둠.

### Qwen2.5-72B-Instruct 기준선 (GQA, KV는 SSD)

#### 조건과 결과

손대지 않은 기본값(게이트 없음, cuFile 기본 json 1 MiB, GPU KV 예산 vLLM 자동, gpu_util 0.9는 첫 prefill OOM이라 캠페인이 0.85로 재시도한 값). host memory 비율 RAM 0.5(host 38 layer 62.1 GiB, SSD 42 layer 68.7 GiB, 168 파일). 입력 LongBench-v2 8건을 8,128 토큰으로 자름, decode 8, cold_fill → settle 15 s → reverse_retrieve. fp16(Turing이라 bf16 불가). 결과 results/qwen72b/ram0.5-*. 실행 experiments/11-observability/campaign_qwen72.sh.

| 조건 | cold_fill(저장) / forward | 저장 단계 prefill 평균 | reverse_retrieve(적중) / forward | 적중 단계 prefill 평균 | 두 단계 합계 |
|---|---|---|---|---|---|
| 재계산 | 1,800 s / 36 | 82 s | 1,539 s / 35 | 67 s | 3,340 s |
| SSD 적중 | 1,795 s / 36 | 82 s | 1,101 s / 38 | 29 s | 2,896 s (−13.3%) |
| SSD 적중 + write-behind 30 s | 1,773 s / 36 | 81 s | 1,076 s / 38 | 28 s | 2,849 s (−14.7%) |

- decode forward 중앙값 28.2~29.0 s, 고정비 모형(62.1 GiB/12.3 + 68.7 GiB/3.44) 26.9 s, 편차 7.5~8%. 66B(1~3%)보다 큰 편차는 GQA라 k_proj·v_proj 파일이 16 MB로 작아 SSD 티어 파일 절반이 작은 읽기인 것과 방향이 같고 미확립.
- 저장 단계 손해 0. 문서 하나의 KV가 2.6 GB(토큰당 0.33 MB)라 SSD의 SLC 쓰기 캐시(약 4 GB) 안이고 다음 문서까지 약 200 s 동안 캐시가 비워짐. 쓰기 19.9 GiB, 스레드 시간 합 67 s(이벤트 대기 32 s + cuFile 34 s), cuFile 구간 기준 4스레드 합 2.5 GB/s로 66B 때(0.5~0.7)의 4배. 가중치 읽기와 겹치는 쓰기가 문서당 1 s 안팎이라 forward가 늘지 않음. write-behind는 강제 재개 0회이고 차이 −47 s는 런 간 편차(decode 28.2 대 28.9 s) 범위.
- 적중 단계. prefill forward가 67 s에서 29 s로 줄어 가중치 고정비만 남음(8k 토큰 prefill 계산 몫 문서당 38 s가 전부 아낀 몫). 읽기 13.3 GiB, 스레드 시간 합 16 s(합 3.5 GB/s). 게이트 없이 forward 35 → 38(66B는 32 → 38).
- 출력 토큰열 16건 전부 재계산과 동일(세 조건 모두), backend 오류 0. Qwen 3B 4k 토큰에서 있던 재계산 간 불일치가 72B 8k에서는 없음.
- 66B와 다른 결론이 나온 원인은 모델 구조. 토큰당 KV가 7분의 1이라 배치당 쓰기가 SSD 캐시 안에 들어가고, 8k 토큰이라 prefill 계산 몫이 커서 적중이 아끼는 양이 큼. 가중치 고정비 자체(26.9 s)는 66B(23.8 s)와 같은 부류.
- GPU 최대 사용 14.9 GiB. 디스크는 SSD 티어 69 GB + KV 20 GB로 런 중 82% 사용. 32건은 디스크 상한 때문에 미실행.

#### nsys 타임라인: 72B SSD 적중 런

같은 조건을 NSYS=1로 다시 돌려(results/qwen72b/ram0.5-cufile-nsys, 리포트는 -nsys-nsys/timeline.1·2.nsys-rep, git 제외) cold_fill과 reverse_retrieve의 앞 12 forward를 기록. 분석은 tools/nsys_overlap.py(forward마다 가중치 cuFileRead, KV cuFileWrite/Read 합집합과 겹침).

| forward | 길이 | 가중치 cuFileRead 합집합 | KV cuFileWrite 합집합 | 쓰기와 가중치 읽기의 겹침 |
|---|---|---|---|---|
| cold_fill prefill(문서 2개째 조각) | 123 s | 24 s | 1.1~1.4 s | 0 |
| cold_fill decode | 28~30 s | 22~24 s | 0 | 0 |
| reverse prefill(적중) | 29 s | 22~23 s | 0 | 0 |

- decode forward 29.1 s의 내부. 0.0 s에 host layer 1~37의 prefetch가 한꺼번에 발행되고 0.1 s에 SSD layer 38의 prefetch, 실제 SSD cuFileRead는 5.6 s에 시작해 29.1 s에 끝남(168 파일, 68.7 GiB, 3.1 GB/s). 즉 host 구간 5.6 s + SSD 구간 23.5 s이고 고정비 모형과 같음.
- KV 쓰기는 prefill step이 끝나며 제출되어 다음 step의 0.0~1.1 s에 512 파일이 모두 나감. 그 구간은 host 구간이라 가중치 SSD 읽기(5.6 s부터)와 안 겹침. 66B는 한 배치 쓰기가 15 s 넘게 걸려 SSD 구간까지 밀렸던 것이고, 72B는 문서당 1 s라 자연히 host 구간 안에 끝남. 규칙으로 쓰면 쓰기 묶음이 host 구간(5~6 s)보다 짧으면 충돌이 없음.
- KV 읽기(적중)는 forward 사이 공백(0.3 s)과 첫 forward 전에 일어나며 wave당 약 1 s, 스레드 시간 합 4.2 s. 가중치 읽기와 겹침 0.
- ssd_window 표시는 0.1 s에 켜져 29.1 s에 꺼짐. 표시가 prefetch 발행 시점이라 실제 SSD 읽기(5.6 s)보다 5.5 s 앞서고, host 구간까지 SSD 구간으로 보고함. 이 구성에서 write-behind는 forward 내내 쓰기를 붙잡았을 것이며 강제 재개가 0회였던 것은 쓰기가 step 경계의 1 s 안에 끝났기 때문. 표시를 실제 읽기 시작에 맞추는 수정은 미적용.
- cuFileHandleNVFS(gds trace의 내부 구간) 10,096건 평균 151 ms, 합 1,527 s. cuFileRead 안에 중첩된 nvidia-fs 처리 구간으로 보이며 별도 비용인지는 미확인.

#### KV 관리 정책: LMCache 정책의 이식

LMCache 0.5.5의 정책 클래스(cache_policy LRU/LFU/FIFO, EvictionPolicy와 축출 목적지, StorePolicy, PrefetchPolicy, lazy offload)를 우리 구조로 옮긴 것. 위치는 포크 vllm/v1/kv_offload/cufile_fs/spec.py의 CuFileFsManager(scheduler 쪽 파이썬). C++ backend는 파일 이동만 하므로 무변경. vLLM in-tree의 cpu/manager.py(LRU, ref_cnt 보호)와 같은 인터페이스(lookup, touch, prepare_load/complete_load, prepare_store/evicted_keys)를 씀.

| 설정 키 | 값 | 하는 일 | LMCache 대응 |
|---|---|---|---|
| cufile_fs_capacity_gb | 0(무제한) 또는 GiB | SSD 파일 총량 상한. 완료분 + 대기 중 저장 예약분(chunk 크기 × 대기 수)이 상한을 넘으면 축출, 자리를 못 만들면 들어가는 만큼만 저장 | EvictionController 용량 |
| cufile_fs_policy | lru, lfu | lru는 마지막 접근(lookup 적중, touch, 저장) 오래된 순, lfu는 적중 횟수 적은 순(같으면 lru). 적재 중·저장 중 키는 보호 | LRUCachePolicy, LFUCachePolicy |
| cufile_fs_admission | all, never, profile, seen_twice | seen_twice는 같은 블록이 저장 후보로 두 번째 제시될 때부터 저장(04의 온라인 규칙). lookup은 첫 miss에서 멈추므로 제시 횟수로 셈 | StorePolicy(LMCache는 전부 저장) |
| cufile_fs_store_window | any, host | write-behind. SSD 구간 신호를 prefetch 발행이 아니라 SSD 티어 읽기 스레드의 실제 cuFileRead 진행(SsdTier.on_activity → IoWindow.activity, 빈틈 50 ms 유예)으로 판정하도록 수정 | lazy offload |

미이식: 프리페치 정책(SSD → host 선적재. 72B에서 wave당 1 s라 보류), host KV 층(host memory를 가중치와 나눠야 해 별도 설계).

manager stats(파일 수, 총량, 축출 수·GiB, 거부 수, lookup hit/miss, admission admit/reject)를 run_obs가 result.json kv_manager에 기록.

Qwen2.5-3B, host 0.02, Bailian 앞 24건(4k 토큰), GPU KV 9,312 토큰(kv-batch 2, GQA 반영 공식)으로 검증. 오류 0.

| 조건 | 쓰기 | 읽기 | manager |
|---|---|---|---|
| 전부 저장 | 684 파일 1.50 GiB | 577 파일 1.25 GiB | lookup hit 10,809 |
| lfu, 상한 1 GiB, write-behind | 913 파일 2.01 GiB | 321 파일 0.69 GiB | 총량 1.00 GiB 유지, 축출 458 파일 1.01 GiB |
| seen_twice | 684 파일 1.50 GiB | 66 파일 0.15 GiB | admit 684 / reject 684 (첫 제시 거부, 두 번째 저장) |

#### Qwen2.5-72B, Bailian 트레이스에서의 정책 비교

조건. 72B RAM 0.5 기본값(게이트 없음, 1 MiB 조각, GPU KV 자동 21k 토큰, gpu_util 0.85), Bailian 트레이스 offset 29,560부터 32건(hash_id → 결정적 16토큰 블록, 8,120 토큰 상한, 중앙값 6.5k, 합 154k 토큰), 창 안 프리픽스 공유 56%(트레이스 전체 66%), 고유 KV 21.7 GB. cold_fill(트레이스 순서) → settle 15 s → reverse_retrieve(역순, 다른 꼬리 8토큰). 용량 조건은 상한 8 GiB(고유량의 37%). 결과 results/qwen72b/bailian-ram0.5-*, 표는 tools/qwen72_policy_table.py.

| 조건 | cold_fill / forward / prefill 평균 | reverse / forward / prefill 평균 | 두 단계 합계 (재계산 대비) | 읽기 | 쓰기 | 축출 |
|---|---|---|---|---|---|---|
| 재계산 | 2,280 s / 54 / 62 s | 2,070 s / 53 / 55 s | 4,350 s | 0 | 0 | |
| SSD 적중, 전부 저장 | 2,355 s / 59 / 45 s | 1,677 s / 59 / 28 s | 4,032 s (−7.3%) | 20.3 GiB | 20.6 GiB | 0 |
| LFU 8 GiB | 2,262 s / 54 / 53 s | 1,934 s / 52 / 46 s | 4,196 s (−3.5%) | 2.6 GiB | 38.3 GiB | 6,201 파일 30.3 GiB |
| LRU 8 GiB | 2,260 s / 54 / 51 s | 1,989 s / 53 / 45 s | 4,249 s (−2.3%) | 2.2 GiB | 37.6 GiB | 6,058 파일 |
| seen_twice admission | 2,251 s / 54 / 51 s | 1,919 s / 55 / 32 s | 4,169 s (−4.2%) | 5.9 GiB | 20.6 GiB | 0 |

- 출력 토큰열 다섯 조건 모두 64건 재계산과 동일, backend 오류 0. decode forward 28.2~28.4 s, 고정비 모형 26.9 s(편차 5~6%).
- 전부 저장이 가장 큼. Bailian은 저장 단계 안에서도 공유 프리픽스가 적중해 prefill 평균이 62 → 45 s. 저장 단계가 +75 s인 것은 쓰기가 아니라 forward 5개 증가(게이트 없음, 적중 요청이 따로 승격). 적중 단계도 forward 6개 증가. 66B·LongBench와 같은 자리라 게이트가 회수할 몫이 약 11 forward × 28 s.
- 용량 상한 8 GiB의 LRU·LFU는 둘 다 쓰기 두 배, 읽기 8분의 1. 적중 단계가 역순 전체 재방문이라 캐시보다 큰 순차 스캔이고, 이 패턴에서 LRU는 방문 직전 항목을 지우는 최악 경우. LFU는 새로 저장된 파일(적중 0)을 먼저 지워 신규 항목이 살아남지 못함(보호 없는 LFU의 알려진 문제). 상한이 고유량보다 작으면 이 워크로드에서는 어느 쪽도 전부 저장에 못 미침. 용량 정책의 의미는 디스크 상한이 강제될 때 손실을 얼마나 줄이느냐이고, 그 답은 LFU가 LRU보다 1.2%p 나음.
- seen_twice는 첫 저장 후보를 거부하고 두 번째 제시(역순 단계) 때 저장하므로 두 단계 설계에서는 적중이 세 번째 방문에서만 생김. 쓰기 총량은 전부 저장과 같고(같은 키가 결국 저장됨) 적중 단계 이득의 절반을 잃음. 이 규칙이 이기는 조건은 04와 같이 1회성 프리픽스가 많은 긴 트레이스이며, 32건 창에서는 검증 불가.
- 저장 단계 wall clock이 LFU·LRU·seen_twice에서 재계산보다 20 s 짧은 것은 저장이 줄거나 늦어져 forward 수가 54로 유지된 것이고, 쓰기 자체의 비용은 어느 조건에서도 forward 길이에 나타나지 않음(decode forward 28.2~28.4 s 동일).
- 다음 후보. 게이트를 켠 전부 저장(예상 −11%), 축출에 신규 보호(LFU 삽입 뒤 유예 또는 2-큐)와 상한을 고유량의 60~80%로 둔 조건, 같은 창을 세 번 방문하는 설계에서 seen_twice 재평가.

### 채널 대역폭과 동시 실행 (KV 소스 섞기의 상수)

#### 측정

experiments/12-channels/bench_channels.py, 모델 없이 4 GiB 버퍼로 단독과 동시 실행 쌍을 잼. cuFile은 backend·gdsio와 같은 1 MiB 호출 4스레드(64 MiB 호출은 bounce 경로에서 1.7 GB/s로 느림). 결과 results/channels/rain.json. SSD 단독 읽기는 직전 쓰기 상태에 따라 2.9~3.6 GB/s로 흔들림.

| 채널 | 단독 |
|---|---|
| host → GPU pinned | 12.3 GB/s |
| host → GPU pageable | 11.2 GB/s |
| GPU → host pinned | 13.2 GB/s |
| SSD → GPU cuFile(bounce) | 2.9~3.6 GB/s |
| GPU → SSD cuFile | 1.5~2.1 GB/s (SLC 캐시 안) |
| GPU fp16 행렬곱 8192³ | 69 TFLOPS |

| 동시 쌍 | 각 채널의 단독 대비 |
|---|---|
| host→GPU + SSD→GPU | host 0.65~0.67, SSD 0.8~1.0 |
| host→GPU + GPU→host | 0.92 / 0.86 (전이중) |
| SSD 읽기 + SSD 쓰기 | 읽기 0.28~0.54, 쓰기 0.33~0.42 |
| host→GPU + GPU 계산 | 0.99 / 0.99 |
| SSD→GPU + GPU 계산 | 1.0 / 0.99 |
| host→GPU + SSD→GPU + GPU 계산 | 0.67 / 1.0 / 0.99 |

- GPU 계산은 어느 전송과도 서로 영향 없음. 재계산 채널은 독립.
- SSD→GPU는 이 카드에서 host bounce를 거치므로 GPU PCIe 링크를 같이 쓰며, 동시에 돌면 host→GPU가 12.3 → 8.0 GB/s로 줄고 SSD 쪽은 유지(SSD가 링크를 먼저 가져감). 두 채널의 합은 약 11.5 GB/s로 링크 한계. 교차 배치에서 72B decode forward가 5.7 s가 아니라 2 s만 준 이유의 후보이며, nsys의 weight_h2d 구간 길이로 확인할 것.
- 양방향(host→GPU와 GPU→host)은 거의 독립. KV 저장(GPU→host bounce)은 가중치 host 복사와 부딪히지 않음.
- SSD 읽기와 쓰기의 공유는 66B에서 본 3.2 → 0.3 GB/s와 같은 현상.
- pageable 메모리는 11.2 GB/s라 느린 PCIe를 흉내 내는 수단이 못 됨.

#### 소스 섞기 모형

요청 프리픽스 H 청크 중 앞 k를 재계산, 나머지를 host(h)와 SSD(s)에서 동시에 적재하면 확보 시간은 max(k·토큰/R, h·바이트/B_host', s·바이트/B_ssd'). B'는 그 시각 가중치 스트리밍이 남긴 유휴 대역폭에 위 동시 실행 저하율을 곱한 값. 72B 8k 기준 R = 215 tok/s(0.07 GB/s KV 환산), B_ssd' ≈ 3.4, B_host' ≈ 8~12 GB/s라 적재가 재계산보다 바이트당 50배 이상 빨라 k ≈ 0. 섞기가 의미를 갖는 영역은 계산이 싼 작은 모델(2.7b: SSD 0.48 s 대 재계산 1.15 s)과 토큰당 KV가 큰 MHA 모델. 구현(뒤쪽 적중 적재 + 앞쪽 재계산 동시 진행, 분할 제어기 VLLM_KV_SPLIT)은 포크 진행 중.

#### KV 소스 분할: 앞 재계산과 뒤 적재의 동시 진행

포크 구현(d3534f1751). 요청의 적중 프리픽스 H 청크 중 앞 k 청크는 GPU가 chunked prefill로 다시 계산하고, 뒤 H−k 청크는 같은 시간에 host·SSD 층에서 비동기 적재. 뒤 KV는 같은 프리픽스에서 나온 것이라 유효하고 앞 chunk의 attention은 뒤 블록을 보지 않으므로, 뒤는 decode 전까지만 도착하면 됨(Cake의 compute-from-front, load-from-back).

| 위치 | 변경 |
|---|---|
| vllm/v1/kv_offload/split_policy.py | 분할 제어기. VLLM_KV_SPLIT = off, fixed:<비율>, serial:<비율>(대조군: 앞 H−k를 보통의 프리픽스 적중으로 적재한 뒤 뒤 k 계산, 겹침 없음), model(max(k·토큰/R, host 바이트/B_host, SSD 바이트/B_ssd)를 최소로 하는 k. R, B는 VLLM_KV_SPLIT_RATE_TOKS, _BW_HOST_GBS, _BW_SSD_GBS) |
| kv_connector/v1/offloading/scheduler.py | lookup 뒤 분할 결정. 뒤 청크 수만 외부 적중으로 보고하고 요청에 head·tail·경계 기록. 적재 제출은 뒤 청크의 블록 id로만 |
| v1/core/sched/scheduler.py | 분할 요청은 WAITING_FOR_REMOTE_KVS에 서지 않고 바로 앞 계산 시작. chunk는 경계를 넘지 않게 상한, 경계에 닿으면 뒤 적재 완료까지 RUNNING 상태로 대기, 완료 시 num_computed_tokens를 경계+tail로 올림. 적재 중 분할 요청은 선점 대상에서 제외. 게이트는 우회하되 도착으로 집계 |
| csrc cufile_fs.cpp, cufile_fs/spec.py | 읽기 정지·재개(pause_reads)와 cufile_fs_load_window=host(가중치 SSD 읽기 중 KV SSD 읽기 정지) |
| base.py, cufile_fs, hybrid | key_tiers(키별 host/ssd) |
| run_obs.py | --kv-split, result.json kv_split(분할 요청 수, 재계산·적재 토큰, 뒤 대기 횟수·초) |

- 단일 full-attention KV 그룹에서만 켜짐(SWA·eagle·mamba는 경계가 한 위치가 아님).
- QA(Qwen2.5-3B, host 0.02, Bailian 24건 4k, KV 2요청): off, fixed:0.75, model, hybrid(host 1.8 GB)+fixed:0.5, off 반복 다섯 런의 출력 토큰열이 48건 전부 동일, backend 오류 0. fixed:0.75는 18요청 분할, 재계산 27,904·적재 9,600 토큰. model은 R 기본값 215 tok/s(72B)라 3B에서 k=0.
- 뒤 대기(tail_wait)는 앞 계산이 끝난 step 경계에서 완료를 확인하는 구조라 최대 한 step 늦게 반영됨(3B fixed:0.75에서 요청당 약 1.2 s).
- nsys: kv_split(요청, head, tail), kv_tail_ready 표시가 잡힘. 3B 4k에서 fixed:0.5의 reverse 단계는 off보다 느림(90 → 99 s): 이 조건은 SSD 적재가 재계산보다 빨라 k>0이 손해인 영역이며 예상과 일치. 격자(campaign_grid3b.sh)로 경계를 잼.

#### 가중치 티어 교차 배치와 prefetch 깊이 2 (72B)

포크 오프로더 VLLM_OFFLOAD_TIER_LAYOUT=interleave(host layer 수는 block 배치와 같게, 위치는 80 layer에 고르게)와 prefetch_step 2. 3B(host 17 / SSD 19 layer)에서 decode forward 1.59 → 1.15 s(−28%), 둘 중 하나만으로는 1.29(깊이 2만), 1.46(교차만).

72B RAM 0.5 재계산, Bailian 32건. 정적 버퍼가 한 세트(1.63 GiB) 늘어 GPU KV 자동 예산(21k 토큰)과 같이 넣으면 첫 prefill 또는 warm-up 샘플러가 OOM. 조건을 맞추려고 prefill 조각 2048(--max-num-batched-tokens)과 GPU KV 고정(--kv-batch 2.0 = 5.75 GiB, 18.8k 토큰)으로 실행.

| 조건 | decode forward | prefill forward 평균 | forward 수 | 두 단계 합계 |
|---|---|---|---|---|
| block 배치, 깊이 1, 조각 8192, KV 21.4k (기준) | 28.4 s | 62 s | 107 | 4,350 s |
| 교차 배치, 깊이 2, 조각 8192, KV 16.3k, gpu_util 0.75 | 26.4 s | 49 s | 138 | 4,712 s |
| 교차 배치, 깊이 2, 조각 2048, KV 18.8k, gpu_util 0.85 | 22.7 s | 29 s (조각당) | 164 | 4,369 s |

- forward 고정비는 28.4 → 22.7 s(−20%). nsys에서 본 SSD 읽기만의 시간 23.3 s와 같으며, host 복사 5.7 s가 SSD 읽기 아래로 다 숨은 값. 채널 측정의 host 저하(동시 실행 시 0.67)는 SSD 읽기 23 s 안에 host 62 GiB / 8 GB/s = 7.7 s가 들어가므로 forward 길이에는 안 나타남.
- 두 단계 합계가 기준과 같은 것은 조각 2048과 KV 18.8k로 forward 수가 107 → 164로 는 몫이 상쇄해서. 같은 조각·KV로 맞춘 조건이 없어 wall clock 이득은 아직 미확정. 조각 8192 + KV 고정 2.0 조건을 추가 예정.
- 0.75·조각 8192 런의 26.4 s가 22.7 s와 다른 원인은 미확립(nsys 없음).
- 출력 토큰열 64건 동일.

#### nsys 캡처 구간과 NVTX, 블록 I/O 추적

- 러너 --nsys-phase NAME --nsys-steps N: 그 phase 시작에 cudaProfilerStart, N개 forward(0.3 s 이상 step) 뒤 또는 phase 끝에 Stop. lib/obs/run_nsys.sh를 NSYS_CAPTURE=cudaProfilerApi로 감싸면 그 구간만 기록(capture-range-end=repeat라 여러 phase도 한 리포트). nsys 2025.3.1(~/nsight-systems-2025.3.1)의 gds trace(실험 기능)를 자동으로 켬. campaign_qwen72.sh는 NSYS=1이면 이 래퍼를 씀.
- NVTX. 러너 phase:NAME, step:NAME. 오프로더 prefetch:L{i}:{cpu|ssd}(layer마다), ssd_window:on/off(SSD 구간 전환). backend kv_store_file, kv_load_file(파일 하나), kv_writes_pause/resume. libnvToolsExt가 있을 때만 링크.
- Qwen2.5-3B, host 0.02, LongBench 8건, cold_fill 캡처로 확인. 리포트 24 MB, NVTX 34,017건. gds trace는 cuFileRead 1,368건(평균 48 ms), cuFileWrite 1,376건(32 ms), cuFileHandleNVFS 5,138건(37 ms, 합 192 s)으로 잡혀 KV 파일 store(kv_store_file 32.8 ms)와 cuFileWrite(32.3 ms)가 1:1로 맞음. cuFileHandleNVFS의 몫은 미분석.
- 블록 I/O 한 건 추적. bcc 0.12(bpfcc-tools)의 biosnoop은 커널 5.15에서 kprobe blk_account_io_completion이 없어 실패. 대신 bpftrace 0.9.4로 tracepoint block_rq_issue/complete 기반 lib/obs/blkio.bt(ns 시각, dev, rwbs, 섹터, 바이트, 지연 us)를 두고 hostmon.sh가 BLKIO=1이면 sudo로 실행. sudoers에 /usr/bin/bpftrace NOPASSWD가 필요하며 미설정.
- SSD 쓰기 캐시 회복 시간(dd 4.5 GiB, 36% 사용, 72B 다운로드와 겹침): 채우기 0.85, 바로 이어서 0.70, 10 s 쉬고 1.06, 바로 이어서 0.70, 20 s 쉬고 0.40 GB/s. 쉬는 시간과 회복이 단조가 아니어서 write-behind의 묶음 크기·휴지 시간을 이 값으로 정하지 않음. 72B 런의 blkio·diskstats로 다시 봄.

#### OPT-66B 종료와 모델 전환

66B에서 볼 것은 위에서 끝남. HF 캐시와 SSD 티어 파일을 삭제(재현은 results/native-66b의 result.json·steps.jsonl·events.jsonl로). 다음 모델은 Qwen2.5-72B-Instruct(GQA, 80 layer, KV 헤드 8, 토큰당 KV 0.33 MB). 가중치 145 GB로 host를 넘쳐 SSD 티어가 남는 조건은 유지하면서 KV만 7분의 1로 줄어 GPU에 요청 여럿이 공존하고 배치당 쓰기 양이 줄어듦. 입력은 LongBench-v2 32건과 Bailian 프로파일. Qwen에서는 토큰열 완전 일치를 정합성 검사로 못 쓰는 점(앞 절)을 그대로 적용.

#### Qwen 구조에서의 오프로더와 native KV 경로

Qwen2.5-3B-Instruct(GQA, gate·up·down FFN, RMSNorm)로 오프로더와 CuFileFsSpec 스모크. 오프로더는 decoder layer 모듈 전체를 파라미터 이름 화이트리스트 없이 감싸므로 구조 의존이 없음. 36 layer, 정적 버퍼 풀 154 MB(layer 하나분)라 BAR1 등록이 4개 모두 성공(OPT-66B는 1.9 GiB라 실패). host 비율 0.02(2.51 GiB)에서 CPU 17 / SSD 19 layer, forward 1.5 s. 32문서·4k 토큰에서 reverse_retrieve wall clock 45.0 s(재계산)에서 10.2 s(SSD 적중, 읽기 833건 1.83 GiB, 오류 0). 4문서는 GPU KV 안에 다 남아 SSD 적중이 없음(축출이 있어야 SSD를 읽음). SSD 티어 런은 layer 0.144 GiB라 forward가 모형보다 26~32% 느려 소형 layer의 cuFile 유효 대역폭이 더 낮은 것으로 보이며 미확립.

출력 토큰열 검사의 한계. 32문서에서 재계산 대 적중이 64건 중 1건 불일치였는데, 재계산끼리 GPU KV 예산만 바꾼 두 런도 2건이 달랐음. Qwen + TRITON_ATTN + chunked prefill에서 greedy 출력이 프리픽스 적중 길이에 따른 prefill 조각 경계에 좌우되는 부동소수점 축약 순서 문제. OPT에서는 없던 현상. 이 조건에서는 토큰열 완전 일치를 정합성 검사로 쓸 수 없고, 재계산 런 사이의 불일치 수를 기준선으로 둔다.

러너 입력. LongBench-v2 32건(양태규 패키지 데이터)을 모델 토크나이저로 토큰화해 프리픽스로 쓰고 단계별로 다른 질문 꼬리를 붙이는 longbench 소스, 02의 Bailian 트레이스 hash_id 열을 결정적 16토큰 블록으로 바꿔 프리픽스 공유 구조를 보존하는 bailian 소스(실제 텍스트·시간 간격·멀티턴 거리는 미재현), 요청별 적중 토큰과 doc별 재사용 횟수를 남기는 --profile-out.

#### BAR1 창과 정적 버퍼 등록

13b는 host 0.7부터 가중치 SSD 읽기가 cuFileRead −1(cuFile 로그 −5011)로 실패. qkv 150 MiB와 out_proj 50 MiB가 등록되어 BAR1 256 MiB 중 227 MiB를 차지했고, 등록 실패한 fc1과 fc2가 쓰는 cuFile bounce 캐시를 매핑할 자리가 29 MiB뿐. 66B는 out_proj 162 MiB 하나만 등록돼 여유가 있었고 6.7b는 128 MiB로 턱걸이. 포크 ssd_tier에 등록 총량 상한 VLLM_OFFLOAD_SSD_REGISTER_MAX_MB(기본 상한 없음, 포크 929df037b9)를 넣어 13b와 30b는 100 MiB로 실행. 상한 적용 후 BAR1 사용 85 MiB. BAR1 256 MiB는 카드 자체의 최대. PCI Resizable BAR capability(0xbb0)를 setpci로 읽으면 BAR1 항목의 지원 크기가 64, 128, 256 MB(capability 0x1c00, control 0x801)뿐이라 BIOS나 커널로 키울 수 없음. 데이터센터 GPU는 VRAM 전체를 BAR1로 광고.
