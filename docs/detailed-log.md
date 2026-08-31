### 상세 기록

README의 요약을 뒷받침하는 설계 근거, 전체 측정표, 발견과 정정의 기록이다. 진행 순서대로 쓴다.

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

causal-closure로 V2에서 C(block 64), E(block 64), D(block 64 재측정)를 추가로 잰다. C의 b16 대 b64가 bucketing 효과를, C 대 E가 control plane 차이를, E 대 D가 cuFile 단독 효과를 분리한다. 결과는 w1/w1_causal.csv에 쌓인다.

### /dev/shm mmap 누출의 근본 수정

vLLM의 SharedOffloadRegion은 정상 종료의 cleanup에서만 파일을 지우므로 SIGKILL, OOM, init 중 예외, 일부 인터프리터 종료 경로에서 /dev/shm 파일이 누적되고, tmpfs가 차면 다음 엔진이 EFAULT로 죽는 2차 장애가 난다.

수정은 flock 기반이다. 모든 참여자가 파일에 shared lock을 프로세스 수명 동안 쥔다. 커널이 어떤 죽음에도 락을 풀므로, 시작하는 엔진이 exclusive lock을 잡을 수 있는 파일은 고아임이 증명되어 안전하게 회수한다. 살아 있는 다른 엔진의 파일은 락이 잡혀 있어 건드리지 않는다. 같은 engine_id 재시작은 stale을 회수하고 새 creator가 되며, init 도중 실패도 파일을 남기지 않는다.

독립 diff 리뷰가 10건을 지적했고 8건을 반영했다. 모든 unlink에 inode 재검증(경로 대 inode TOCTOU), blocking shared lock(락 실패 삼킴 제거), 합류 후 inode 재확인(재시작 split-brain 차단), 자기 경로의 0바이트 고아는 짧은 유예 후 회수(engine_id 영구 고착 방지). 잔여 위험: 미패치 vLLM 인스턴스가 같은 호스트에 공존하면 그 파일에는 락이 없어 회수될 수 있다(flock 계열 방식의 태생적 한계). EC connector 쪽 동일 계열 결함은 수정 범위 밖으로 기록만 했다.

검증: 회귀 포함 34개 테스트가 4연속 통과, 실제 엔진을 SIGKILL한 뒤 8GiB stale이 다음 시작에서 자동 회수되는 E2E 통과. harness는 삭제 대신 관찰만 한다. 런 전 목록과 용량을 기록하고, 회수는 엔진의 Reclaimed 로그로, 정상 종료는 잔재 0으로 확인한다.

### 교훈

복사를 없앤다고 빨라지지 않는다는 오래된 교훈이 세 번 확인됐다. 소조각 IO에서, control plane 분해(E 대조군)에서, production trace의 store 동시성에서. 표준 프로파일러가 관측 대상을 바꿔 버리는 경우(GIL 콘보이)에는 프로파일러의 개입 자체가 진단 단서가 된다. 그리고 synthetic 벤치의 승자는 production trace 앞에서 겸손해야 한다.
