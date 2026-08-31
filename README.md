# vLLM + GDS KV 오프로드 실험

> **한 문장 정의**: GPU·CPU 메모리보다 큰 재사용 prefix working set을 가진 agent/RAG/장기
> session workload에서 SSD KV load가 prefix prefill 재계산보다 유리한 구간을 먼저 입증하고
> (RQ1), 그 구간에서 기존 SSD→CPU→GPU 경로를 SSD→GPU GDS 직행으로 바꿀 때의 성능·메모리·
> 복잡성 손익을 검증한다(RQ2).
> 작업 소스: `~/vllm` (v0.26.1.dev = tag v0.26.0, commit 568afb3a13, 2026-07-26), conda env `gdsllm` editable 설치.
> 실행 환경: `source env.sh` (conda libstdc++ LD_PRELOAD + PYTHONHASHSEED=0 필수).

## §0. 연구 질문과 당위성 (v3, 2026-08-31)

- **RQ1**: SSD에 저장된 KV를 읽는 것이 동일 prefix를 prefill로 재계산하는 것보다 유리한
  workload가 실제로 존재하는가? — **RQ1을 먼저 검증해야 한다.** RQ1 없이 GDS vs POSIX만
  비교하면 구현 성능은 비교되지만 SSD KV 자체의 정당성이 없다.
- **RQ2**: 그 workload에서 GPU→CPU→POSIX 경로보다 filesystem↔GPU cuFile 경로가 유리한가?

핵심 조건: `T_SSD_load(prefix) < T_prefill_recompute(prefix)`
store 포함 break-even: `T_store + H×T_load ≤ H×T_recompute` → `H_min = ceil(T_store / (T_recompute − T_load))`

SSD-backed KV가 의미를 갖는 조건(요약): 동일한 긴 prefix의 반복(agent system
prompt/tool schema, 반복 문서/repo context, 장기 session 재개, 프로세스 재시작·다중
인스턴스 공유) + working set이 GPU·CPU 용량 초과 + 재사용률 충분. 선행 근거: vLLM
multi-tier(FS 티어), NVIDIA Dynamo KVBM G3, LMCache local storage, Tutti, RFC #48504.
단 "존재"가 "유리"를 의미하지 않으므로 위 부등식을 실측한다(→Phase 0.5).

결과 3분기 모두 유의미: ① SSD load > recompute → 이 환경에선 SSD KV 부적합
② SSD 유효 + GDS ≤ 기존 → CPU-staged 설계가 합리적 ③ SSD 유효 + GDS 우세 → 도입 근거.

### 0.1 세 질문의 분리 (v4, 2026-08-31 — workload 조사 반영)

| 질문 | 상태 |
|---|---|
| **Transport feasibility** — SSD hit가 prefill 재계산보다 빠른가? | **확인 완료** — opt-2.7b에서 P≥1024 성립 구간 존재 (§3) |
| **Workload legitimacy** — 실제 workload에서 filesystem hit가 자연 발생하는가? | **외부 근거 확보** (Bailian/Mooncake/Tutti/vLLM docs/RFC #48504, §0.5) — 이후 W1/W2 replay로 rain에서 재현 |
| **GDS benefit** — SSD→CPU→GPU보다 SSD→GPU가 빠르거나 자원을 아끼는가? | **미확인 — C와 D의 직접 비교로만 답할 수 있음** (Phase 3) |

**핵심 결정**: workload 조사는 GDS 구현의 허가 조건이 아니다. 외부 정당성(production
trace의 장기 재사용 KV 존재, Mooncake의 실제 SSD 티어 구현, Tutti의 vLLM+SSD+GDS 계열
직접 평가, vLLM 공식 fs 티어·persistence 지원, RFC #48504의 worker-side backend 제안)은
이미 충분하다. **W1/W2는 구현 허가 gate가 아니라, 구현된 GDS 경로를 end-to-end로 평가할
대표 workload다.**

**Mooncake의 위치**: Mooncake는 production-derived workload에서 SSD-backed KV tier의
필요성을 제시하고 실제 SSD offload를 구현했지만, SSD→GPU load는 아직 DRAM을 경유한다.
Mooncake가 후속 과제로 제시한 GDS direct path를 이번 실험에서는 vLLM native
OffloadingConnector 위에 구현해 성능과 자원비용의 손익을 검증한다. **GDS 미구현은 실험
결격 사유가 아니라 우리가 측정할 미구현 구간이다.**

역할 분담: Mooncake×Bailian = workload 정당성 + SSD 티어 실증 / Tutti·HiFC = SSD↔GPU
direct path의 기술 선행(단 HiFC는 active-KV swapping이므로 workload 근거로는 사용 금지,
cuFile 구현 방식·마이크로벤치 참고자료로만) / 우리 = vLLM native OffloadingConnector 위
worker-side GDS filesystem backend를 직접 구현해 C(SSD→CPU→GPU) vs D(SSD→GPU)를 실제
vLLM에서 비교.

## §0.5 워크로드 정당성 조사 (2026-08-31, 조사 전용 — 코드/실험 없음)

목적: 새 workload를 만들어 SSD 필요성을 "입증"하는 게 아니라, **이미 SSD-backed KV를 쓴
실제 시스템·trace·선행 연구**를 분류하고 현재 vLLM fs 티어와 의미론이 맞는 대표 workload를
고르는 것. CPU 티어를 인위로 줄이는 방식은 기능 검증용으로만 허용, 본 정당성으로 금지.

### 0.5.1 소스 검증: v0.26 OffloadingConnector는 정확히 A 계열이다

- 공식 문서 자기정의: "**extends the prefix cache** by offloading completed KV blocks...
  Hits ... are promoted back to GPU on demand" (`docs/features/kv_offloading_usage.md`).
- `offload_prompt_only` 기본 **true** (`v1/kv_offload/base.py:510`) — 기본은 prefill 블록만.
- store = 매 스케줄 스텝에서 계산 완료된 full chunk 증분 저장(`_build_store_jobs`,
  prompt 토큰 상한 `offloading/scheduler.py:454`); load = 요청 admission 시
  `get_num_new_matched_tokens` prefix lookup 뿐. **디코드 중 활성 KV 스왑 경로 없음** →
  B 계열(FlexGen류) 논문은 이 서브시스템의 근거가 될 수 없다.
- persistence/multi-instance는 공식 intended use: 같은 root_dir 공유(PVC 예시),
  동일 config → 동일 digest 디렉터리 재사용, 전 인스턴스 `PYTHONHASHSEED` 고정.
  `max_offload_tokens` 설명에 "known prefix (e.g., a **system prompt or shared context**)"
  — 공식 문서가 상정하는 workload 자체가 A 계열.

### 0.5.2 후보 비교표

근거 수준: 1 production trace+실SSD / 2 production trace·다른 스토리지 / 3 공개 실데이터 시스템 평가 / 4 공식 합성 벤치 / 5 마이크로벤치·인위적 압박

| 후보 | 실제 workload | 근거 | SSD 이유 | KV | 로컬NVMe | vLLM FS 적합성 | rain 재현성 | 주요 한계 |
|---|---|---|---|---|---|---|---|---|
| 1 Qwen-Bailian trace | Aliyun Bailian 2h production(coder/thinking/to-C/to-B), 16-tok SipHash 블록해시 | 2 | 원논문은 DRAM으로 충분 주장(to-B "small on-GPU cache... eliminating CPU-RDMA-SSD") — SSD 근거는 coder/thinking cold tail만 | A | (무관) | 블록해시 16tok=vLLM granularity 1:1, **공식 replayer가 OpenAI API로 vLLM 리플레이** | 상: `--scale-factor`+session 샘플링 | 2h 샘플, 합성 토큰, 원논문의 SSD 회의론 |
| 2 Mooncake SSD | 같은 Bailian trace 분석+실SSD 구현. coding 재사용 69.2% 10분내·**10.3% 30분후 cold**, reasoning 83.0%/5.0% | **1**(trace 분석)/4(E2E 수치) | cold tail은 DRAM에 비경제적, tail hit의 prefill 이득 비선형 | A | 혼합(노드 로컬 NVMe→분산 풀) | FilePerKeyBackend ≈ vLLM fs 구조 동일(단 "성능용 아님" 명시). 현 경로 SSD→DRAM 스테이징, **"GDS로 DRAM hop 제거"가 명시된 ongoing work** | 상: 라운드-7 cliff(hit 83→36%, TTFT 6→16s) 단일노드 재현 가능 | GDS 미구현(aspirational), E2E는 벤치 기반 |
| 3 LMCache bench | long-doc-qa/multi-round-chat 등 합성 생성기 | 4(psf-thrash는 5) | 문서 자체에 디스크 결과 없음; psf-thrash는 명시적 오버플로 노브 | A(permutator 제외) | n/a(클라이언트) | **OpenAI 호환이라 LMCache 없이 vLLM native fs에 리플레이 가능**; `--kv-cache-volume`>물리DRAM로 자연 구성 가능 | 최상(pip 클라이언트) | 대표성 주장 없음, permutator는 non-prefix라 부적합 |
| 4 Tutti | LEval(3K–200K)/LooGLE(100K+) 실데이터, vLLM KVConnector 통합, GPU당 로컬 NVMe 14TB | 3 | **256GB DRAM에서도** hit: DRAM 53→SSD 84%(LEval), 24→86%(LooGLE) = 자연 발생 | A | **예** | 자기 설계는 GPU-doorbell(GDS 비판), **LMCache-GDS(cuFile 로컬NVMe→GPU)가 이 논문의 baseline = 우리 설계의 위치** | 중: 데이터셋 공개·축소 가능, 절대수치·100K 컨텍스트는 불가 | Table 1 방법론 미비, Poisson 합성 도착 |
| 5 DualPath | production 에이전트 RL trace(500 traj, 평균 60~157턴, **턴간 재사용 >95%**) | 2 | 용량+교차엔진 공유; working set=λT̄×len/2, 69→681GB | A | 아니오(분산 3FS, 400Gbps NIC) | 의미론 일치, 아키텍처 무관 | trace 미공개 — Table 2 통계로 합성 드라이버 파라미터화만 | 수치가 3FS/MLA/660B 특화 |
| 6 HyMCache | LMSYS-Chat-1M 유래+Mooncake trace 합성 | 3 | 명시적 **비용/용량**("TB-scale... difficult to support economically") | A | 아니오(CXL.mem) | 접근패턴(read-dominant·predictable·append-only) 논제는 정확히 fs 티어 전제 | 시스템 재현 불가, workload 논제만 이식 | CXL 프로토타입, 파일시스템 아님 |
| 7 HiFC | vLLM 통합, GDS로 GPU↔로컬NVMe(pSLC), Qasper/GovReport/NarrativeQA | 3–4 | TCO(pSLC 1TiB $136 vs DRAM 128GiB $614); 압박은 실험적 유도 | **B**(victim sequence swap-out/in, 요청과 함께 소멸) | **예** | **아니오** — vLLM이어도 preemption swap 서브시스템(함정!) | 데이터 경로만 참조 | production trace 없음 |
| 8 DUAL-BLADE | FlexLLMGen, OPT-6.7B, cgroup으로 host 2–11GB 클램프 | 5 | 인위적 host 제한 | **B**(매 디코드 스텝 active KV 스트리밍) | 예(LBA-direct, GDS 아님) | 아니오 | 재현 쉬우나 무가치(우리 질문에) | 정당성으로 사용 금지 |
| 9 vLLM persistence | 재시작 warm-start/다중 인스턴스 공유 — 공식 문서 명시 intended use | 공식 docs | 프로세스 소멸 후 유일한 생존 티어 | A | 예 | 정의상 완벽(그 자체가 vLLM fs) | **최상**(fs_hit.py가 이미 이 형태) | inference workload라기보다 operational |

### 0.5.3 판정

- **추천 워크로드 (2.5개)**:
  1. **Mooncake×Bailian coder/thinking** (사실상 한 몸): production validity 최고(근거 1).
     trace-replayer→vLLM 서버 리플레이, 라운드-cliff 재현. **우리 GDS 실험은 Mooncake이
     명시한 ongoing work(DRAM hop 제거)의 선행 검증**이라는 위치까지 확보.
  2. **Tutti형 long-doc QA**: 로컬 NVMe에서 "DRAM 안 줄여도 SSD가 hit를 올린다"의 실데이터
     증거 + **LMCache-GDS baseline이 우리 설계와 동일 슬롯**(데이터 경로 최근접 선행).
     실행 수단은 lmcache bench long-doc-qa/multi-round-chat(vLLM native로 리플레이,
     kv-cache-volume>물리DRAM 자연 구성, **psf-thrash 사용 금지**).
  2.5. **후보 9 재시작 warm-start**: 공식 intended use·최고 재현성. GDS read-only load
     (기동 직후 대량 프로모션 버스트) 시나리오로 Phase 3 보조군.
- **배경 사례 전용**: DualPath(>95% 재사용·working set 성장 법칙 인용), HyMCache(경제성·
  접근패턴 논제 인용), ATC'25 원논문(정직한 반대 근거로 명시 인용 — to-B에선 SSD 불필요).
- **다른 계열로 분리(정당성 사용 금지)**: HiFC·DUAL-BLADE = B(active-KV swapping).
  단 데이터 경로 증거는 차용: HiFC — KV 텐서 자체를 4KB 정렬 GDS 버퍼로 등록, 16스레드 I/O,
  **64-token block sweet spot**, 순차 append 레이아웃 WAF 1.02, gdsio seq 4.7–5.0GiB/s vs
  rnd write 1.6GiB/s, **"동시성으로 지연 은폐 시 SSD 티어 ≈ DRAM 티어 e2e 동률(1–2%)"**;
  DUAL-BLADE — 페이지캐시 대비 direct I/O 43–120% 처리량 우위.
- **세 기준의 답이 서로 다름 (합치지 않음)**: production validity 최고 = Mooncake×Bailian /
  rain 재현 최용이 = 후보 9(다음 LMCache bench) / GDS 데이터 경로 최근접 = Tutti의
  LMCache-GDS baseline(A계열 내), HiFC(B계열, 기술 최근접).
- **프레임 문장**: "DualPath의 workload 통계와 HyMCache의 접근패턴 논제, Mooncake이 예고만
  한 GDS 경로를 — 어떤 선행연구도 평가하지 않은 상용 스택(vLLM native fs + 로컬 ext4 +
  cuFile)에서 실증한다."

### 0.5.4 대표 workload 확정 (v4 — 구현 후 end-to-end 평가용; 허가 gate 아님)

- **W1 — production-derived primary**: Qwen-Bailian coder/thinking trace, 공식
  trace-replayer 사용. request order·reuse distance·session 관계 보존.
  **CPU 티어를 임의 2~4×로 잡지 않음** — rain에서 안정적으로 쓸 수 있는 실제
  operational CPU budget 사용, natural filesystem hit 발생 여부를 먼저 확인.
- **W2 — long-document secondary**: LEval/LooGLE 실제 document dataset 우선.
  LMCache long-doc-qa는 실행 harness로만 참고. `psf-thrash`류 인위 오버플로는
  workload 정당성으로 사용 금지.
- **W3 — operational**: engine restart 후 persistent filesystem warm-start,
  read-only GDS 시나리오. `phase0/fs_hit.py` 활용 가능.
- 관측: 오프라인은 몽키패치 계수, 서빙은 `vllm:kv_offload_*` Prometheus 메트릭.
- HiFC의 "동시성 은폐 → e2e 동률" 결과에 따라 **단독 요청 TTFT와 동시성 처리량 분리
  보고**(D vs C 차이가 파이프라인 slack에 은폐될 수 있음 — 교훈 3의 재림 후보).

## §1. 배경: vLLM 구조와 GDS 부재의 이유

### 1.1 vLLM이 뭔가

LLM **추론 서빙 엔진**. 핵심 발명은 PagedAttention(SOSP 2023):
KV 캐시를 OS 페이징처럼 고정 크기 블록(기본 16토큰) 단위로 관리해 메모리 단편화를 없애고,
그만큼 동시 배치를 키워 처리량을 올린다. 여기에 continuous batching(요청이 토큰 단위로
배치에 join/leave)이 결합된 구조.

**설계 전제: 가중치는 VRAM에 전부 상주한다.** 정상상태 서빙 루프(prefill/decode)에
스토리지 I/O가 존재하지 않는다. 이 전제가 GDS 부재의 근원.

### 1.2 구조 (v1 엔진, 설치본 기준)

```
AsyncLLM / LLM (API 계층)
  └─ EngineCore ─ Scheduler        # 매 스텝: 어떤 요청의 어떤 토큰을 실행할지 + KV 블록 할당
       └─ Worker (GPU당 1개) ─ ModelRunner  # forward 실행, KV 캐시 물리 소유
```

**이 빌드에는 model runner가 두 벌 공존** (`VLLM_USE_V2_MODEL_RUNNER`로 선택,
`gpu_worker.py:402`): 구형 `gpu_model_runner.py`(V1)와 신형 `gpu/model_runner.py`(V2, 기본).
KV 배치가 서로 다르다 — §2.1' 참조.

블록 크기 감각 (block_size=16토큰, fp16 기준 레이어당 페이지):
- MHA (opt-6.7b: 32헤드×128): 2(K,V)×16×32×128×2B = **256KiB**
- GQA (Llama-3-8B: KV 8헤드×128): 2×16×8×128×2B = **64KiB**
- opt-125m (12헤드×64): **48KiB**/레이어, 전 레이어 chunk = 576KiB (실측 일치)

스토리지/메모리 I/O가 등장하는 서브시스템은 3개뿐:

1. **가중치 로드 (1회성)** — safetensors mmap→H2D가 기본. `--load-format fastsafetensors`를
   주면 cuFile 직행 경로가 이미 존재(IBM 기여, 옵션일 뿐 코어 아님).
2. **KV connector** (`vllm/distributed/kv_transfer/`) — prefill/decode 분리 등 인스턴스 간 KV 전송.
3. **KV 오프로드 티어링** (`vllm/v1/kv_offload/`) — 이번 실험 대상. 아래 상세.

### 1.3 KV 오프로드 데이터 평면 (코드로 확인한 사실)

```
GPU KV 텐서 (num_gpu_blocks, gpu_page_size)
   │  ① copy engine 배치 swap (ops.swap_blocks_batch, 전용 CUDA 스트림+이벤트)
   ▼      cpu/gpu_worker.py:394 — GPU→CPU는 "대역폭 바운드라 copy engine이 Triton보다 낫다"고 주석 명시
CPU 1차 티어: pinned 텐서 (num_cpu_blocks, cpu_page_size = gpu_page × blocks_per_chunk)
   │  ② POSIX 스레드풀 (읽기16+쓰기16), O_DIRECT os.write/readv
   ▼      tiering/fs/io.py:53-66 — CPU memoryview 슬라이스를 그대로 파일에 씀
2차 티어: fs (파일) / obj (오브젝트 스토어) / p2p (NIXL)
```

핵심 관찰: **2차 티어 인터페이스(`SecondaryTierManager`)는 CPU 뷰(`primary_kv_view:
memoryview`)만 받는다** (`tiering/fs/manager.py:108`). GPU 텐서는 시야에 없음 —
GPU→스토리지 직행은 인터페이스 수준에서 배제된 설계다. tiering RFC #38260도
"GPU-storage 직접 접근 비지원"을 명시.

### 1.4 DeepSpeed ZeRO-Inference와 왜 갈리나

| | DeepSpeed ZeRO-Inference | vLLM |
|---|---|---|
| 목적 함수 | VRAM보다 큰 모델을 **돌게** 만들기 | VRAM에 들어가는 모델을 **빠르게** 서빙 |
| 가중치 위치 | NVMe 상주, 매 forward마다 스트리밍 | VRAM 상주 (로드 1회) |
| 정상상태 I/O | 토큰당 모델 전체 바이트 (opt-6.7b: ~13GB/token) | **없음** (KV 오프로드는 선택 기능) |
| I/O 비중 (rain 실측) | 시간의 98.6% | ~0% |
| I/O 단위 | 레이어 가중치 덩어리 (수십MB~) | KV chunk (전 레이어 합산, 수백KiB~수MB) |
| GDS 효과 | decode +40% 실측 (gds-llm-demo) | ? ← 이번 실험 |

GDS 부재의 이유, 3층위:

1. **워크로드**: I/O가 크리티컬 패스에 없다. GDS가 최적화할 대상 자체가 정상상태에 없음.
2. **있는 I/O의 성격**: KV 오프로드의 목적은 prefix cache 확장(TTFT 절감·재계산 회피).
   재사용 히트 시 로드 지연이 중요 → 1차 티어는 DRAM(pinned, copy engine ~12GB/s)이
   NVMe(~3.5GB/s)보다 합리적. 디스크는 용량 티어(2차)로만.
3. **인터페이스 경계 + 운영 복잡성**: 2차 티어가 스케줄러 프로세스에 있어 GPU 접근 불가
   (§2.1)가 구조적 원인 중 하나이며(RFC #48504가 이를 공식 확인), GDS native의 스택 요구
   (nvidia-fs+MOFED+open driver)는 범용 프레임워크 기본 경로로는 이식성이 나쁘다.

## §2. 실험 설계 (v2, 2026-08-31 — 사용자 검토 반영)

### 2.1 제어 흐름 조사 핵심 (소스 검증 완료)

**활성화 경로는 두 가지다:**
1. `--kv-offloading-size <GiB>` — native CPU offloading을 자동 구성하는 **편의 경로**
   (`VllmConfig._post_init_kv_transfer_config`가 OffloadingConnector+`cpu_bytes_to_use` 주입).
2. tiering·out-of-tree spec은 `--kv-transfer-config`에서 `OffloadingConnector`와
   `spec_name`을 직접 지정해 활성화. **이 경우 `--kv-offloading-size`는 필수가 아님**
   (Phase 0가 이 플래그 없이 동작함으로써 실증).

```bash
PYTHONHASHSEED=0 vllm serve <model> --kv-transfer-config '{
  "kv_connector": "OffloadingConnector", "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 10737418240, "block_size": 16,
    "secondary_tiers": [{"type": "fs", "root_dir": "/path/kv_cache"}]}}'
```

스펙 선택: `OffloadingSpecFactory`(`v1/kv_offload/factory.py:29`)가 `spec_name`으로 결정,
기본 `CPUOffloadingSpec`. **out-of-tree 스펙은 `spec_module_path`로 지원됨** — in-tree 수정
없이 GDS 스펙을 얹을 수 있는 공식 이음새.

**구조적 제약 3개 (GDS "2차 티어"가 불가능한 이유):**
1. 2차 티어는 **스케줄러 프로세스**에서 실행 (`tiering/base.py:106`) — CUDA 컨텍스트/GPU 텐서 없음.
2. 티어가 받는 유일한 데이터 핸들 = `/dev/shm` mmap 위 host memoryview (`shared_offload_region.py:56`).
3. `JobMetadata.block_ids` = CPU 슬롯 인덱스 — GPU 블록을 지칭할 방법 자체가 없음.

**오프로드 단위**: GPU 블록이 아니라 **chunk** = `blocks_per_chunk`×GPU블록,
전 rank·전 레이어 텐서를 연결한 **플랫 버퍼 1개 = 파일 1개**, `mmap.PAGESIZE` 정렬(→O_DIRECT 합법).

### 2.1' GPU span 기하 — 러너에 따라 갈린다 (실측 확정, 2026-08-31)

GPU 쪽 연속성은 모델·attention backend·**model runner**에 따라 달라진다.
`OffloadingConnector.prefer_cross_layer_blocks=True`이고 uniform KV 모델 + block-stride
indexing 지원 backend면 **cross-layer layout**(한 블록의 전 레이어 KV가 GPU에서 연속)이
설계상 선택된다. rain에서 opt-125m + TRITON_ATTN으로 실측(`phase0/check_layout.py`):

| | V2 러너 (기본) | V1 러너 (`VLLM_USE_V2_MODEL_RUNNER=0`) |
|---|---|---|
| 게이트 4조건 | 전부 True | 전부 True |
| cross-layer 할당 | **안 됨** — `gpu/kv_connector.py:54`에 `TODO: support cross_layers_kv_cache` | **됨** — `(13268, 12L, 12H, 16, 128)` 단일 7.8GB 스토리지 |
| chunk의 GPU 기하 | 레이어별 12개 분산 텐서 = **12 span × 48KiB** | **1 연속 span × 576KiB** (레이어 오프셋 4096B 실측) |

즉 같은 빌드에서 러너 선택만으로 두 기하를 모두 실현 가능 — Phase 1의 Case A/B가 둘 다
실존 경로다. HMA·MLA·미지원 backend의 일반 경로에서도 다중 span이 생기므로 Case B는
fallback 모사가 아니라 현행 기본값이다.

**핵심 가설 (수정판)**: ~~"코얼레싱 2홉 vs 산발 1홉"~~ →
**"현재 runtime의 실제 GPU span 기하에서, pinned CPU staging이 제공하는 비동기성·수명
분리와 GDS가 제공하는 CPU 우회의 손익 비교"**. (CPU 홉의 역할은 코얼레싱만이 아니라
GPU 블록을 빨리 놓아주는 수명 분리이기도 하다 — store 동안 GPU 블록 점유 시간이 측정 항목.)

### 2.2 삽입 지점: standalone `OffloadingSpec` (RFC #48504와 동일 결론)

티어링 계층 우회, `OffloadingSpec` 레벨에서 신규 구현. RFC 구조에 가깝게 명명:

```
ExperimentalFilesystemSpec (out-of-tree, spec_module_path 로드)
├─ FilesystemManager       # hash, lookup, 파일 소유권, completion 의미 (scheduler-side)
│    └─ FileMapper·AsyncLookupManager 재사용
├─ FilesystemWorker        # GPU pointer/span 해석, 동기화, job 수명 (worker-side)
│    └─ CanonicalKVCaches = (num_blocks, page_size_bytes) int8 GPU 텐서
└─ Transport (교체 가능)
   ├─ CuFileTransport      # GDS 직행 (nvidia-cufile 바인딩)
   └─ PosixBounceTransport # 같은 제어 흐름 + CPU bounce — Phase 3의 C군 대조
```

**범위 명시**: 본 구현은 v0.26.1.dev·TP=1·uniform full-attention 모델의 성능 검증용
out-of-tree prototype이다. GPU pointer/span 해석을 backend가 직접 수행하므로 upstream
RFC 구조(공통 worker로 pointer 변환 이동)와 달리 일반 모델 호환성·API 안정성을 목표로
하지 않는다. Phase 1에서 이득 확인 후에만 공통 refactor를 따라간다.

**정확성 필수 항목** (Phase 2 체크리스트):
- store 전 compute stream 완료 fence / load 완료 전 해당 GPU 블록 compute 접근 차단 fence
- store 중 GPU 블록 재사용 시 `wait(job_id)` 보장
- 임시 파일→rename 원자적 commit, short read/write·실패 파일 정리
- 4KiB file offset·size·GPU pointer alignment 확인
- 저장→로드 KV byte checksum 또는 출력 토큰 일치 검증 (fs_hit.py 방식 재사용)
- 동일 namespace에 writer 하나만 허용

### 2.3 RFC #48504와의 관계 (전부 GitHub에서 검증, 2026-08-31)

- **#48504** "[RFC]: GDS support for filesystem KV-cache offloading" — 2026-07-13 open, 현재도 open, 구현 PR 미확인.
  제안: ① GPU block→pointer 변환을 공통 connector worker로 이동 ② 독립 FilesystemSpec
  ③ 파일 의미/전송 분리(CPU bounce·direct GDS·NIXL GDS transport 교체) ④ MultiConnector로
  CPU 우선 + GDS 2차 read 경로. 유지보수자(Ofer Rozen)도 같은 순서 권고.
- 선행: PR #40020(multi-tier framework, merged), #45053(OffloadingWorker 인터페이스, merged),
  #44865(transfer data model 재구성, open).
- #38260(tiering, GPU 직접 접근 금지)과 모순 아님 — #48504는 tiering의 2차 티어가 아니라
  **별도 OffloadingSpec** 경로.
- 함의: 이 RFC는 "GDS가 빠르다"의 증거가 아니다. ① GDS 부재 원인 중 하나가 인터페이스
  경계임을 확인 ② upstream도 GPU-direct filesystem을 유효 후보로 봄 ③ 성능 이득·BAR1
  영향·최적 span 기하는 미검증 → **rain 실험은 RFC가 남긴 성능 질문을 실 하드웨어에서
  검증하는 역할.** 본 실험 설계는 RFC와 독립적으로 같은 삽입 지점에 도달했다.

### 2.4 Phase 구조

- **Phase 0 — 개통 + 베이스라인** (완료, §3): fs 티어 구동, FS-hit 분리 증명, GPU 기하 확정.
  남은 것: nvidia-fs 카운터 +0 확인, nsys.
- **Phase 0.5 — SSD KV 정당성 게이트 (신설, RQ1)**: 4경로 비교 —
  A. KV miss→prefill 재계산 / B. CPU hit→CPU→GPU / C. SSD hit→SSD→CPU→GPU / (D. GDS는 Phase 3에서)
  - 프로토콜(`phase05/bench.py`, 메커니즘 검증 완료 — §3): 한 프로세스에서 prefix별
    A(최초 요청) → `reset_prefix_cache()`(GPU만 리셋) → B → `reset_prefix_cache(reset_connector=True)`
    (CPU primary 리셋, **FS는 의도적 보존** — `tiering/manager.py:784` 문서화됨) → C.
    fs load는 O_DIRECT라 같은 프로세스여도 페이지캐시 우회.
  - 경로 검증: 몽키패치 계수 — A: fs_load=0·cpu_load=0 / B: fs_load=0·cpu_load>0 / C: fs_load>0.
  - synthetic: prefix 128~2032tok(모델 상한 내)×독립 prefix 반복×짧은 고유 suffix, max_tokens=1(TTFT 프록시).
    realistic(후속): 긴 system prompt/tool schema/반복 문서 + 상이한 질문, CPU 티어 초과 working set으로 eviction 유도.
  - 측정표(results.csv): model, prefix_tokens, rep, arm, ttft_s, fs_load_calls, cpu_load_jobs,
    io_s(스레드 IO시간 합), io_wall_s, io_bytes, chunk_files, store_drained, path_check.
    파생: 평균 TTFT A/B/C, C<A 여부, `H_min = ceil(store_wall / (TTFT_A − TTFT_C))`.
  - **게이트**: C<A 구간이 없으면 이 장비·모델에서 SSD KV 당위성 불성립 → 연구 결론은
    "전송 최적화 이전에 storage tier 자체가 손해"로 확정하고 GDS 통합은 참고 실험으로 격하.
- **Phase 1 — cuFile 마이크로벤치 (vLLM 밖), Phase 2 투자 판정 게이트.** 측정 행렬:
  - geometry: chunk 실물 기준 opt-125m 576KiB(L=12) / Llama-3-8B 2MiB(L=32) / opt-6.7b
    8MiB(L=32), × offload chunk 16/64/256 tokens(×1/×4/×16), × span 구조 1 / L / 2L
    (V1 러너 cross-layer = 1 span, V2 러너 = L spans, K/V 분리 = 2L — §2.1')
  - transport 5종: ① explicit cuFileBufRegister ② **미등록 cuFile I/O**(BAR1 256MB에서
    전체 등록은 전제 금지 — **등록 실패는 정상적인 실험 분기**; 미등록은 내부 캐시+추가
    복사 경로라 폴백이 아닌 본선) ③ 작은 재사용 registered GPU staging buffer + D2D
    ④ 가능 시 cuFile batch API ⑤ baseline: pinned GPU↔CPU DMA + POSIX O_DIRECT
  - 필수 지표: read/write latency p50/p95/p99, effective GiB/s, IOPS, 평균 request/span
    크기, CPU utilization, host memory 왕복 바이트, GPU D2D/H2D/D2H 발생량,
    **nvidia-fs native 카운터**(경로 증명), BAR1/register 실패 기록, cuFile internal
    cache 사용 여부, correctness checksum
  - **runner 레버 주의**: 러너 차이는 compute에도 영향 → raw 마이크로벤치는 pointer
    geometry만 재현; end-to-end는 러너별로 POSIX baseline과 GDS를 각각 비교.
    **V1 GDS vs V2 POSIX를 직접 비교해 GDS 효과라고 주장하지 않는다.**
- **Phase 2 게이트**: 다음 중 하나라도 확인될 때만 vLLM 통합 진행 —
  GDS latency/throughput 우세, 또는 동률이지만 CPU/RAM 대역 절약, 또는 특정 chunk/span
  영역의 명확한 crossover, 또는 BAR1 256MB에서 미등록 GDS가 안정적으로 native path 사용.
- **Phase 2 — `ExperimentalFilesystemSpec` 최소 구현** (Phase 1 게이트 통과 후, 순서 고정):
  ① skeleton ② scheduler-side hash→file mapping ③ worker-side GPU pointer/span resolution
  ④ cuFile handle open/register ⑤ **우선 unregistered cuFile read/write**
  ⑥ registered/staging-buffer 모드 추가 ⑦ store 전 compute completion fence
  ⑧ load 완료 전 model compute 차단 ⑨ store 중 GPU block reuse 방지
  ⑩ job completion/failure propagation ⑪ thread·cuFile resource teardown
  ⑫ temp write 후 atomic commit ⑬ alignment·short I/O 검증 ⑭ 출력 토큰/checksum 검증.
  가능하면 기존 FS on-disk layout과 호환; **호환 layout이 direct I/O geometry를 훼손하면
  호환 layout과 GDS-optimized layout을 구분 측정·문서화.** GPU pointer resolution 복제는
  upstream-ready가 아닌 성능 검증용 기술부채임을 명시.
- **Phase 3 — 비교군 (v4)**:
  - A. recompute (filesystem 미사용)
  - B. CPU primary hit (CPU→GPU)
  - C. 기존 Tiering FS (SSD→CPU→GPU, 현행 vLLM 전체 설계)
  - D. standalone GDS filesystem (SSD→GPU)
  - E. (가능 시) standalone FS + PosixBounceTransport — control plane을 D와 동일하게
    유지, transport만 CPU bounce
  - 해석: A vs C/D = SSD KV load가 재계산보다 유리한가 / **C vs D = 현행 native 구조 vs
    GDS 구조(핵심 판정)** / E vs D = control plane 고정 transport-only 비교.
  - **store와 load는 반드시 별도 결론**: store는 CPU 홉 제거 대신 느린 disk write 동안
    GPU 블록을 더 오래 점유할 수 있음 / load는 FS→GPU 직행으로 promotion latency 단축
    가능성 / 최종 판정은 TTFT·동시 처리량·scheduler stall. `jobs_to_flush` 대기 시간과
    store 중 GPU 블록 비재사용 시간 포함.

### 2.5 재료·제약

- `nvidia-cufile` 1.15.1.6 (pip, gdsllm env), 시스템 libcufile(CUDA 12.8), nvidia-fs 2.25.7 (native 개통 상태)
- rain: RTX 5000 16GB(sm75), BAR1 256MB, 루트 NVMe ext4(MOFED nvme — GDS native 검증 완료 경로)
- 디스크 여유 313GB (2026-08-31 확인). VRAM 16GB가 실질 제약.
- 관측 수단: 오프라인 실험은 `VLLM_ENABLE_V1_MULTIPROCESSING=0` + 몽키패치 계수가 유효
  (fs_hit.py에서 확립). 서빙 실험은 `vllm:kv_offload_{load,store}_{bytes,time,size}` Prometheus 메트릭.

## §3. 진행 로그

### Phase 0 스모크 (2026-08-31) — 통과

- `phase0/smoke.py`: opt-125m + TieringOffloadingSpec + fs 티어, 오프라인 API.
  run1에서 chunk 37개 store, run2 prefix hit(0.455s→0.355s — 단 CPU/FS hit 미구분, 아래서 분리).
- **chunk 기하 실측 검증**: 파일 크기 589,824B = 2(K,V)×16tok×12head×64dim×2B×12layer —
  손계산과 정확히 일치. `blocks_per_file: 1`, 경로에 run-config digest 포함.
- 함정: **conda libstdc++ 선로드 필수** (`env.sh`) — 시스템(20.04) libstdc++가
  conda libicui18n의 CXXABI_1.3.15 요구를 못 채워 vllm import 실패.
- sm75 어텐션: XFORMERS 강제 불필요 — TRITON_ATTN 자동 선택으로 동작.

### GPU span 기하 판정 (2026-08-31) — `phase0/check_layout.py`

- 게이트 4조건(kv transfer group, prefer_cross_layer, 단일 attn group, indexes_kv_by_block_stride)
  전부 True인데 **V2 러너(기본)가 cross-layer 미구현**(`gpu/kv_connector.py:54` TODO)이라 무시.
- `VLLM_USE_V2_MODEL_RUNNER=0`(V1 러너)로 cross-layer 발동 확인:
  "Allocating a cross layer KV cache of shape (13268, 12, 12, 16, 128)" —
  HND layered, 전 레이어 단일 스토리지(7.8GB), 레이어 오프셋 4096B, **블록당 576KiB 연속 span**.
- 결론: 같은 빌드에서 러너 선택으로 Case A(연속)/Case B(단편) 모두 실현 가능. §2.1' 반영.

### FS-hit 분리 증명 (2026-08-31) — `phase0/fs_hit.py`

- store 프로세스: chunk 37개 저장 + 출력 토큰 기록. **완전히 새 프로세스**(CPU primary 빈 상태)에서
  같은 프롬프트 실행: `load_block` **37회 호출**(저장분 전량 FS→CPU 프로모션), `store_block` 0회,
  **출력 토큰 완전 일치**. → run2 가속이 CPU hit가 아니라 FS hit임을 직접 증명 + KV 정합성 검증.
- 관측 기법: `VLLM_ENABLE_V1_MULTIPROCESSING=0` in-process 엔진 + 엔진 생성 전
  `fs.manager.load_block/store_block` 몽키패치 계수 — Phase 3에서도 재사용 예정.

### Phase 0.5 하네스 구축 + 메커니즘 검증 (2026-08-31) — `phase05/bench.py`

- **레버 발견 (소스 검증)**: `LLM.reset_prefix_cache(reset_connector=...)` —
  False면 GPU prefix cache만 리셋(→B군: CPU hit 강제), True면 connector의 CPU primary까지
  리셋하되 **FS 등 persistent 2차 티어는 의도적으로 보존**(`tiering/manager.py:775-785`)
  (→C군: 같은 프로세스에서 SSD hit 강제). fs load가 O_DIRECT라 페이지캐시 混入 없음.
- **함정 발견: store 캐스케이드는 스케줄러 스텝에서만 흐른다.** CPU→FS enqueue가
  `update_connector_output` 경유라 max_tokens=1 요청 종료 후엔 엔진이 idle이면 store가
  다음 generate까지 정체 → 측정 오염. 해법: 16토큰 미만(chunk 미생성) 더미 요청으로
  스텝을 공급하는 **nudge-drain**. (이전 smoke가 store를 보인 건 32토큰 디코드 덕.)
- 메커니즘 검증 (opt-125m, P∈{512,1024,2032}, rep 3): **path_check 전 구간 통과**
  (A: fs/cpu load 0, B: cpu만, C: fs 로드 발생).
- opt-125m 프리뷰 수치(참고용 — prefill이 거의 공짜인 극소 모델): B(CPU hit)는 P≥1024에서
  A를 이김(P=2032: 0.032s vs 0.096s, 3×), **C(SSD hit)는 전 구간 A보다 느림** → RQ1의
  예상대로 "SSD KV는 prefill이 비싼 모델에서만 성립" 방향. 본 판정은 opt-2.7b+에서.

### Phase 0.5 synthetic 게이트 — opt-2.7b (2026-08-31) — **C<A 성립**

- rep 3, path_check 전 구간 통과. 평균(TTFT proxy):
  | P | A 재계산 | B CPU hit | C SSD hit | C<A | H_min |
  |---|---|---|---|---|---|
  | 512 | 0.331s(중앙값 ~0.145, rep0 워밍업 이상치) | 0.049s | 0.146s | 경계(중앙값 기준 동률) | 1 |
  | 1024 | 0.365s | 0.070s | 0.256s | **성립** | 2 |
  | 2032 | 1.155s | 0.110s | 0.483s | **성립 (2.4×)** | 1 |
- 해석: rain+opt-2.7b에서 **P≥1024이면 SSD KV load가 prefill 재계산보다 빠르고 재사용 1~2회면
  store 비용 회수**(synthetic 상한 조건). P=512는 경계 — prefix 길이 하한 존재 확인.
- **C−B 갭(1024: 0.19s / 2032: 0.37s)의 올바른 해석**: C−B ≈ SSD read + filesystem
  lookup/promotion/control 비용. 이 중 **NVMe read 자체는 D(GDS)도 지불한다** — 따라서
  C−B를 "GDS가 제거할 비용"이나 "GDS 개선 상한"으로 쓰면 안 된다. GDS가 줄일 수 있는
  것은 CPU bounce buffer, host memory bandwidth, 추가 H2D, 일부 CPU/CUDA sync,
  pinned CPU memory 사용량이며, **실제 개선폭은 C vs D 직접 비교로만 판정한다.**
- 이 결과가 확인한 것: SSD load **feasibility**(재계산 대비 유리 구간 존재). 확인하지
  않은 것: GDS 개선폭. workload 정당성은 W1/W2(§0.5.4)에서 end-to-end로 재현.

### Phase 1 — cuFile 마이크로벤치 (2026-08-31) — **게이트 통과**

**스택 함정 2개 (재현 시 필수):**
- `cuda.bindings.cufile`(cuda-python 12.9)은 pip 휠 libcufile(cu12 1.14/cu13 1.15)을
  dlopen → 시스템 1.13과 **이중 로드로 비결정 segfault**. 해법: `phase1/gdslib.py` —
  ctypes로 시스템 `/usr/local/cuda/.../libcufile.so.0`만 단일 로드 (Phase 2 transport의
  예행이기도 함).
- rain의 nvidia-fs는 **"IO stats: Disabled"** — per-IO Ops 카운터가 0에 고정
  (`/sys/module/nvidia_fs/parameters/rw_stats_enabled`, sudo 필요, 미적용 상태).
  대체 증거: ① `Bar1-map`/`Mmap` 카운터(등록 이벤트) ② **cufile.log TRACE 분류**
  (`phase1/path_classify.py`): "write_through_bounce_buffer"=내부 바운스,
  cufio-px IO=compat, 둘 다 없음=direct.

**경로 순수성 판정** (TRACE 분류): gds_reg = DIRECT/DIRECT, gds_unreg =
write INTERNAL-BOUNCE(nvidia-fs 내부 GPU 바운스, 예상대로) / read DIRECT(동적 피닝 추정),
gds_staging = DIRECT/DIRECT. **compat-POSIX 폴백 0건** — 전 행렬 native 스택 위에서 측정됨.

**행렬 결과** (27 geometry × 4 transport × 2 op, `phase1/results.csv`, 체크섬 216/216 통과):
- **등록 실패 0건** — op별 등록(쓰기=src만/읽기=dst만)으로 BAR1 256MB에서 **128MiB span까지
  등록 성공**. "BAR1 때문에 등록 불가" 우려는 동시 등록량 관리로 해소 가능함을 실측.
- **1-span (cross-layer 기하 = V1 러너)**: gds_reg 평균 write 2.44 / read 2.71 GiB/s vs
  posix 2.14 / 2.23 → **write +14%, read +22%** (최대 read 3.2~3.3 GiB/s, +35%).
- **다중 span + 작은 조각 (≤2.25MiB chunk의 L/2L = 24~192KiB 조각)**: POSIX 코얼레싱
  우세/동률 — vLLM 기본(V2 러너, 16tok chunk, 소형 모델) 기하에서는 GDS가 이길 수 없음.
- **crossover**: span 조각 ~1MiB† 이상이면 gds_reg ≥ posix, 1-span이면 명확히 우세.
  († 8MiB chunk의 L=32 조각 256KiB에서는 posix 우세, 32MiB chunk의 L 조각 1MiB부터 동률↑)
- **CPU**: gds_reg cpu_util 0.29(w)/0.33(r) vs posix 0.40/0.46 — **~30% 낮음** + posix는
  chunk당 host RAM 왕복 2×chunk 바이트를 추가 소모(GDS 0).
- staging(D2D 경유)은 전 구간 최하위 — 동기 D2D 비용이 큼. cuFile batch API는 v1 미측정.

**게이트 판정: 통과** — 기준 ①(1-span·대형 span에서 GDS 처리량 우세) ③(명확한 crossover:
span 조각 크기) ④(미등록 GDS가 BAR1 256MB에서 안정적 native — bounce/direct, compat 0건)
충족, ②(CPU/host RAM 절감)도 지지. → **Phase 2 통합 진행.**
함의: GDS 이득은 **V1 러너(cross-layer) 또는 큰 blocks_per_chunk**와 결합할 때 실현되고,
기본 V2 러너 + 소형 모델 기하에서는 POSIX 코얼레싱이 옳다 — 러너/chunk 설정이 Phase 3의
1급 실험 변수임이 확정.

### Phase 2 — ExperimentalFilesystemSpec 구현 (2026-08-31) — **완료·검증 통과**

`phase2/expfs.py` (~380줄, out-of-tree): `spec_module_path="expfs"`로 vLLM 무수정 로드.
범위: TP=1·단일 KV group·**blocks_per_chunk=1**(assert로 강제)·성능 검증용 prototype.

- **구조**: FilesystemManager(scheduler-side, FileMapper 재사용, 용량 정책 없음 — lookup=
  파일 존재, 원자적 rename 덕에 exists=loadable) + FilesystemWorker(worker-side, canonical
  텐서→span 해석, `DualQueueThreadPool` 재사용) + Transport 2종(CuFileTransport /
  PosixBounceTransport — Phase 3 E군용, control plane 완전 동일).
- **14단계 체크리스트**: ①~⑤⑦~⑭ 구현+검증. ⑥은 registered_tensors 모드(HiFC 방식
  전체 KV 텐서 등록)로 구현 — opt-125m KV풀 7.8GB 등록 시도 → **err=5016 실패 →
  unregistered 폴백 정상 동작 실증**(BAR1 분기). staging 모드는 Phase 1 전 구간 최하위라
  구현 제외(기록). fence: store=CUDA event record→IO 스레드 synchronize,
  load=동기 cuFileRead+완료 보고 계약, 블록 재사용=wait(job_ids)+커넥터 jobs_to_flush.
- **검증 결과** (opt-125m, 오프라인 in-process):
  | 시험 | 결과 |
  |---|---|
  | cufile store→새 프로세스 load | 37 chunk 저장, **read_chunk 37회(SSD→GPU 직행)**, 재저장 0, **토큰 일치** |
  | posix transport 왕복 | 동일 통과 (37/37/일치) |
  | V1 러너(cross-layer) | canonical 텐서 1개 → **chunk=1 span**(GDS 유리 기하), 28 store/28 load/일치 |
  | 파일 기하 | 589,824B = tiering-fs 티어와 **바이트 단위 동일**(4K 정렬 시) |
  | 경로 순수성(TRACE) | load 37건 전부 **DIRECT** — bounce 0, posix-pool 0, per-op compat 0 |
  | 원자성 | stray .tmp 0 |
- **미시험 항목**(정직 고지): preemption 유발 wait 실경로, IO 실패 주입 시 propagate,
  서빙(multiprocess) 모드의 spec pickle 경로, E2E 성능(→Phase 3).
- V1/V2 러너는 canonical 구성이 달라(1 vs L 텐서) **파일 상호 호환 불가** — root_dir 분리 필수.

### Phase 3 — A~E 비교군 실측 (2026-08-31) — synthetic 매트릭스 완료

`phase3/bench.py`: 구성(none/tiering/expfs-cufile/expfs-posix)×러너(V1/V2)별 별도 프로세스,
opt-2.7b(chunk 5MiB), P∈{1024,2032}×rep3 + 8-prefix 동시 load. path_ok 실패 0.
**러너 간 교차 비교 금지 규율 준수** — 아래 표는 러너 내부 비교만 유효.

**TTFT 중앙값(초), P=2032** (P=1024도 동일 순위):

| | A 재계산 | B CPU hit | C Tiering FS | D expfs+GDS | E expfs+posix |
|---|---|---|---|---|---|
| **V2 러너**(기본, 32-span) | 1.134 | 0.110 | **0.449** | 0.494 | 0.904 |
| **V1 러너**(cross-layer, 1-span) | 1.152 | 0.103 | 1.765 | 0.867 | **0.321** |

**동시 8-load(P=1024×8, 총 2.5GiB)**: 재계산 ≈2.8~3.3s / V2: C 1.43 < E 1.64 < D 1.97 /
V1: C 1.26 < E 2.11 < D 3.01 — **동시성에서는 두 러너 모두 C가 최선** (HiFC의 "동시성이
지연을 은폐" 예측 실증).

**store**: 전 구성 wall 동률(P=2032 ≈0.27s), warm TTFT 영향 ≤3% — 비동기 store에서 GDS는
이득도 해도 없음.

**판정 (store/load 분리):**
- **store**: GDS 동기 없음 — 비동기+코얼레싱으로 이미 은폐됨.
- **load 단일 스트림**: V2에서 C≈D(D가 9% 느림 — Phase 1 소조각 예측 그대로).
  V1에서 D가 C를 2.04× 이기지만, **E-대조군이 원인을 분해한다**: 같은 control plane의
  posix bounce(E)가 D보다 2.7× 빠름 → D의 對C 우세는 **standalone control plane 효과가
  대부분이고 transport로서의 cuFile은 오히려 손해**. "GDS 직행" 자체의 이득은 이 환경에서
  미실현.
- **load 동시**: C 최선 — SSD 티어 자체는 유효(재계산 대비 ~2×)하나 GDS가 낄 자리 없음.
- **B(DRAM 1차 티어)는 전 조건 지배**(0.10s) — vLLM CPU-staged 설계의 정당성 실측 확인.
- **종합 = §0 결과 3분기 중 ②**: "SSD tier 유효 + CPU-staged 설계가 합리적" — 단
  **per-chunk-file 레이아웃 + 미등록 IO + 엔진 상호작용**이라는 현 구현 조건 하에서.
  Mooncake이 FilePerKeyBackend를 "성능용 아님"으로 명시한 것, HiFC의 순차 대형 레이아웃
  선택이 실측으로 뒷받침됨.

**미해명 2건 → nsys로 해명 완료 (2026-08-31, 아래 별도 절):**
1. ~~in-engine D 4.5× 갭~~ → **GIL 콘보이**로 확정.
2. ~~V1 C anomaly~~ → 동일 병인.

### Phase 3-진단 — nsys로 in-engine 갭 원인 확정 (2026-08-31)

도구: `phase3/prof_standalone.py`(엔진 밖 동일 기하 재현) + `phase3/prof_engine.py`
(cudaProfilerApi로 load 구간만 캡처), NVTX는 expfs 상시 계측.
perf_event_paranoid=4라 CPU 샘플링 불가 → cuda,nvtx,osrt 트레이스.

**결정적 관측: nsys를 붙이면 빨라진다.**
| 측정 | nsys 없음 | nsys 있음 |
|---|---|---|
| in-engine D (V1, 127×5MiB) | 1.223s (9.6ms/chunk) | **0.268s (2.1ms/chunk)** |
| in-engine C (V1) | 2.531s | **0.649s** |
| standalone 동일 기하 | 0.188s (1.5ms/chunk) | 동일 |

nsys 하에서 in-engine D ≈ standalone (per-task NVTX 분포도 유사: 23 vs 26ms,
8-way 중첩 포함) — 엔진 고유 병목이 프로파일러 개입만으로 소멸.
추가 검증: `sys.setswitchinterval(0.0005)`만으로 D 1.223→0.531s (2.3×).

**원인 = GIL 콘보이**: 엔진 busy loop(메인 스레드)이 GIL을 스위치 인터벌(기본 5ms)
단위로 독점 → IO 스레드가 chunk당 여러 Python 구간(open/handle_register/read 랩퍼)마다
GIL 재획득에서 인터벌 단위로 굶음. nsys의 osrt 개입은 메인 루프에 시스템콜을 끼워
스위치를 유발해 병목을 "치료"함. 관측된 6.8~9.6ms/chunk = 인터벌×크로싱 수와 부합.
V1 C가 V2 C보다 느린 anomaly도 같은 병인(nsys로 3.9× 가속) — config별 크로싱
패턴 차가 증폭 정도를 가름. E의 device-wide synchronize 영향은 미조사(한계로 기록).

**함의**: ① Phase 3 절대값들은 GIL 콘보이를 포함한 값 — 상대 비교(같은 러너·같은
크로싱 패턴)는 유효하나, 콘보이 민감도가 config별로 달라 D가 과대 불이익을 받았을
수 있음. ② 처방 = **바이트당 GIL 크로싱 수 축소** — blocks_per_chunk 확대 + 인접
GPU 블록 span 병합(expfs v2에 구현). ③ 근본 처방은 IO 제출의 C-레벨 배치화
(cuFile batch API) 또는 엔진 루프의 GIL 양보 — upstream 이슈 소재.

### Phase 3-재대결 — bucketed/chunk 확대 (expfs v2) — **완료, D 역전**

- expfs v2: blocks_per_chunk>1 지원(파일 = [tensor: bpc pages] 연결) + **인접 GPU 블록
  span 병합**(id 연속 시 단일 대형 IO — V1 러너+연속 할당이면 chunk당 IO 1회).
  partial chunk(skip>0) load 처리, posix transport는 불연속 span 가드.
  스모크: b16 회귀·b64 cufile/posix 왕복 토큰 일치.
- ~~발견: 현행 Tiering(C)은 block_size≥64에서 vLLM 자체 크래시~~ → **오판, 정정
  (2026-08-31 W1 중)**: 진범은 크래시한 이전 런들이 누적시킨 `/dev/shm/vllm_offload_*.mmap`
  누출로 tmpfs(63G) 고갈 → `MADV_POPULATE_WRITE`가 페이지 확보 실패로 EFAULT.
  **깨끗한 shm에서 tiering b64 정상 동작 확인.** 따라서 아래 표의 "(불가)" 셀은
  측정 가능 — C-b64/b256 백필은 후속 과제. **새 함정(위생 필수)**: 엔진이 비정상/
  일부 종료 경로로 끝나면 mmap이 남는다 — 매 런 전 `rm /dev/shm/vllm_offload_*.mmap`.
- 결과 (opt-2.7b, P=2032 단일 TTFT 중앙값 / conc8 총시간; V1 D·E는 n=7 보강):

| 러너 | 설정 | C tiering | D gds | E posix | 단일 승자 | conc8 승자 |
|---|---|---|---|---|---|---|
| V1 | b16 | 1.765 | 0.867 | **0.321** | E | C 1.26 |
| V1 | b256 | (불가) | **0.513** (min 0.449) | 0.582 | **D** | **D 0.85~0.95** (C-b16 1.26, E 1.35) |
| V2 | b16 | **0.449** | 0.494 | 0.904 | C | C 1.43 |
| V2 | b64 | (불가) | **0.382** | 0.570 | **D** | C-b16 1.43 (D 1.70) |

- **진단→처방 검증 성공**: b16에서 E가 D를 2.7× 이기던 것이 bucketed에서 역전 —
  V1 b256 단일 D/E 1.13×·conc 1.5×, V2 b64 단일 1.49× D 우위. GIL 크로싱 축소가
  cuFile의 상대 열세 원인이었음을 처방으로 재확인.
- store wall도 D-b256이 최선(0.234~0.244s). 잔여 캐비앗: V1 단일 스트림 분산 큼
  (b64 D 0.30~2.75 — GIL 콘보이 잔재; b256이 안정), V2 conc는 여전히 C 우위.
- **러너별 승자 (교차 비교 금지 규율)**: V1 → **D(expfs+cufile, b256)** 단일·동시 석권
  (대 C-V1 단일 3.4×, conc 1.4×) / V2 → 단일 D-b64(대 C 1.17×), 동시 C-b16.
- **W1/W2 진출 설계: expfs+cuFile+b256+V1 러너**, 대조군은 각 러너의 C-b16(현행 최선).

### W1 — Bailian coder trace replay (2026-08-31, 진행 중) — 방법·사전 계산

- 프레임(명시): **각 설계의 최선 구성 간 end-to-end 비교**(순수 transport 비교 아님).
  V1: D-b256 vs C-b16 / V2: D-b64 vs C-b16. 러너 간 수치 직접 비교 금지.
- 방법: `w1/replay.py` — closed-loop 순차 리플레이(도착 순서 보존, 시간 간격 미재현),
  hash_id→결정적 16-tok 블록 재구성(공식 replayer 원리), 2032-tok 절단(rain 제약),
  max_tokens=1(한계: decode 부하 부재), 동일 600요청·동일 순서·동일 GPU 풀·SSD 무제한,
  C는 현행 그대로 +CPU 8GiB(티어 하나 더 있는 구성임을 명시).
- 계측: TTFT, storage-matched tokens(connector 패치), num_cached_tokens(GPU/storage
  분해), SSD read/store bytes·IOs, CPU time, host-traffic(유도).
- **trace 사전 계산(600요청 윈도, chain-hash 기준)**: 재사용률은 granularity에 거의
  불변 — b16 35.3% / b64 34.2% / b256 32.6% (재사용이 multi-turn full-prefix 연장이라
  chunk 정렬에 둔감). unique KV는 **~190–210GB (opt-2.7b, 전 granularity)** —
  GPU(~6GB)·CPU(8GiB)를 25×+ 자연 초과 → SSD 티어에 인위적 압박 불필요.
  실측 검증: v2-D-b64 저장 193GB ≈ 예측 207GB.
- **운영 함정 2개(이 규모가 드러낸 것)**: ① 디스크 — run당 ~200GB 저장, kvroot를
  런 종료마다 삭제 + free<25GB 가드 필요 ② /dev/shm mmap 누출(위 정정 참조).

### W1 결과 (2026-08-31) — **판정: V2-D 실전 승리 / V1-D 실전 패배 (synthetic 반전)**

**vLLM 버그 발견 #2**: V1 러너 + tiering + 본 trace에서 요청 ~80 부근 재현성 크래시(2회:
req 84, 79) — `_build_store_jobs`가 `finished_req_ids`를 순회하는데 매니저가
`on_request_finished`에서 `_req_state`를 이미 삭제한 뒤 마지막 chunk `prepare_store` 도착
→ KeyError(`tiering/manager.py:542`). **종료-시점 레이스, 업스트림 보고 후보.**
v1-C는 문서화된 가드(미지 req_id store 스킵)로 완주 — guard 발화 10회/약 3.9만 store
(영향 무시 가능).

**본표** (600요청 동일 순서, opt-2.7b, 러너 내부 비교만):

| | TTFT p50/p95/p99 | tok/s | GPU/storage/계산 hit% | SSD read (IOs) | CPU s/req | host 왕복 |
|---|---|---|---|---|---|---|
| **V1-C-b16** | **1.165**/3.641/5.080 | **1140** | 14.1/**20.9**/65.0 | 59.6GiB (12,212) | **0.94** | 496GiB |
| V1-D-b256 | 1.880/4.133/5.828 | 830 | 14.1/18.0/67.8 | 54.2GiB (**713**) | 4.44 | **0** |
| V2-C-b16 | 1.160/3.107/4.043 | 1392 | 14.1/**21.2**/64.7 | 56.1GiB (11,488) | **0.67** | 498GiB |
| **V2-D-b64** | 1.142/**2.090/2.205** | **1596** | 14.1/20.5/65.4 | 61.5GiB (**3,191**) | 8.23 | **0** |

- **V2: D-b64 승** — p50 동률, **p95/p99 1.49×/1.83× 우위, 처리량 +14.7%**, IO 3.6×↓,
  host DRAM 왕복 498GiB→0. 대가: CPU 12×(store의 미등록 cuFile write 내부 바운스).
- **V1: C-b16 승 (synthetic 재대결 결과의 반전)** — D-b256이 p50 1.61× 뒤지고 처리량 −27%.
  반전의 기전: ① b256 조립도가 storage-hit를 깎음(18.0% vs 20.9% → 재계산 +2.8%p)
  ② 실전은 store(178GB)가 서빙과 **연속 동시 진행** — 80MiB 미등록 cuFile write의
  CPU/GIL 간섭이 전체 분포를 오염(miss까지 느려짐) ③ synthetic은 store-drain 후 load만
  측정하는 구조라 이 간섭이 보이지 않았음. **외적 타당성 검증이 정확히 제 역할을 함:
  synthetic 승자 ≠ production 승자.**
- 관찰: C의 storage-hit 요청 p50(2.13)이 miss(1.15)보다 느림 — hit의 이득은 "동일 요청
  재계산 대비"로만 판정 가능(counterfactual 부재), TTFT 절대값으로 hit 손익 단정 금지.
- **W2 진출: V2 + D-b64** (실전 검증). V1-D-b256은 store 간섭 처방(등록/스테이징 write,
  write 스레드 수, store 백프레셔) 전까지 보류.
- 한계: max_tokens=1(decode 부하 부재), 2032-tok 절단, closed-loop 순차(동시성 무),
  guard 10건, C는 CPU 8GiB 추가 티어 보유 구성.

**한계**: E transport의 per-chunk `torch.cuda.synchronize()`(device-wide)가 V2 E를
과도하게 불리하게 만들 수 있음(V2의 E 최하위는 구현 탓 가능성). conc의 none 구성 hit는
GPU prefix hit라 재계산 대조는 warm 값 사용. W1(Bailian replay)/W2(LEval) end-to-end는
미실시 — synthetic 매트릭스만으로 낸 결론임.

**GDS가 이기기 위한 조건 (Phase 3가 도출한 다음 단계)**: ① bucketed/대형 파일 레이아웃
(핸들·파일 수 감소 + blocks_per_chunk>1) ② 엔진 상호작용 갭 해명 ③ registered-buffer
경로(BAR1 내 등록 풀 스테이징) — 이 셋 없이 per-chunk cuFile은 잘 만든 bounce를 못 이긴다.
