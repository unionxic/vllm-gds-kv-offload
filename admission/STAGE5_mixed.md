### 5단계 real-text synthetic-interleaving admission workload (V2 b64)

카테고리별 connector-tier(CPU+SSD offload) useful hit. 128요청, one_shot 16·
near_reuse 32(간격14)·far_reuse 32(간격62)·repeated 48(3회).

| arm | p95 | far_hit | near_hit | rep_hit | 총 |
|---|---|---|---|---|---|
| tiering | 5.17 | 28,576 | 26,656 | 55,072 | 110,304 |
| random_skip | 1.70 | 2,880 | 1,840 | 7,040 | 11,760 |
| seen_twice | 1.51 | 0 | 0 | 10,544 | 10,544 |
| value_density | 1.75 | 0 | 0 | 10,272 | 10,272 |

핵심 발견(Bailian 결론 조건부화): champion은 재사용 구조에 의존.
- seen_twice는 cold-tail(far 1회 재사용) 구조적 실패 — 2번째 등장에서 저장하는데
  far는 2회만 등장, 도움될 3번째 없음. repeated(3회)만 잡음. far_hit=0.
- random_skip은 1번째 등장 저장 → far 2번째 잡음(2,880). 총 hit도 random>seen_twice.
  Bailian(반복형)과 정반대.
- value_density도 seen-twice 게이트로 동일 실패.
결론: 앞선 seen_twice 우위는 반복형 재사용(Bailian coding) 한정. cold-tail 지배
workload(Mooncake 10.3% cold 재사용)에선 store-on-first가 옳고 frequency 필터는 역효과.
올바른 cold-tail 정책 = 1번째 저장 + one_shot만 예측 drop(seen_twice와 정반대 방향).
near_reuse가 offload-tier hit 0인 것은 GPU 캐시가 잡기 때문(matched는 GPU캐시 제외,
connector 티어만 계수) — near는 SSD 무가치가 맞음.
