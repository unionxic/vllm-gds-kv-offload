### 5단계 혼합 workload V1 b256 — V2 확증

| arm | p95 | far_hit | near_hit | rep_hit | 총 |
|---|---|---|---|---|---|
| tiering | ~4.3 | 28,576 | 26,656 | 55,072 | 110,304 |
| random_skip | 2.70 | 2,928 | 1,888 | 7,648 | 12,464 |
| seen_twice | 1.58 | 0 | 0 | 10,416 | 10,416 |
| value_density | 1.70 | 0 | 0 | 10,544 | 10,544 |

V2 발견 러너 무관 재현: seen_twice·value_density far_hit=0(cold-tail 구조적 실패),
random(store-first)이 far 잡음, 총 hit random>seen_twice. C-r2는 종료 race 가드 추가
후 완주(v0.26.0 race, README 한계 명시).
