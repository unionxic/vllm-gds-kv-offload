### 3단계 Bailian 150 V1 b256 판정

| arm | p95 | CPU | hit | write | hit/GB |
|---|---|---|---|---|---|
| 기존 tiering | 3.73 | 97s | 20,960 | 38.1GB | 550 |
| skip (random) | 1.17 | 22s | 8,768 | 8.9GB | 984 |
| seen_twice | 1.17 | 17s | 10,560 | 3.2GB | 3,297 |
| value_density | 1.17 | 16s | 10,560 | 3.2GB | 3,297 |

V2 결론 확증: seen_twice가 random 이김(hit +20%·write -64%·같은 tail).
value_density = seen_twice(거리 변별 이득 없음, 양 러너 일관). champion=seen_twice.
러너 내부 비교만(V1↔V2 절대치 비교 금지).
