### 3단계 Bailian 150 V2 b64 판정

| arm | p95 | CPU | hit(matched) | write | hit/GB |
|---|---|---|---|---|---|
| 기존 tiering | 5.28 | 138s | 20,960 | 43.5GB | 482 |
| skip (random_skip) | 1.21 | 22s | 8,480 | 7.3GB | 1,155 |
| seen_twice | 1.17 | 22s | 10,976 | 2.7GB | 4,014 |
| value_density(재보정) | 1.17 | 21s | 8,432 | 2.6GB | 3,198 |

세 결론:
1. value admission이 random보다 개선 — 예(seen_twice). hit +29%(10,976 vs 8,480)를
   write 63%↓(2.7 vs 7.3GB)로 달성, 같은 tail. hit/GB 4,014로 오프라인 oracle 효율
   (2,845~3,372)에 근접. 기전: random은 일회성 chunk에 슬롯 낭비+포화 시 재사용
   chunk 맹목 drop, seen_twice는 재등장 chunk만 저장.
3. 전체 GDS(seen_twice) vs tiering: tail 4.5배·CPU 6배 우위.
   (②cuFile vs POSIX는 champion posix-staging으로 분리 — 다음.)

champion = seen_twice. value_density의 reuse-distance 정교화는 이 워크로드에서 이득
없음(hit 8,432 < seen_twice 10,976): Bailian 재사용은 frequency로 예측되나 거리로
변별 안 됨(워킹셋 200GB≫GPU+CPU 13.5GB라 짧은 거리 재사용도 evict). value_density는
기록으로 보존, champion 아님.
