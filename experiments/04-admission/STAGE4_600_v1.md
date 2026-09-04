### 4단계 600요청 확장 V1 b256

| arm | p95 | CPU | hit | write | hit/GB |
|---|---|---|---|---|---|
| tiering | 4.59 | 753s | 202,352 | 188GB | 1,074 |
| skip(random) | 1.49 | 128s | 98,688 | 38GB | 2,605 |
| seen_twice(cuFile) | 1.17 | 95s | 108,384 | 22GB | 4,851 |
| seen_twice(POSIX) | 1.19 | 123s | 109,152 | 22GB | 4,920 |

V2 600과 동일 결론: seen_twice>random(hit +10%·write -41%·tail 우수, hit/GB 1.9배),
cuFile≈POSIX, tiering 트레이드오프. 세 결론 600 양 러너 확증.
