### 4단계 600요청 확장 V2 b64

| arm | p95 | CPU | wall | hit | write | hit/GB |
|---|---|---|---|---|---|---|
| tiering | 5.18 | 768s | 1072s | 206,656 | 189GB | 1,091 |
| skip(random) | 1.52 | 155s | 509s | 108,912 | 32GB | 3,408 |
| seen_twice(cuFile) | 1.32 | 182s | 496s | 114,432 | 16.7GB | 6,845 |
| seen_twice(POSIX) | 1.27 | 173s | 492s | 114,944 | 16.5GB | 6,981 |

규모 확증: ① seen_twice>random(hit +5%·write 절반·tail 우수, hit/GB 2배).
② cuFile≈POSIX 재확인. ③ tiering 대비 tail 4배·CPU 4배·write 11배↓, hit ~55% 트레이드오프.
150 대비 hit 격차 축소(+29%→+5%)·write 효율 격차 확대 — 규모서 admission 주이득은
write 볼륨 절감(→압박·tail↓). random도 규모 크면 재등장셋을 점진 포착해 hit 격차 좁힘.
