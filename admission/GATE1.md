### 1단계 오프라인 시뮬레이션 게이트 — 통과

Bailian 600 trace, GPU+CPU 2764블록 LRU, prefix-closed 저장, hash_ids[:127] 캡.
지표 hit/GiB = useful external hit 토큰 ÷ 실제 write 바이트(=write 압박=tail 대리).

| geo | random | reuse_distance | value_density | oracle |
|---|---|---|---|---|
| V1 b256 | 633 | 1,183 | 1,180 | 3,015 |
| V2 b64 | 477 | 1,174 | 1,341 | 3,372 |

value/reuse-distance가 random 대비 효율 ~2배(같은 바이트로 2배 hit). wasted write
53~59% vs random 84~87%. oracle이 정상 상한 형성. 주신호=재사용 거리(evict 예측),
seen-twice는 필터. 게이트 통과 → 실제 구현 진행.

한계: 오프라인 전역 랭킹(신호 예측력 측정용). 온라인 근사는 2단계 이후. budget을
chunk 수로 잡아 prefix-closed 확장으로 정책별 실제 GiB가 달라짐 — 그래서 raw hit이
아닌 hit/GiB로 판정.
