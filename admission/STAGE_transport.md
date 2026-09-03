### 결론 ② transport 분리 (champion seen_twice, cuFile vs POSIX staging)

| 러너 | transport | p95 | CPU | hit | write |
|---|---|---|---|---|---|
| V2 b64 | cuFile-staging | 1.17 | 22s | 10,976 | 2.7GB |
| V2 b64 | posix-staging | 1.17 | 21s | 10,976 | 2.7GB |
| V1 b256 | cuFile-staging | 1.17 | 17s | 10,560 | 3.2GB |
| V1 b256 | posix-staging | 1.16 | 17s | 10,560 | 3.2GB |

결론 ②: cuFile = POSIX (전 지표 동일, 양 러너). value-admission+비차단 staging에서
write는 임계 경로 밖(비동기 writer)이라 transport 종류가 foreground에 무관.
이득 원천은 transport가 아니라 store 실행 구조(수명 분리+admission)임을 재확증.
