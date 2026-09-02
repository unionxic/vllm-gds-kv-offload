# perf 단계 (rootless 결과로 원인이 안 갈릴 때만)

Claude가 명령을 준비하고 사용자가 rain 터미널에서 직접 sudo 인증 후 실행한다.
현재 계정은 sudoers 미등록으로 보이므로(sudo -n 불가, 그룹 unionxic뿐) 관리자
권한이 실제로 있는지 `sudo -v`로 먼저 확인할 것.

## 0. 기존 설정 기록 (시험 후 복구용)

```bash
cat /proc/sys/kernel/perf_event_paranoid   # 현재 4
cat /proc/sys/kernel/kptr_restrict
cat /proc/sys/kernel/yama/ptrace_scope     # 현재 1
```

## 1. 완화 (시험 동안만)

```bash
sudo sysctl kernel.perf_event_paranoid=1
sudo sysctl kernel.yama.ptrace_scope=0     # py-spy 임의 attach가 필요할 때만
```

## 2. 수집 — 벤치가 도는 동안 별도 셸에서

```bash
# 스레드 실행·대기·context switch (30초 창)
sudo perf sched record -p <run_bench PID> -- sleep 30
sudo perf sched latency --sort max > sched/prof/perf_sched_latency.txt
sudo perf sched timehist > sched/prof/perf_sched_timehist.txt

# native 스택·CPU hotspot (컴파일드 구간: libcufile, CUDA driver, memcpy)
sudo perf record -g -F 499 -p <run_bench PID> -- sleep 30
sudo perf report --stdio > sched/prof/perf_report.txt
```

## 3. nvidia-fs 통계 (경로 검증 보조용 — 최종 성능 측정에서는 끔)

```bash
echo 1 | sudo tee /sys/module/nvidia_fs/parameters/rw_stats_enabled
cat /proc/driver/nvidia-fs/stats
echo 0 | sudo tee /sys/module/nvidia_fs/parameters/rw_stats_enabled
```

## 4. 복구

```bash
sudo sysctl kernel.perf_event_paranoid=4
sudo sysctl kernel.yama.ptrace_scope=1
