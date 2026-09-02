#!/bin/bash
# 3단계 프로파일링: nsys와 py-spy는 별도 실행(프로파일러 간섭 방지).
#   usage: run_prof.sh nsys|pyspy|pyspy-gil <name> <run_bench args...>
#   예: run_prof.sh nsys Dsync --design D --policy baseline --block 64
set -eu
cd "$(dirname "$0")/.."
source env.sh
mode=$1; name=$2; shift 2
mkdir -p sched/prof
id="prof-${mode}-${name}"

case "$mode" in
  nsys)
    # paranoid=4 환경: CPU 샘플링·ctxsw perf 이벤트 불가 → sample/cpuctxsw 비활성.
    # GIL은 python-gil trace, Python 스택은 --python-sampling으로.
    PROF_INSTRUMENT=1 nsys profile \
      --trace=cuda,nvtx,osrt,python-gil \
      --python-sampling=true --python-sampling-frequency=250 \
      --sample=none --cpuctxsw=none \
      -o "sched/prof/${id}" --force-overwrite=true \
      python sched/run_bench.py --run-id "${id}" "$@" \
        --note "nsys 프로파일 런 — 성능 수치는 참고용(간섭 있음)"
    ;;
  pyspy)
    PROF_INSTRUMENT=1 ~/miniconda3/envs/gdsllm/bin/py-spy record \
      --rate 25 --threads --format speedscope \
      -o "sched/prof/${id}.speedscope.json" -- \
      python sched/run_bench.py --run-id "${id}" "$@" \
        --note "py-spy 런 — 성능 수치는 참고용"
    ;;
  pyspy-gil)
    # GIL 보유 시점만 샘플 → 어떤 함수가 GIL을 쥐는지
    PROF_INSTRUMENT=1 ~/miniconda3/envs/gdsllm/bin/py-spy record \
      --rate 25 --threads --gil --format speedscope \
      -o "sched/prof/${id}.speedscope.json" -- \
      python sched/run_bench.py --run-id "${id}" "$@" \
        --note "py-spy GIL 런 — 성능 수치는 참고용"
    ;;
  *) echo "mode must be nsys|pyspy|pyspy-gil"; exit 1;;
esac
echo "done: sched/prof/${id}*"
