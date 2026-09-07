#!/usr/bin/env bash
# QA harness for the prefetch weight-offload SSD tier.
#   ./run_qa.sh                       # all arms: baseline cpu ssd-posix ssd-cufile
#   ./run_qa.sh baseline cpu          # subset (later runs reuse baseline.json for token compare)
#   ./run_qa.sh --nsys ssd-posix ssd-cufile   # profile ssd arms, writes <arm>.nsys-rep + <arm>-nsys.csv
#   KEEP=1 ./run_qa.sh ...            # keep files under results/weight-offload/qa-ssd/<arm>/
# Results: results/weight-offload/qa-ssd/{<arm>.json,<arm>.log,<arm>.stderr.log,<arm>.err,<arm>-cufile.log,env.json,summary.json}
# Native-GDS proof for ssd-cufile = TRACE-level cufile.log classification (lib/path_classify.py rules), not nvidia-fs Reads n=.
# Exit code != 0 if any check FAILs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../env.sh"
# prefetch weight offloader only hooks the V1 model runner; in-process engine so /proc/self == engine
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
exec python "$HERE/qa_ssd.py" "$@"
