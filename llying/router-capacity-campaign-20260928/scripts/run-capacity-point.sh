#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
point=${1:?usage: run-capacity-point.sh b5|b6|b7 PREVIOUS_RUN [--restart-prefill]}
previous=${2:?previous completed run is required}
[[ "$point" == b[567] ]] || { echo 'Expected b5, b6, or b7' >&2; exit 1; }
shift 2
python3 "$ROOT/scripts/make_capacity_point.py" "$point" --previous "$previous" "$@"
export CONFIG=$ROOT/config/$point-real.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/transition_placement.py"
python3 "$ROOT/scripts/placement_answer_probe.py"
export CONFIG=$ROOT/config/$point.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/transition_placement.py"
python3 "$ROOT/scripts/preflight_placement.py"
python3 "$ROOT/scripts/run_placement_performance.py"
python3 "$ROOT/scripts/review_point.py" "$RUN" /home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928/$point
