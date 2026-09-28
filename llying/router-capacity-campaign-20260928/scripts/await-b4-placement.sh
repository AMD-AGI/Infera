#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
python3 "$ROOT/scripts/review_point.py" "$ROOT/runs/campaign-rb-rebaseline" /home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928/rb --wait
export CONFIG=$ROOT/config/b4-real.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/transition_placement.py"
python3 "$ROOT/scripts/placement_answer_probe.py"
export CONFIG=$ROOT/config/b4.sh
set -a; source "$CONFIG"; set +a
python3 "$ROOT/scripts/transition_placement.py"
python3 "$ROOT/scripts/preflight_placement.py"
python3 "$ROOT/scripts/run_placement_performance.py"
python3 "$ROOT/scripts/review_point.py" "$RUN" /home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928/b4
