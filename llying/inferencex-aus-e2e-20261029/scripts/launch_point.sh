#!/usr/bin/env bash
# Run one matrix point of the infera-sglang key through the InferenceX CI launcher
# (InferenceY runners/manual_launch.py, smci355 cluster record pointed at Compute-DCPT).
#   launch_point.sh <label> <exp-name> [manual_launch options, e.g. --fast | --duration 3600]
# Log: /apps/liyingli/runs/<label>.log. Detached; survives the calling shell.
set -euo pipefail
label="${1:?label}"; exp="${2:?exp-name}"; shift 2
E2E="${E2E:-/home/liyingli/code/bench_agentx/infx-aus/inferencex-e2e}"
log="/apps/liyingli/runs/$label.log"
mkdir -p /apps/liyingli/runs
# The launcher names job-local checkouts after the workflow run; any unique number works.
run_id="$(date -u +%s)"
cd "$E2E"
{
    echo "label=$label exp=$exp opts=$* run_id=$run_id head=$(git log -1 --format='%h %s') $(date -u +%FT%TZ)"
} > "$log"
export PATH="$HOME/.local/bin:$PATH" SBATCH_QOS=batch GITHUB_RUN_ID="$run_id" GITHUB_RUN_ATTEMPT=1
setsid nohup "$HOME/venvs/infx-launch/bin/python" runners/manual_launch.py "$@" -- test-config \
    --config-files configs/amd-master.yaml \
    --config-keys glm5.2-fp4-mi355x-infera-sglang-agentic-disagg \
    --scenario-type agentic-coding --exp-names "$exp" >> "$log" 2>&1 < /dev/null &
echo "launched pid $! log $log"
