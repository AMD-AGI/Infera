#!/bin/bash
# Drive Optimus-AgenticBench against the router from the LOGIN NODE.
#
# Runs from the login node, not inside a container: the bench is pure HTTP +
# tokenizer, the venv already has it installed, and driving it from inside the
# prefill container would put the load generator on the same host it is
# measuring.
#
# Usage: run_bench.sh <probe|full> [name]
set -u
VARIANT="${1:?probe|full}"
NAME="${2:-caseA_${VARIANT}}"

BENCH=/home/yihou/dev/git/Optimus-AgenticBench
PY=/shared_nfs/yihou_agentbench/venv/bin/python3
WL=/home/yihou/dev/git/infera.yihou.glm5.2.mxfp4/work.agenticbench/workloads/caseA_${VARIANT}.yaml
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
ROUTER="${ROUTER:-http://10.245.153.38:8190}"
OUT=/shared_nfs/yihou_agentbench/bench/${NAME}

mkdir -p "$OUT"
cd "$BENCH" || exit 1

echo "[bench] variant=$VARIANT router=$ROUTER out=$OUT"
echo "[bench] workload=$WL"

# --dashboard-mode IS MANDATORY, and not for a dashboard: summary.json,
# metrics.jsonl and metadata.json are all written inside
#     if dashboard_mode and benchmark_name and data_dir:
# (agent_throughput.py:1674). Without the flag the run prints its report to
# stdout and persists NOTHING structured -- which loses Goal item 2's
# percentiles and Goal item 4's num_sessions_active time series entirely.
"$PY" -m agent.agent_throughput \
  --mode realistic \
  --workload-config "$WL" \
  --server "$ROUTER" \
  --model glm5.2-mxfp4 \
  --tokenizer "$MODEL" \
  --dashboard-mode \
  --name "$NAME" \
  --data-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo "[bench] exit=$? artifacts:"
find "$OUT" -maxdepth 3 -type f -newermt '-1 hour' | sed 's/^/    /' | head -20
