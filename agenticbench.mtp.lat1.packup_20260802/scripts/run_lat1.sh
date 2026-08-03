#!/bin/bash
# Drive the lat1 (concurrency-1 latency floor) workload against the router.
#
# Same invocation shape as run_bench.sh -- deliberately. The only differences
# from the Case A run are the workload YAML and the output name; the driver,
# the flags, the tokenizer and the router are identical, so any latency delta
# between the two runs is attributable to the workload and not to the harness.
#
# --dashboard-mode IS MANDATORY. summary.json, metrics.jsonl and metadata.json
# are all written inside `if dashboard_mode and benchmark_name and data_dir:`
# (agent_throughput.py:2067). Without it the run prints a full report to stdout,
# exits 0, and persists nothing structured. It has already cost one run here.
#
# Usage: run_lat1.sh <probe|full> [name]
set -u
VARIANT="${1:?probe|full}"
NAME="${2:-lat1_${VARIANT}}"

BENCH=/home/yihou/dev/git/Optimus-AgenticBench
PY="${PY:-/shared_nfs/yihou_agentbench/venv/bin/python3}"
WL=/home/yihou/dev/git/infera.merge.liying.kv.mtp/work.agenticbench.mtp/workloads/lat1_${VARIANT}.yaml
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
ROUTER="${ROUTER:-http://10.245.157.89:8190}"
OUT=/shared_nfs/yihou_agbench_mtp/bench/${NAME}

mkdir -p "$OUT"
cd "$BENCH" || exit 1

echo "[lat1] variant=$VARIANT router=$ROUTER out=$OUT"
echo "[lat1] workload=$WL"
echo "[lat1] driver=$(git -C "$BENCH" rev-parse --short HEAD)"

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

echo "[lat1] artifacts:"
find "$OUT" -maxdepth 3 -type f -newermt '-4 hours' | sed 's/^/    /' | head -20
