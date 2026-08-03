#!/bin/bash
# Drive Optimus-AgenticBench against the router from the LOGIN NODE.
#
# From the login node, not inside a container: the bench is pure HTTP plus a
# tokenizer, and running the load generator inside the prefill container would
# place it on the host it is measuring.
#
# --dashboard-mode IS MANDATORY, and not for a dashboard. summary.json,
# metrics.jsonl and metadata.json are ALL written inside
#     if dashboard_mode and benchmark_name and data_dir:      (:1674)
# Without the flag the run completes normally, prints a full report to stdout,
# exits 0 -- and persists NOTHING structured. That loses every percentile and the
# whole num_sessions_active time series. It has already cost one run here.
#
# Usage: run_bench.sh <probe|full> [name]
set -u
VARIANT="${1:?probe|full}"
NAME="${2:-caseA_${VARIANT}}"

BENCH=/home/yihou/dev/git/Optimus-AgenticBench
PY="${PY:-/shared_nfs/yihou_agentbench/venv/bin/python3}"
WL=/home/yihou/dev/git/infera.merge.liying.kv.mtp/work.agenticbench.mtp/workloads/caseA_${VARIANT}.yaml
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
ROUTER="${ROUTER:-http://10.245.157.89:8190}"
OUT=/shared_nfs/yihou_agbench_mtp/bench/${NAME}

mkdir -p "$OUT"
cd "$BENCH" || exit 1

echo "[bench] variant=$VARIANT router=$ROUTER out=$OUT"
echo "[bench] workload=$WL"
echo "[bench] python=$PY"

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

echo "[bench] artifacts:"
find "$OUT" -maxdepth 3 -type f -newermt '-4 hours' | sed 's/^/    /' | head -20
