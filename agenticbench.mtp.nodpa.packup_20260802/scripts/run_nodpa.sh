#!/bin/bash
# Drive the noDPA arm: the SAME concurrency-1 workload lat1 ran, against a
# prefill leg with DP-attention turned OFF.
#
# Deliberately byte-identical in shape to run_lat1.sh, and the workload YAML is
# byte-identical to lat1_full.yaml except for random_seed. That is the whole
# design: lat1 varied the client and held the server fixed; this run does the
# exact opposite -- holds the client fixed and moves ONE server flag -- so the
# delta between the two is attributable to prefill DP-attention.
#
# --dashboard-mode IS MANDATORY. summary.json, metrics.jsonl and metadata.json
# are all written inside `if dashboard_mode and benchmark_name and data_dir:`
# (agent_throughput.py:2067). Without it the run prints a full report to stdout,
# exits 0, and persists nothing structured. It has already cost one run here.
#
# Usage: run_nodpa.sh <probe|full> [name]
set -u
VARIANT="${1:?probe|full}"
NAME="${2:-nodpa_${VARIANT}}"

BENCH=/home/yihou/dev/git/Optimus-AgenticBench
PY="${PY:-/shared_nfs/yihou_agentbench/venv/bin/python3}"
WL=/home/yihou/dev/git/infera.merge.liying.kv.mtp/work.agenticbench.mtp/workloads/nodpa_${VARIANT}.yaml
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
ROUTER="${ROUTER:-http://10.245.150.172:8190}"
OUT=/shared_nfs/yihou_agbench_mtp/bench/${NAME}

mkdir -p "$OUT"
cd "$BENCH" || exit 1

echo "[nodpa] variant=$VARIANT router=$ROUTER out=$OUT"
echo "[nodpa] workload=$WL"
echo "[nodpa] driver=$(git -C "$BENCH" rev-parse --short HEAD)"

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

echo "[nodpa] artifacts:"
find "$OUT" -maxdepth 3 -type f -newermt '-4 hours' | sed 's/^/    /' | head -20
