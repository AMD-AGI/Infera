#!/bin/bash
# Drive Optimus-AgenticBench par8 against one arm's router, FROM THE LOGIN NODE.
#
# From the login node, not inside a container: the bench is pure HTTP plus a
# tokenizer, and running the load generator inside the prefill container would
# place it on the host it is measuring.
#
# --dashboard-mode IS MANDATORY, and not for a dashboard. summary.json,
# metrics.jsonl and metadata.json are ALL written inside
#     if dashboard_mode and benchmark_name and data_dir:      (:1674)
# Without the flag the run completes normally, prints a full report to stdout,
# exits 0 -- and persists NOTHING structured. That has already cost one run here.
#
# NO load knobs on the CLI. The YAML is the single source of truth; passing
# --initial-sessions here would silently shadow it and the two arms would no
# longer share a workload.
#
# Usage: ROUTER=http://ip:port ab_bench.sh <name>
set -u
NAME="${1:?run name}"

BENCH=/home/yihou/dev/git/Optimus-AgenticBench
PY="${PY:-/shared_nfs/yihou_agentbench/venv/bin/python3}"
WL="${WL:-/home/yihou/dev/git/infera.merge.liying.kv.mtp/work.final.pr/workloads/par8.yaml}"
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
ROUTER="${ROUTER:?ROUTER=http://<prefill ip>:<port>}"
OUT=/shared_nfs/yihou_final_pr/bench/${NAME}

mkdir -p "$OUT"
cd "$BENCH" || exit 1

echo "[bench] name=$NAME router=$ROUTER out=$OUT"
echo "[bench] workload=$WL"
echo "[bench] md5=$(md5sum "$WL" | cut -d' ' -f1)"

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
find "$OUT" -maxdepth 3 -type f -newermt '-6 hours' | sed 's/^/    /' | head -20
