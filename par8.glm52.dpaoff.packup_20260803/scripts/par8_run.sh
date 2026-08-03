#!/usr/bin/env bash
# PAR8 (par8.yaml = Case A request shape at reduced offered load) against the
# chi2835/chi2878 PD deployment with DP-attention OFF on the prefill leg.
#
# Sibling of caseA_run.sh. Two deltas, both because this is a different
# deployment and a different workload file, NOT a different method:
#   * WORKLOAD defaults to par8.yaml (initial_sessions 8 / max_sessions 32 /
#     max_inflight 24; everything else byte-identical to caseA.fix.yaml)
#   * ROUTER defaults to 10.2.122.78 (chi2835), not 10.2.122.10 (chi2879)
#
# NO load knobs are passed on the CLI. The YAML is the single source of truth
# for offered load; passing --initial-sessions here would silently shadow it and
# make the run unreproducible from the file alone.
#
#   TAG=full RAMP=400 SUSTAIN=3600 bash par8_run.sh
set -u
W=/mnt/vast/c_huggingface/bench_20260801
TAG="${TAG:-full}"
RAMP="${RAMP:-400}"
SUSTAIN="${SUSTAIN:-3600}"
ROUTER="${ROUTER:-http://10.2.122.78:8100}"
MODEL="${MODEL:-glm5.2-mxfp4}"
TOKENIZER="${TOKENIZER:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
WORKLOAD="${WORKLOAD:-$W/par8.yaml}"
OUT="$W/results/par8_$TAG"
LOG="$W/logs/par8_$TAG.log"

mkdir -p "$OUT" "$W/logs"

echo "===== PAR8 [$TAG] ramp=${RAMP}s sustain=${SUSTAIN}s -> $OUT ====="
echo "  router:   $ROUTER"
echo "  workload: $WORKLOAD"
curl -sf -m10 "$ROUTER/health" || { echo "ROUTER NOT HEALTHY -- abort" >&2; exit 1; }
echo

cd "$W/agbench" || exit 1
# --dashboard-mode is MANDATORY: without it nothing structured is persisted and
# the run is unrecoverable after the terminal scrolls.
nohup "$W/venv/bin/python" -u -m agent.agent_throughput \
  --workload-config "$WORKLOAD" \
  --server "$ROUTER" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --ramp-duration "$RAMP" \
  --sustain-duration "$SUSTAIN" \
  --dashboard-mode \
  --name "par8_$TAG" \
  --data-dir "$OUT" \
  > "$LOG" 2>&1 &

echo "  pid $! -> $LOG"
echo "  ETA $(( (RAMP + SUSTAIN) / 60 )) min + drain"
