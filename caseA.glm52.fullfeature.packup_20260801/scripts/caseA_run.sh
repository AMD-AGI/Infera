#!/usr/bin/env bash
# Case A (Optimus-AgenticBench glm52_crxx_caseA.fix.yaml) against the merged-e
# PD deployment. Runs FROM THE JUMP HOST, not inside a node container -- the
# driver is a pure HTTP client and the jump host can reach the router rail.
#
# The `profile:` block in the YAML auto-selects realistic (closed-loop) mode,
# so there is deliberately NO --mode flag here. Passing one would be wrong:
# 'preview' is a dry run *of* the selected mode, not a competing mode.
#
# CLI overrides YAML (merge_workload_yaml checks argparse defaults), which is
# how SUSTAIN/RAMP/SESSIONS/RATE below shorten a probe without editing caseA.yaml.
#
#   TAG=probe SUSTAIN=600 bash caseA_run.sh
#   TAG=full                bash caseA_run.sh      # full 400+3600 as shipped
set -u
W=/mnt/vast/c_huggingface/bench_20260801
TAG="${TAG:-probe}"
RAMP="${RAMP:-400}"
SUSTAIN="${SUSTAIN:-600}"
ROUTER="${ROUTER:-http://10.2.122.10:8100}"
MODEL="${MODEL:-glm5.2-mxfp4}"
TOKENIZER="${TOKENIZER:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
OUT="$W/results/caseA_$TAG"
LOG="$W/logs/caseA_$TAG.log"

# Only pass these when explicitly set -- an unset var must fall through to the
# YAML value, and argparse's "did the CLI change it?" test is by default value.
EXTRA=()
[ -n "${SESSIONS:-}" ] && EXTRA+=(--initial-sessions "$SESSIONS")
[ -n "${RATE:-}"     ] && EXTRA+=(--new-session-rate "$RATE")
[ -n "${INFLIGHT:-}" ] && EXTRA+=(--max-inflight "$INFLIGHT")

mkdir -p "$OUT" "$W/logs"

echo "===== Case A [$TAG] ramp=${RAMP}s sustain=${SUSTAIN}s -> $OUT ====="
echo "  router: $ROUTER"
curl -sf -m10 "$ROUTER/health" || { echo "ROUTER NOT HEALTHY -- abort" >&2; exit 1; }
echo

cd "$W/agbench" || exit 1
# --dashboard-mode is MANDATORY: without it nothing structured is persisted and
# the run is unrecoverable after the terminal scrolls.
nohup "$W/venv/bin/python" -u -m agent.agent_throughput \
  --workload-config "$W/caseA.yaml" \
  --server "$ROUTER" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --ramp-duration "$RAMP" \
  --sustain-duration "$SUSTAIN" \
  --dashboard-mode \
  --name "caseA_$TAG" \
  --data-dir "$OUT" \
  "${EXTRA[@]}" \
  > "$LOG" 2>&1 &

echo "  pid $! -> $LOG"
echo "  ETA $(( (RAMP + SUSTAIN) / 60 )) min + drain"
