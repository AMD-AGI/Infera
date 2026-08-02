#!/usr/bin/env bash
# SOLO latency-floor run: Case A request shape at concurrency EXACTLY 1.
# Same harness as caseA_run.sh (jump host -> router over the data plane); only
# the workload YAML differs. See workloads/solo.yaml for why each knob is set.
#
# Requires the SOLO_M1 driver patch (scripts/apply_solo_metrics.py) -- without
# it metrics.jsonl carries no new_e2es / new_tpots and the two headline ladders
# of this experiment cannot be produced.
#
#   TAG=probe SUSTAIN=300 bash solo_run.sh    # short shakeout
#   TAG=full                bash solo_run.sh   # 400 + 1800 as shipped
set -u
W=/mnt/vast/c_huggingface/bench_20260801
TAG="${TAG:-probe}"
RAMP="${RAMP:-400}"
SUSTAIN="${SUSTAIN:-1800}"
ROUTER="${ROUTER:-http://10.2.122.10:8100}"
MODEL="${MODEL:-glm5.2-mxfp4}"
TOKENIZER="${TOKENIZER:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
YAML="${YAML:-$W/solo.yaml}"
OUT="$W/results/solo_$TAG"
LOG="$W/logs/solo_$TAG.log"

mkdir -p "$OUT" "$W/logs"

echo "===== SOLO [$TAG] ramp=${RAMP}s sustain=${SUSTAIN}s -> $OUT ====="
echo "  router: $ROUTER"
echo "  yaml:   $YAML"

# Hard gate: the run is pointless if the driver cannot persist E2E/TPOT.
if ! grep -q SOLO_M1 "$W/agbench/agent/agent_throughput.py"; then
  echo "DRIVER NOT PATCHED (no SOLO_M1) -- run scripts/apply_solo_metrics.py first" >&2
  exit 1
fi
curl -sf -m10 "$ROUTER/health" || { echo "ROUTER NOT HEALTHY -- abort" >&2; exit 1; }
echo

cd "$W/agbench" || exit 1
# No --initial-sessions / --max-sessions / --max-inflight on the CLI: they must
# fall through to the YAML. merge_workload_yaml() decides "did the operator set
# this?" by comparing against the argparse DEFAULT, so passing a value that
# happens to equal the YAML's would still register as a CLI override and print
# a misleading "Skipped" line.
nohup "$W/venv/bin/python" -u -m agent.agent_throughput \
  --workload-config "$YAML" \
  --server "$ROUTER" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --ramp-duration "$RAMP" \
  --sustain-duration "$SUSTAIN" \
  --dashboard-mode \
  --name "solo_$TAG" \
  --data-dir "$OUT" \
  > "$LOG" 2>&1 &

echo "  pid $! -> $LOG"
echo "  ETA $(( (RAMP + SUSTAIN) / 60 )) min + drain"
