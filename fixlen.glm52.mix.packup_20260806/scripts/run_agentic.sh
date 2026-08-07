#!/usr/bin/env bash
# what: run ONE Optimus-AgenticBench workload against the mix router.
# why : one entry point for both the conc-1 repeats (task 2) and the loaded run (task 3);
#       the only difference between them is which YAML is passed.
# how : WORKLOAD=<abs path> TAG=<name> bash run_agentic.sh      (runs ON the node)
#
# NO load knobs are passed on the CLI. The YAML is the single source of truth for offered
# load — passing --initial-sessions here would silently shadow it and make the run
# unreproducible from the file alone.
#
# The driver is the STAGED copy at $AGB, branch fix/realistic-profile-session-driver @1cf01cb
# WITH the SOLO_M1 patch. That patch is load-bearing for task 2: upstream records neither
# per-request E2E nor an index-aligned TPOT array, so a conc-1 latency measurement would have
# to back-solve E2E from TTFT and TPOT. Verified by diff against the local checkout.
set -uo pipefail
W="${W:-/mnt/vast/c_huggingface/glm52_mix_20260806}"
AGB="${AGB:-/mnt/vast/c_huggingface/bench_20260801/agbench}"
VENV="${VENV:-/mnt/vast/c_huggingface/bench_20260801/venv/bin/python}"

WORKLOAD="${WORKLOAD:?WORKLOAD=<abs path to yaml>}"
TAG="${TAG:?TAG=<run name>}"
ROUTER="${ROUTER:-http://10.2.122.78:8100}"
MODEL="${MODEL:-glm5.2-mxfp4}"
TOKENIZER="${TOKENIZER:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"

OUT="$W/results/agentic_$TAG"
LOG="$W/logs/agentic_$TAG.log"
mkdir -p "$OUT" "$W/logs"

echo "===== agentic [$TAG] -> $OUT ====="
echo "  router:   $ROUTER"
echo "  workload: $WORKLOAD"
[ -f "$WORKLOAD" ] || { echo "workload not found" >&2; exit 1; }
curl -sf -m10 "$ROUTER/health" >/dev/null || { echo "ROUTER NOT HEALTHY -- abort" >&2; exit 1; }
md5sum "$WORKLOAD"

cd "$AGB" || exit 1
# --dashboard-mode is MANDATORY: without it nothing structured is persisted and the run is
# unrecoverable once the terminal scrolls.
# ramp/sustain are deliberately NOT passed — they come from the YAML, like every other knob.
nohup "$VENV" -u -m agent.agent_throughput \
  --workload-config "$WORKLOAD" \
  --server "$ROUTER" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --mode realistic \
  --dashboard-mode \
  --name "$TAG" \
  --data-dir "$OUT" \
  > "$LOG" 2>&1 &

echo "  pid $! -> $LOG"
