#!/usr/bin/env bash
# Send one AIPerf Mooncake replay at the deployment, optionally cutting a
# torch-profiler window out of the middle of it.
#
# Runs on the LOGIN NODE; everything that touches the cluster goes through
# assets/lib/remote.sh's `on`.
#
# The caller chooses the round:
#   run_profiling_mode_off   E2E_CAPTURE=0   one handoff
#   run_profiling_mode_on    E2E_CAPTURE=1   two handoffs (bench + trace)
#
# **Why the profiler-attached round is one task and not two.** The profiler window has to
# fall inside the load window. As two sibling tasks agent_sys would schedule them
# concurrently with no synchronisation between them, so the only way to line them
# up would be a rendezvous file — an edge the graph cannot see and cannot report
# on. One task that starts the load, waits for the engine to actually report a
# batch, and then cuts the window is the honest shape for a coupling this tight.
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
ROUND="${E2E_LOAD_ROUND:?}"
CAPTURE="${E2E_CAPTURE:?}"
OUT_AIPERF="${E2E_OUTPUT_AIPERF:?}"

. "$PKG/assets/lib/remote.sh"

LOAD="$PKG/assets/load"
WORK="${E2E_WORK_ROOT:?}"
AIPERF_OUT="$WORK/aiperf"
TRACE_OUT="$WORK/profiles"
CTR="${E2E_CONTAINER:?}"
R="http://${E2E_NODE_IP:?}:${E2E_PORT_ROUTER:?}"
TAG="${ROUND}_$(date +%Y%m%d_%H%M%S)"

WORKDIR="$(pwd)/load.$ROUND"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

say() { printf '[%s] %s\n' "$ROUND" "$*"; }

say "endpoint=$R trace=$(basename "$E2E_AIPERF_TRACE") window=0..${E2E_TRACE_END_MS}ms capture=$CAPTURE"

# ---- 1. preconditions -------------------------------------------------------
require_visible_on_node "$LOAD/aiperf_replay.sh" "staged task package" || exit 1
if ! on "curl -sf -m10 '$R/health'" >/dev/null 2>&1; then
  say "ABORT: no answer from the router at $R -- is the deployment up?"
  exit 1
fi
say "router healthy"

# ---- 2. the load, in the background -----------------------------------------
# Backgrounded here rather than on the node: `on` is an srun step, and letting it
# run in this shell's background keeps its lifetime tied to this body. A process
# nohup'd on the node would outlive a failed attempt and go on sending requests
# at whatever the next round brings up.
say "starting AIPerf (cold prompt synthesis can take minutes before the first request)"
on "NODE_IP='$E2E_NODE_IP' ROUTER_PORT='$E2E_PORT_ROUTER' SERVED='$E2E_SERVED_NAME' \
    MODEL='$E2E_MODEL_PATH' MODEL_MOUNT='$(dirname "$E2E_MODEL_PATH")' \
    AIPERF_IMAGE='$E2E_AIPERF_IMAGE' AIPERF_OUT='$AIPERF_OUT' \
    AIPERF_TRACE='$E2E_AIPERF_TRACE' SCRIPTS='$LOAD' \
    TRACE_END_MS='$E2E_TRACE_END_MS' MAX_CONC='$E2E_MAX_CONC' \
    WORKERS='$E2E_WORKERS' BLOCK_SIZE='$E2E_BLOCK_SIZE' \
    REQ_TIMEOUT='$E2E_REQ_TIMEOUT' TAG='$TAG' \
    bash '$LOAD/aiperf_replay.sh'" > "$WORKDIR/aiperf.log" 2>&1 &
LOAD_PID=$!

# ---- 3. the profiler windows, inside the load --------------------------------
# Two windows, not one. The measurement window runs without Python stacks
# because stacks cost 13x the bytes for a trace nobody would keep at that size
# (measured; see capture.sh section 5/6). The stack window is seconds long and
# exists only so the ranking step can name the Python frame that launched each
# ranked kernel.
STACK_OK=0
if [ "$CAPTURE" = "1" ]; then
  say "cutting the measurement window (warmup ${E2E_WARMUP_S}s, window ${E2E_WINDOW_S}s, with_stack=0)"
  on "NODE_IP='$E2E_NODE_IP' ROUTER_PORT='$E2E_PORT_ROUTER' CTR='$CTR' \
      TRACE_OUT='$TRACE_OUT' WARMUP_S='$E2E_WARMUP_S' WINDOW_S='$E2E_WINDOW_S' \
      WITH_STACK=0 OUT_SUBDIR=mixed \
      TAG='$TAG' bash '$LOAD/capture.sh'" 2>&1 | tee "$WORKDIR/capture.log"
  cap_rc="${PIPESTATUS[0]}"
  if [ "$cap_rc" != "0" ] || ! grep -q CAPTURE_OK "$WORKDIR/capture.log"; then
    say "capture failed (rc=$cap_rc); waiting for the load to finish before reporting"
    wait "$LOAD_PID"
    tail -30 "$WORKDIR/capture.log" >&2
    exit 1
  fi

  # **Not fatal when it fails.** The ranking is complete without it and the
  # launcher block is an enrichment, so losing the stack window must not cost
  # the round its aiperf report and its traces. Whether a round missing it is
  # acceptable is `check_trace_coverage`'s call, which is where "did we get what
  # we needed" belongs -- and it says so with the count from the manifest rather
  # than with this script's opinion.
  #
  # WARMUP_S=0: the engine has been serving through the whole first window.
  if [ "${E2E_STACK_WINDOW_S:-0}" -gt 0 ]; then
    say "cutting the stack window (window ${E2E_STACK_WINDOW_S}s, with_stack=1)"
    on "NODE_IP='$E2E_NODE_IP' ROUTER_PORT='$E2E_PORT_ROUTER' CTR='$CTR' \
        TRACE_OUT='$TRACE_OUT' WARMUP_S=0 WINDOW_S='$E2E_STACK_WINDOW_S' \
        WITH_STACK=1 OUT_SUBDIR=mixed_stacks \
        TAG='$TAG' bash '$LOAD/capture.sh'" 2>&1 | tee "$WORKDIR/capture_stacks.log"
    if [ "${PIPESTATUS[0]}" = "0" ] && grep -q CAPTURE_OK "$WORKDIR/capture_stacks.log"; then
      STACK_OK=1
      say "stack window captured"
    else
      say "WARN: the stack window failed; the ranking will carry no launcher frames"
      tail -20 "$WORKDIR/capture_stacks.log" >&2
    fi
  else
    say "stack window disabled (E2E_STACK_WINDOW_S=${E2E_STACK_WINDOW_S:-0})"
  fi
fi

# ---- 4. wait for the load ----------------------------------------------------
say "waiting for AIPerf to finish"
wait "$LOAD_PID"; load_rc=$?
if [ "$load_rc" != "0" ] || ! grep -q AIPERF_OK "$WORKDIR/aiperf.log"; then
  say "AIPerf failed (rc=$load_rc). Last 30 lines:"
  tail -30 "$WORKDIR/aiperf.log" >&2
  exit 1
fi
RUN_DIR="$AIPERF_OUT/$TAG"
say "load done -> $RUN_DIR"

# ---- 5. the aiperf handoff ---------------------------------------------------
say "assembling $OUT_AIPERF"
A_ITEMS="$OUT_AIPERF/items"
rm -rf "$A_ITEMS"; mkdir -p "$A_ITEMS/result" "$A_ITEMS/env" "$A_ITEMS/logs"

# The node writes straight into the handoff: $HOME is NFS from the same server
# the compute nodes mount, which assets/lib/remote.sh asserts up front.
on "cp '$RUN_DIR'/profile_export_aiperf.csv '$RUN_DIR'/profile_export_aiperf.json \
       '$RUN_DIR'/profile_export_console.txt '$RUN_DIR'/server_metrics_export.csv \
       '$A_ITEMS/result/' 2>/dev/null; \
    gzip -9 -c '$RUN_DIR'/profile_export.jsonl > '$A_ITEMS/result/profile_export.jsonl.gz'; \
    for f in '$RUN_DIR'/logs/*; do gzip -9 -c \"\$f\" > '$A_ITEMS/logs/'\$(basename \"\$f\").gz; done"

# A summary the validator and the next stage read, so neither has to parse the
# CSV's 63 rows of AIPerf metric names.
python3 "$LOAD/summarise.py" "$A_ITEMS/result/profile_export_aiperf.csv" \
  "$A_ITEMS/result/summary.json" || {
    say "ABORT: could not summarise the AIPerf export"
    exit 1
  }

on "docker inspect $CTR --format '{{.Image}}'" > "$A_ITEMS/env/image.txt" 2>&1
on "docker exec $CTR bash -c \"tr '\\0' '\\n' < /proc/\\\$(pgrep -f 'infera.engine.sglang' | head -1)/cmdline\"" \
  > "$A_ITEMS/env/engine_argv.txt" 2>&1
on "docker exec $CTR cat /run_router.sh" > "$A_ITEMS/env/router_cmd.txt" 2>&1

python3 - "$A_ITEMS/env/load.json" <<PYEOF
import json, sys
json.dump({
    "round": "$ROUND",
    "endpoint": "$R",
    "served_model_name": "$E2E_SERVED_NAME",
    "trace": "$(basename "$E2E_AIPERF_TRACE")",
    "trace_window_ms": [0, $E2E_TRACE_END_MS],
    "concurrency_ceiling": $E2E_MAX_CONC,
    "aiperf_workers": $E2E_WORKERS,
    "isl_block_size": $E2E_BLOCK_SIZE,
    "aiperf_image": "$E2E_AIPERF_IMAGE",
    "profiler_window": {"captured": $CAPTURE, "warmup_s": $E2E_WARMUP_S, "window_s": $E2E_WINDOW_S},
}, open(sys.argv[1], "w"), indent=2)
PYEOF

MODEL_NAME="$(basename "$E2E_MODEL_PATH")"
cat > "$A_ITEMS/command" <<EOF
#!/usr/bin/env bash
# Reproduce this load. Executable because agent.gate requires it of a 'command'
# item, and written with shell variables rather than absolute paths so that the
# locality seal has nothing to reject and nothing had to be substituted away.
set -eu
: "\${MODEL_MOUNT:?export MODEL_MOUNT=<directory holding the checkpoint>}"
: "\${WORK_ROOT:?export WORK_ROOT=<node-local work area>}"
: "\${AIPERF_TRACE:?export AIPERF_TRACE=<the Mooncake trace JSONL>}"
: "\${SCRIPTS:?export SCRIPTS=<the package's assets/load directory>}"

NODE_IP=$E2E_NODE_IP ROUTER_PORT=$E2E_PORT_ROUTER SERVED=$E2E_SERVED_NAME \\
MODEL="\$MODEL_MOUNT/$MODEL_NAME" MODEL_MOUNT="\$MODEL_MOUNT" \\
AIPERF_IMAGE=$E2E_AIPERF_IMAGE AIPERF_OUT="\$WORK_ROOT/aiperf" \\
AIPERF_TRACE="\$AIPERF_TRACE" SCRIPTS="\$SCRIPTS" \\
TRACE_END_MS=$E2E_TRACE_END_MS MAX_CONC=$E2E_MAX_CONC WORKERS=$E2E_WORKERS \\
BLOCK_SIZE=$E2E_BLOCK_SIZE REQ_TIMEOUT=$E2E_REQ_TIMEOUT \\
bash "\$SCRIPTS/aiperf_replay.sh"
EOF
chmod +x "$A_ITEMS/command"

gzip -9 -c "$WORKDIR/aiperf.log" > "$A_ITEMS/logs/replay.log.gz"

cat > "$A_ITEMS/watchout" <<EOF
This trace saturates a single-node MIX deployment. Effective concurrency reaches
the ceiling and stays there, so the latency percentiles here describe a queue and
not the model. They are a load, not an SLA. To measure latency instead, slow the
replay down with a synthesis speedup ratio below 1, or lower the concurrency
ceiling until nothing queues.

Round '$ROUND': this deployment was running with decode CUDA graphs
$( [ "$CAPTURE" = 1 ] && echo "OFF and the torch profiler control plane ON, so the throughput here is NOT comparable to a normal run" || echo "ON and no profiler attached, so the throughput here is the one worth quoting" ).
env/engine_argv.txt and env/router_cmd.txt are the processes' own command lines,
which is what makes that claim checkable rather than asserted.

ignore_eos is sent, so output length follows the trace rather than the model's
own stopping. That makes decode numbers here incomparable to a benchmark that
lets the model stop.
EOF

cat > "$OUT_AIPERF/README.md" <<EOF
# $ROUND.bench_result

## Purpose

One AIPerf replay of a Mooncake production trace against \$E2E_MODEL_NAME, at the
timestamps the trace recorded. Unlike a synthetic sweep, every request's prompt
is expanded from the trace's hash ids, so the radix cache and the kv-aware router
are actually exercised.

## How to run

\`items/command\` is the invocation; export \`MODEL_MOUNT\`, \`WORK_ROOT\`,
\`AIPERF_TRACE\` and \`SCRIPTS\` first. Site paths elsewhere in this handoff are
written as \`@NAME@\`.

## Result

\`items/result/summary.json\` is the short form: request count, throughput, TTFT,
inter-token latency and the sequence lengths. \`profile_export_aiperf.csv\` is
AIPerf's own export with every percentile, \`profile_export.jsonl.gz\` the
per-request records, and \`profile_export_console.txt\` the tables as printed.

## Environment

\`items/env/load.json\` carries the replay's own shape. \`engine_argv.txt\` and
\`router_cmd.txt\` are the command lines of the processes that served it, which is
what identifies the round.

## Watch out

See \`items/watchout\`. In short: this load saturates the deployment, so the
latency percentiles describe a queue; and \`ignore_eos\` makes decode numbers
incomparable to a benchmark that lets the model stop on its own.
EOF

# ---- 6. the trace handoff ----------------------------------------------------
if [ "$CAPTURE" = "1" ]; then
  OUT_TRACE="${E2E_OUTPUT_TRACE:?}"
  TRACE_DIR="$TRACE_OUT/$TAG/mixed"
  say "assembling $OUT_TRACE from $TRACE_DIR"
  T_ITEMS="$OUT_TRACE/items"
  rm -rf "$T_ITEMS"; mkdir -p "$T_ITEMS/result/traces" "$T_ITEMS/env" "$T_ITEMS/logs"

  # ~60 MB per rank, 8 ranks. Copied on the node rather than streamed through
  # this shell: both sides see the same NFS home, so this is one cp and not
  # 462 MB through an srun pipe.
  on "cp '$TRACE_DIR'/*.trace.json.gz '$T_ITEMS/result/traces/'" || {
    say "ABORT: could not copy traces out of $TRACE_DIR"
    exit 1
  }

  python3 "$LOAD/manifest.py" "$T_ITEMS/result/traces" "$T_ITEMS/result/manifest.json" || {
    say "ABORT: could not build the trace manifest"
    exit 1
  }

  # ---- the stack window's traces ---------------------------------------------
  # **Only E2E_STACK_RANKS of them are carried.** Every rank runs the same Python,
  # so one rank's stacks resolve every launcher that is resolvable and the second
  # is corroboration -- which is why `launchers.py` reads two files and why
  # Hyperloom's own resolver caps at two. Carrying all eight would multiply the
  # one genuinely expensive part of this handoff for no new frames.
  if [ "$STACK_OK" = "1" ]; then
    STACK_DIR="$TRACE_OUT/$TAG/mixed_stacks"
    mkdir -p "$T_ITEMS/result/stacks"
    keep="${E2E_STACK_RANKS:-2}"
    say "carrying $keep of the stack window's rank file(s) from $STACK_DIR"
    on "ls '$STACK_DIR'/*.trace.json.gz 2>/dev/null | sort | head -$keep \
        | xargs -r -I{} cp {} '$T_ITEMS/result/stacks/'" || {
      say "WARN: could not copy the stack traces; continuing without them"
      rm -rf "$T_ITEMS/result/stacks"
    }
  fi
  if [ -d "$T_ITEMS/result/stacks" ] && ls "$T_ITEMS/result/stacks"/*.trace.json.gz >/dev/null 2>&1; then
    python3 "$LOAD/manifest.py" "$T_ITEMS/result/stacks" "$T_ITEMS/result/stacks_manifest.json" || {
      say "WARN: could not build the stack manifest; continuing without it"
      rm -f "$T_ITEMS/result/stacks_manifest.json"
    }
    gzip -9 -c "$WORKDIR/capture_stacks.log" > "$T_ITEMS/logs/capture_stacks.log.gz"
  else
    say "no stack traces in this handoff"
  fi

  cp "$A_ITEMS/env/engine_argv.txt" "$A_ITEMS/env/router_cmd.txt" \
     "$A_ITEMS/env/image.txt" "$T_ITEMS/env/"
  cp "$A_ITEMS/env/load.json" "$T_ITEMS/env/"
  gzip -9 -c "$WORKDIR/capture.log" > "$T_ITEMS/logs/capture.log.gz"

  cat > "$T_ITEMS/command" <<EOF
#!/usr/bin/env bash
# Reproduce this capture. Needs a load already in flight -- an idle window
# profiles an empty scheduler loop, not the model.
#
# Two windows, in this order: the measurement window without Python stacks, then
# a short one with them. Running the measurement window with stacks on costs 13x
# the bytes and measures the same thing.
set -eu
: "\${WORK_ROOT:?export WORK_ROOT=<node-local work area>}"
: "\${SCRIPTS:?export SCRIPTS=<the package's assets/load directory>}"

NODE_IP=$E2E_NODE_IP ROUTER_PORT=$E2E_PORT_ROUTER CTR=$CTR \\
TRACE_OUT="\$WORK_ROOT/profiles" WARMUP_S=$E2E_WARMUP_S WINDOW_S=$E2E_WINDOW_S \\
WITH_STACK=0 OUT_SUBDIR=mixed \\
bash "\$SCRIPTS/capture.sh"

NODE_IP=$E2E_NODE_IP ROUTER_PORT=$E2E_PORT_ROUTER CTR=$CTR \\
TRACE_OUT="\$WORK_ROOT/profiles" WARMUP_S=0 WINDOW_S=${E2E_STACK_WINDOW_S:-3} \\
WITH_STACK=1 OUT_SUBDIR=mixed_stacks \\
bash "\$SCRIPTS/capture.sh"
EOF
  chmod +x "$T_ITEMS/command"

  cat > "$T_ITEMS/watchout" <<'EOF'
This is a MIX deployment, so prefill and decode run in one process and this trace
holds both kinds of kernel interleaved. They cannot be separated by role after
the fact; a PD-disaggregated deployment can, because the router's role selector
picks a different worker.

The engine ran with decode CUDA graphs OFF for this capture. That is what makes
individual kernels attributable -- with graphs on the profiler records one launch
instead of the kernels inside it -- and it is also why the throughput measured
during this window is not a control. The graphs-on numbers are in the
profiling_mode_off round's handoff.

This handoff carries two captures of the same load, and they are not
interchangeable.

result/traces/ is the MEASUREMENT window: with_stack false, one file per rank.
Every number quoted about this round comes from here.

result/stacks/ is the STACK window: with_stack true, seconds long, and only the
first E2E_STACK_RANKS rank files. It exists so the ranking step can name the Python
frame that launched each ranked kernel, and it is useless as a measurement --
it is short, it was taken after the measurement window, and it holds only some
of the ranks. Do not aggregate the two.

The split is a cost decision, measured on this hardware rather than assumed:
the same workload profiled with and without stacks came to 2,996,700 against
228,553 bytes, so 13.1x uncompressed and 16.5x gzipped, from 9,565
python_function events against none. Kernel counts and total kernel time were
identical across the pair, so stacks do not change what is measured -- but at
60.5 MB per rank for a 15 s window, stacks on would be about 1 GB per rank.

record_shapes is true in both windows, and Magpie's Input Shapes column exists
only because of it.
EOF

  cat > "$OUT_TRACE/README.md" <<EOF
# $ROUND.profile_result

## Purpose

One torch-profiler window cut out of a running AIPerf replay: ${E2E_WINDOW_S}s,
opened after ${E2E_WARMUP_S}s of warm-up and only once the engine had reported an
actual batch. One trace file per tensor-parallel rank.

## How to run

\`items/command\` reproduces the capture against a deployment brought up with the
profiling control plane enabled, while a load is in flight.

## Result

\`items/result/traces/\` holds one \`*.trace.json.gz\` per rank, from the
measurement window. \`items/result/manifest.json\` records each rank's size,
SHA-256, GPU kernel event count, \`python_function\` count and time span, which is
what makes "the capture covered every rank" a checkable claim rather than a file
listing.

\`items/result/stacks/\` holds the stack window: the same load profiled with
\`with_stack\` on for ${E2E_STACK_WINDOW_S:-0}s, kept for the first
${E2E_STACK_RANKS:-2} rank(s) only, with its own
\`items/result/stacks_manifest.json\`. It is the input to the ranking step's
launcher resolution and it is **not** a measurement. See \`items/watchout\`.

## Environment

\`items/env/engine_argv.txt\` is the engine's own command line; the decode graph
backend in it is what identifies this as the profiler-attached round.

## Watch out

See \`items/watchout\`: prefill and decode kernels share this trace, and the
throughput during the window is not a control.
EOF
fi

# ---- 7. make it publishable --------------------------------------------------
say "redacting site-specific paths"
REDACT_ARGS=(
  "MODEL_MOUNT=$(dirname "$E2E_MODEL_PATH")"
  "WORK_ROOT=$WORK"
  "TASK_PACKAGE=$PKG"
  "TRACE_DIR=$(dirname "$E2E_AIPERF_TRACE")"
  "TMPDIR=/tmp"
  "HOME=$HOME"
)
python3 "$PKG/assets/lib/redact.py" "$OUT_AIPERF" "${REDACT_ARGS[@]}" || {
  say "ABORT: the AIPerf evidence still names local paths redact.py could not place"
  exit 1
}
if [ "$CAPTURE" = "1" ]; then
  python3 "$PKG/assets/lib/redact.py" "${E2E_OUTPUT_TRACE:?}" "${REDACT_ARGS[@]}" || {
    say "ABORT: the trace evidence still names local paths redact.py could not place"
    exit 1
  }
fi

say "done"
exit 0
