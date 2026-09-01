#!/usr/bin/env bash
# Measure one arm: correctness first, then performance, in a fixed order.
#
# Runs on the LOGIN NODE; every step reaches the deployment through
# assets/lib/remote.sh's `on`.
#
# **The order is part of the measurement.** "Round 1 is cold for this trace" is
# only true if the same things happened before it in both arms, so the sequence
# is fixed here, recorded in env/measure.json with timestamps, and
# check_bench_report compares the two arms' recorded sequences:
#
#   1 smoke     four checks, seconds
#   2 needle    two lengths x three depths, under a minute
#   3 probe     gates the eval; catches the failures that produce a number
#   4 lm-eval   gsm8k and mixed_prefix_gsm8k
#   5 bench     the trace replay, N rounds: the first cold for this trace, the
#               rest warm
#
# **One task, two handoffs.** Correctness and performance have to run against the
# same deployment instance and must not overlap -- a saturating replay during an
# eval invalidates both. As sibling tasks agent_sys would schedule them
# concurrently with nothing to synchronise them.
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
ARM="${IT_ARM:?}"
OUT_ACCEPT="${IT_OUTPUT_ACCEPT:?}"
OUT_BENCH="${IT_OUTPUT_BENCH:?}"

. "$PKG/assets/lib/remote.sh"

ACCEPT="$PKG/assets/accept"
BENCH="$PKG/assets/bench"
WORK="${IT_WORK_ROOT:?}"
CTR="${IT_CTR:?}"
R="http://${IT_NODE_IP:?}:${IT_ROUTER_PORT:?}"
ROUNDS="${IT_BENCH_ROUNDS:-2}"

WORKDIR="$(pwd)/measure.$ARM"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR/accept" "$WORKDIR/bench" "$WORKDIR/logs"
STEPS="$WORKDIR/steps.tsv"
: > "$STEPS"

say() { printf '[%s] %s\n' "$ARM" "$*"; }
step() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$STEPS"; }

require_visible_on_node "$ACCEPT/needle.py" "staged task package" || exit 1
require_visible_on_node "$WORKDIR" "the attempt zone" || exit 1

on "curl -sf -m10 '$R/health'" >/dev/null 2>&1 || {
  say "ABORT: nothing is serving at $R"
  exit 1
}

run_step() {
  local name="$1"; shift
  local t0 t1 rc
  t0=$(date +%s)
  say "step $name"
  on "$*" > "$WORKDIR/logs/$name.log" 2>&1
  rc=$?
  t1=$(date +%s)
  step "$name" "$rc" "$((t1 - t0))" "$(date -Is -d "@$t0")"
  say "  $name rc=$rc in $((t1 - t0))s"
  tail -3 "$WORKDIR/logs/$name.log" | sed 's/^/    /'
  return 0
}

# ---- 1. smoke ---------------------------------------------------------------
run_step smoke "python3 '$ACCEPT/smoke.py' \
  --url '$R' --model '$IT_SERVED' --container '$CTR' \
  --out '$WORKDIR/accept/smoke.json'"

# ---- 2. needle --------------------------------------------------------------
run_step needle "python3 '$ACCEPT/needle.py' \
  --url '$R' --model '$IT_SERVED' --out '$WORKDIR/accept/needle.json' \
  --gated-tokens '${IT_NEEDLE_TOKENS:-76000}' \
  --frontier-tokens '${IT_NEEDLE_FRONTIER_TOKENS:-127000}' \
  --depths '${IT_NEEDLE_DEPTHS:-0.02,0.5,0.98}'"

# ---- 3. probe ---------------------------------------------------------------
# **It gates the eval's INTERPRETATION, not its execution**, and that is a
# deliberate departure from the 1P1D kit this is ported from.
#
# There, `probe` refuses to spend thirteen minutes on a score nobody can read,
# and the operator watching it reruns. Here the eval is one step inside a
# two-arm run that costs the better part of an hour, and skipping it strands
# every measurement after it. Worse, skipping produces an EMPTY result that
# `check_acceptance` then fails on, so a single probe failure destroys both arms'
# worth of work for a reason unrelated to the patch under test.
#
# So the eval runs regardless, `probe.ok` travels in the handoff, and `compare`
# marks an eval comparison `uninterpretable` — never `same` — when either arm's
# probe failed. Nothing is silently believed and nothing is thrown away.
run_step probe "python3 '$ACCEPT/probe.py' \
  --url '$R' --model '$IT_SERVED' --out '$WORKDIR/accept/probe.json' \
  --max-tokens '${IT_EVAL_MAX_TOKENS:-2048}'"
probe_rc="$(awk -F'\t' '$1=="probe"{print $2}' "$STEPS")"
[ "${probe_rc:-1}" = "0" ] || say "probe FAILED — the eval still runs, and its comparison will be marked uninterpretable"

# ---- 4. llm-eval ------------------------------------------------------------
run_step lm_eval "URL='$R' SERVED='$IT_SERVED' CTR='$CTR' \
  OUT='$WORKDIR/accept/lm_eval' GSM8K_SRC='${IT_GSM8K_DATA:?}' \
  EVALS='${IT_EVAL_NAMES:-gsm8k}' NUM_EXAMPLES='${IT_EVAL_EXAMPLES:-200}' \
  THREADS='${IT_EVAL_THREADS:-32}' MAX_TOKENS='${IT_EVAL_MAX_TOKENS:-2048}' \
  bash '$ACCEPT/lm_eval.sh'"

# ---- 5. bench ---------------------------------------------------------------
# AIPerf artifacts stay on node-local disk while they are written -- the dataset
# mmap cache lives beside them and is reused between rounds -- and are copied
# into the zone once each round is complete.
AIPERF_OUT="$WORK/aiperf/$ARM"
for r in $(seq 1 "$ROUNDS"); do
  TAG="r$r"
  run_step "bench_$TAG" "NODE_IP='$IT_NODE_IP' ROUTER_PORT='$IT_ROUTER_PORT' \
    SERVED='$IT_SERVED' MODEL='$IT_MODEL' MODEL_MOUNT='$(dirname "$IT_MODEL")' \
    AIPERF_IMAGE='${IT_AIPERF_IMAGE:?}' AIPERF_OUT='$AIPERF_OUT' \
    AIPERF_TRACE='${IT_AIPERF_TRACE:?}' SCRIPTS='$BENCH' \
    TRACE_END_MS='${IT_TRACE_END_MS:-120000}' MAX_CONC='${IT_MAX_CONC:-256}' \
    WORKERS='${IT_WORKERS:-16}' BLOCK_SIZE='${IT_BLOCK_SIZE:-512}' \
    REQ_TIMEOUT='${IT_REQ_TIMEOUT:-900}' TAG='$TAG' \
    bash '$BENCH/aiperf_replay.sh'"

  on "cp -r '$AIPERF_OUT/$TAG' '$WORKDIR/bench/$TAG'" >/dev/null 2>&1 || {
    say "  WARN: round $TAG produced no artifact directory to collect"
    mkdir -p "$WORKDIR/bench/$TAG"
  }
  csv="$WORKDIR/bench/$TAG/profile_export_aiperf.csv"
  if [ -r "$csv" ]; then
    python3 "$BENCH/summarise.py" "$csv" "$WORKDIR/bench/$TAG/summary.json" \
      | sed 's/^/    /'
  else
    say "  WARN: round $TAG has no profile_export_aiperf.csv"
  fi
done

# ---- 6. the two handoffs ----------------------------------------------------
say "assembling handoffs"

python3 - "$WORKDIR" "$ARM" <<'PYEOF'
import json, pathlib, sys
workdir, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
steps = []
for line in (workdir / "steps.tsv").read_text().splitlines():
    if not line.strip():
        continue
    name, rc, secs, at = (line.split("\t") + ["", "", ""])[:4]
    steps.append({"step": name, "rc": rc, "seconds": int(secs or 0), "started": at})
(workdir / "steps.json").write_text(json.dumps({"arm": arm, "steps": steps}, indent=2))
print(f"  sequence: {' -> '.join(s['step'] for s in steps)}")
PYEOF

emit_common_env() {
  local dest="$1"
  mkdir -p "$dest"
  cp "$WORKDIR/steps.json" "$dest/"
  python3 - "$dest/context.json" <<PYEOF
import json, sys
json.dump({
    "arm": "$ARM",
    "node": "$IT_NODE",
    "slurm_jobid": "$IT_JOBID",
    "endpoint": "$R",
    "container": "$CTR",
    "image": "$IT_IMAGE",
    "served_model_name": "$IT_SERVED",
    "model_path": "$IT_MODEL",
    "eval": {
        "names": "${IT_EVAL_NAMES:-gsm8k}".split(),
        "num_examples": int("${IT_EVAL_EXAMPLES:-200}"),
        "threads": int("${IT_EVAL_THREADS:-32}"),
        "max_tokens": int("${IT_EVAL_MAX_TOKENS:-2048}"),
        "thinking_mode": "glm-45",
    },
    "needle": {
        "gated_tokens": int("${IT_NEEDLE_TOKENS:-76000}"),
        "frontier_tokens": int("${IT_NEEDLE_FRONTIER_TOKENS:-127000}"),
        "depths": [float(x) for x in "${IT_NEEDLE_DEPTHS:-0.02,0.5,0.98}".split(",")],
    },
    "bench": {
        "rounds": int("$ROUNDS"),
        "trace_end_ms": int("${IT_TRACE_END_MS:-120000}"),
        "max_concurrency": int("${IT_MAX_CONC:-256}"),
        "workers": int("${IT_WORKERS:-16}"),
    },
}, open(sys.argv[1], "w"), indent=2)
PYEOF
}

# --- acceptance ---------------------------------------------------------------
A="$OUT_ACCEPT/items"
rm -rf "$A"; mkdir -p "$A/result" "$A/logs"
cp -r "$WORKDIR/accept/." "$A/result/"
emit_common_env "$A/env"

cat > "$A/command" <<EOF
#!/usr/bin/env bash
# Re-run this arm's correctness suite. Executable because \`agent.gate\` requires
# it, and a script rather than a transcript because every site path arrives as a
# shell variable — which is also what lets it past the locality seal.
set -eu
: "\${ACCEPT:?export ACCEPT=<the package's assets/accept directory>}"
: "\${URL:?export URL=<router base url>}"
: "\${OUT:?export OUT=<directory to write results into>}"
: "\${GSM8K_SRC:?export GSM8K_SRC=<the GSM8K test split, 1319 rows>}"
SERVED="\${SERVED:-$IT_SERVED}"
CTR="\${CTR:-$CTR}"

mkdir -p "\$OUT"
python3 "\$ACCEPT/smoke.py"  --url "\$URL" --model "\$SERVED" --container "\$CTR" --out "\$OUT/smoke.json"
python3 "\$ACCEPT/needle.py" --url "\$URL" --model "\$SERVED" --out "\$OUT/needle.json" \\
  --gated-tokens ${IT_NEEDLE_TOKENS:-76000} --frontier-tokens ${IT_NEEDLE_FRONTIER_TOKENS:-127000} \\
  --depths '${IT_NEEDLE_DEPTHS:-0.02,0.5,0.98}'
python3 "\$ACCEPT/probe.py"  --url "\$URL" --model "\$SERVED" --out "\$OUT/probe.json"
URL="\$URL" SERVED="\$SERVED" CTR="\$CTR" OUT="\$OUT/lm_eval" GSM8K_SRC="\$GSM8K_SRC" \\
  EVALS='${IT_EVAL_NAMES:-gsm8k}' NUM_EXAMPLES=${IT_EVAL_EXAMPLES:-200} \\
  THREADS=${IT_EVAL_THREADS:-32} bash "\$ACCEPT/lm_eval.sh"
EOF
chmod +x "$A/command"

for f in "$WORKDIR"/logs/{smoke,needle,probe,lm_eval}.log; do
  [ -r "$f" ] && gzip -9 -c "$f" > "$A/logs/$(basename "$f").gz"
done

cat > "$A/watchout" <<EOF
${IT_EVAL_EXAMPLES:-200} questions per eval is a Wilson interval of roughly plus or minus five
points. That is enough to catch a broken deployment and not enough to see a
two-point regression; the full set is about 1300 questions and about six minutes
per eval. Raise eval_examples before reading a small difference as real.

temperature 0 is not byte-reproducible on this stack. Everything here compares
ANSWERS, and so should anything built on it.

The needle is a REGRESSION DETECTOR, not a capability gate, and neither of its
two lengths carries a pass/fail on all three depths. Nine measurements on this
deployment found retrieval to be non-monotonic in prompt length, sensitive to the
needle's wording, and sensitive to the generation budget in the wrong direction —
raising --max-tokens from 256 to 2048 turned two passing depths into failures,
because the engine reasons before it answers and a bigger budget lets it reason
past the answer. The table is in assets/accept/needle.py.

What is asserted here is that the prompt reached the length it claims and that at
least one depth retrieved. What the result is FOR is the comparison: a depth the
stock arm retrieved and the patched arm did not. Do not read any of it as a
statement about the model's long-context ability.
EOF

cat > "$OUT_ACCEPT/README.md" <<EOF
# acceptance_$ARM

## Purpose

Everything this arm can say about whether the deployment is CORRECT, as opposed
to fast. Three directions of text plus a scored eval, because each of them fails
in a way the others cannot see.

## How to run

\`items/command\`, with \`ACCEPT\`, \`URL\`, \`OUT\` and \`GSM8K_SRC\` exported.
Site paths in this record are \`@NAME@\` placeholders.

## Result

\`items/result/smoke.json\` — four checks: one worker in \`mixed\` mode, the served
model name, an arithmetic answer that must be 391, and 512 tokens of continuous
prose with no 8-gram repeated more than four times.

\`items/result/needle.json\` — a distinctive passphrase buried at three depths of
a multi-chunk prompt, at two lengths. The gated length is a pass/fail; the
frontier length is recorded for comparison. Prompt length is read back from
\`usage.prompt_tokens\`, so a run that silently sent a short prompt is visible.

\`items/result/probe.json\` — the gate: reachability, answerability, stability
across repeats and under concurrency, and the one check nothing else here makes,
that a long shared prefix does not change the answer.

\`items/result/lm_eval/\` — \`.index\` gives one row per eval with the number of
questions actually scored, counted out of the html report rather than assumed.
The \`.json\` files hold the scores and the \`.html\` files every prompt and
completion, which is the only place to see WHY a score moved.

## Environment

\`items/env/context.json\` — the arm, the endpoint, and every knob the suite ran
with. \`items/env/steps.json\` — the order the steps ran in and how long each
took. The order matters: it is what makes "the first replay round was cold for
this trace" true of both arms.

## Watch out

See \`items/watchout\`.
EOF

# --- bench --------------------------------------------------------------------
B="$OUT_BENCH/items"
rm -rf "$B"; mkdir -p "$B/result" "$B/logs"
cp -r "$WORKDIR/bench/." "$B/result/" 2>/dev/null || true
emit_common_env "$B/env"

cat > "$B/command" <<EOF
#!/usr/bin/env bash
# Re-run this arm's trace replay.
set -eu
: "\${BENCH:?export BENCH=<the package's assets/bench directory>}"
: "\${MODEL_MOUNT:?export MODEL_MOUNT=<directory holding the checkpoint>}"
: "\${AIPERF_OUT:?export AIPERF_OUT=<node-local output directory>}"
: "\${AIPERF_TRACE:?export AIPERF_TRACE=<the mooncake trace>}"
NODE_IP="\${NODE_IP:?export NODE_IP=<the node's data-plane IP>}"

for r in \$(seq 1 $ROUNDS); do
  NODE_IP="\$NODE_IP" ROUTER_PORT=$IT_ROUTER_PORT SERVED=$IT_SERVED \\
  MODEL="\$MODEL_MOUNT/$(basename "$IT_MODEL")" MODEL_MOUNT="\$MODEL_MOUNT" \\
  AIPERF_IMAGE=$IT_AIPERF_IMAGE AIPERF_OUT="\$AIPERF_OUT" AIPERF_TRACE="\$AIPERF_TRACE" \\
  SCRIPTS="\$BENCH" TRACE_END_MS=${IT_TRACE_END_MS:-120000} MAX_CONC=${IT_MAX_CONC:-256} \\
  WORKERS=${IT_WORKERS:-16} TAG="r\$r" bash "\$BENCH/aiperf_replay.sh"
  python3 "\$BENCH/summarise.py" "\$AIPERF_OUT/r\$r/profile_export_aiperf.csv" \\
    "\$AIPERF_OUT/r\$r/summary.json"
done
EOF
chmod +x "$B/command"

for r in $(seq 1 "$ROUNDS"); do
  f="$WORKDIR/logs/bench_r$r.log"
  [ -r "$f" ] && gzip -9 -c "$f" > "$B/logs/$(basename "$f").gz"
done

cat > "$B/watchout" <<EOF
Round 1 is cold for this trace and round 2 is warm, and they are not comparable
to each other. Compare round 1 against round 1 and round 2 against round 2. This
is not a precaution: profiling-demo replayed the same trace against the same
configuration twice and measured 631 output tok/s cold against 1004 warm, with
mean TTFT 25.9 s against 484 ms, because a Mooncake trace carries hash_ids and
prefix hit rate decides how much prefill there is to do.

--concurrency is a CEILING, not a target: fixed-schedule sends at the timestamps
the trace recorded regardless. Measured on this trace and this deployment,
effective concurrency reaches the ceiling and stays there, so the latency
percentiles describe a queue rather than the model.

There is no repeat within a round, so a single round has no interval. What the
report compares is two arms measured the same way, not one number against a
population.
EOF

cat > "$OUT_BENCH/README.md" <<EOF
# bench_$ARM

## Purpose

What this arm does under a production-shaped load: $ROUNDS replay(s) of the first
$(( ${IT_TRACE_END_MS:-120000} / 1000 )) seconds of a Mooncake trace through AIPerf.

A trace replay and not a synthetic sweep, because a random-prompt sweep builds
every prompt independently and therefore has no shared prefix by construction —
it cannot exercise the radix cache or kv-aware routing at all, which are two of
the things a kernel change is most likely to disturb.

## How to run

\`items/command\`, with \`BENCH\`, \`MODEL_MOUNT\`, \`AIPERF_OUT\`, \`AIPERF_TRACE\`
and \`NODE_IP\` exported. AIPerf runs in its own container: the engine image ships
Python 3.10 and AIPerf needs 3.11 or newer.

## Result

One directory per round under \`items/result/\`, each holding AIPerf's exports
(\`profile_export_aiperf.csv\`, \`profile_export.jsonl\`,
\`profile_export_console.txt\`, \`server_metrics_export.csv\`) and a
\`summary.json\` reducing the 63-row CSV to the metrics anything downstream reads.

## Environment

\`items/env/context.json\` and \`items/env/steps.json\` — the same pair the
acceptance handoff carries, so the two can be lined up.

## Watch out

See \`items/watchout\`.
EOF

# ---- 7. publishable ----------------------------------------------------------
say "redacting site-specific paths"
for dest in "$OUT_ACCEPT" "$OUT_BENCH"; do
  python3 "$PKG/assets/lib/redact.py" "$dest" \
    "MODEL_MOUNT=$(dirname "$IT_MODEL")" \
    "WORK_ROOT=$WORK" \
    "TASK_PACKAGE=$PKG" \
    "GSM8K_DIR=$(dirname "${IT_GSM8K_DATA:-/none}")" \
    "TRACE_DIR=$(dirname "${IT_AIPERF_TRACE:-/none}")" \
    "ZONE=$(pwd)" \
    "TMPDIR=/tmp" \
    "HOME=$HOME" || {
      say "ABORT: $dest still names paths the handoff seal will refuse"
      exit 1
    }
done

say "done: acceptance $(find "$A" -type f | wc -l) files, bench $(find "$B" -type f | wc -l) files"
exit 0
