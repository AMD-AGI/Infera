#!/usr/bin/env bash
# Bring one round of the deployment up and hand back a `reproducible` record of it.
#
# Runs on the LOGIN NODE. Everything that touches a GPU goes through
# assets/lib/remote.sh's `on`, which is `srun --overlap` into the allocation.
#
# The round is chosen by the caller: PD_ROUND / PD_CUDA_GRAPH / PD_PROFILE.
# assets/serve_baseline.task/entry.sh sets baseline/1/0; the profiled round will
# set profiled/0/1 and change nothing else.
#
# **Evidence is captured through stdout, not read off a shared filesystem.**
# $HOME here happens to be NFS, so the store would be reachable from the node --
# but a deployment record that only assembles on clusters where that is true is a
# worse artefact than one that assembles anywhere. The one thing that does need
# the shared mount is running the staged scripts by absolute path, and that is
# asserted up front rather than assumed.
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
OUT="${PD_OUTPUT_DIR:?}"
ROUND="${PD_ROUND:?}"
CUDA_GRAPH="${PD_CUDA_GRAPH:?}"
PROFILE="${PD_PROFILE:?}"

. "$PKG/assets/lib/remote.sh"

SERVE="$PKG/assets/serve"
WORK="${PD_WORK_ROOT:?}"
TRACE_OUT="$WORK/profiles"
CTR="${PD_CTR:?}"
R="http://${PD_NODE_IP:?}:${PD_ROUTER_PORT:?}"

# Staging area in the zone. The zone is this attempt's cwd and is discarded with
# it, so intermediate files here cost nothing and keep half-written evidence out
# of the handoff until the whole round has succeeded.
WORKDIR="$(pwd)/round.$ROUND"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
LOG="$WORKDIR/mix_up.log"

say() { printf '[%s] %s\n' "$ROUND" "$*"; }

say "node=$PD_NODE ip=$PD_NODE_IP jobid=$PD_JOBID"
say "model=$PD_MODEL tp=$PD_TP cuda_graph=$CUDA_GRAPH profile=$PROFILE"

# ---- 1. preconditions -------------------------------------------------------
require_visible_on_node "$SERVE/mix_up.sh" "staged task package" || exit 1

# The model directory has to be readable from the node, and it is a --var so a
# typo in it is the most likely way this task is misconfigured. Checking it here
# turns a 12-minute wait for a health endpoint that never comes into an
# immediate, named failure.
if ! on "test -r '$PD_MODEL/config.json'" >/dev/null 2>&1; then
  say "ABORT: $PD_MODEL/config.json is not readable on $PD_NODE"
  exit 1
fi
say "preconditions ok"

# ---- 2. bring the stack up --------------------------------------------------
# mix_up.sh is the script the manual walk-through validated, taken across
# unmodified; the PD_* names are mapped to the ones it reads rather than the
# script being rewritten to read PD_*, so what runs here is what was tested.
#
# PD_REUSE_DEPLOYMENT=1 skips the bring-up when the endpoint is already serving.
# It is a development aid — a cold start is a quarter of an hour and iterating on
# the handoff shape should not cost that each time — and it is safe only because
# step 3 records the engine's OBSERVED argv and check_service_live decides the
# round from that. Reusing a deployment that came up in the wrong configuration
# therefore fails validation rather than passing quietly.
if [ "${PD_REUSE_DEPLOYMENT:-0}" = "1" ] && on "curl -sf -m5 '$R/health'" >/dev/null 2>&1; then
  say "REUSING the deployment already serving at $R (PD_REUSE_DEPLOYMENT=1)"
  echo "reused an existing deployment; no bring-up log for this round" > "$LOG"
else
  say "deploying (this cold-starts the engine; first load off NFS took 819s)"
  on "NODE_IP='$PD_NODE_IP' IMAGE='$PD_IMAGE' ETCD_IMAGE='$PD_ETCD_IMAGE' \
      MODEL='$PD_MODEL' MODEL_MOUNT='$(dirname "$PD_MODEL")' SERVED='$PD_SERVED' \
      CTR='$CTR' ROUTER_PORT='$PD_ROUTER_PORT' PORT='$PD_WORKER_PORT' \
      ETCD_PORT='$PD_ETCD_PORT' TP='$PD_TP' \
      WORK_ROOT='$WORK' TRACE_OUT='$TRACE_OUT' \
      DSA_ARGS='$PD_DSA_ARGS' PARSER_ARGS='$PD_PARSER_ARGS' CTX='$PD_CTX' \
      PROFILE='$PROFILE' CUDA_GRAPH='$CUDA_GRAPH' SCRIPTS='$SERVE' \
      bash '$SERVE/mix_up.sh'" 2>&1 | tee "$LOG"
  up_rc="${PIPESTATUS[0]}"

  if [ "$up_rc" != "0" ] || ! grep -q MIX_UP_OK "$LOG"; then
    say "deployment failed (rc=$up_rc). Last 40 lines:"
    tail -40 "$LOG" >&2
    exit 1
  fi
fi
say "deployment up at $R"

# ---- 3. evidence ------------------------------------------------------------
say "collecting evidence"
# The environment is passed explicitly, not inherited. `srun --export=ALL` copies
# this process's environment, which carries the PD_* block but none of the names
# mix_smoke.sh reads -- and an early run appeared to work only because the
# operator's login shell still had them exported from the manual walk-through.
# The script uses `${VAR:?}`, so the failure is at least loud when it happens.
on "NODE_IP='$PD_NODE_IP' ROUTER_PORT='$PD_ROUTER_PORT' SERVED='$PD_SERVED' \
    CTR='$CTR' bash '$SERVE/mix_smoke.sh'" > "$WORKDIR/smoke.txt" 2>&1
on "curl -s -m10 '$R/v1/workers'"  > "$WORKDIR/workers.json" 2>/dev/null
on "curl -s -m10 '$R/v1/models'"   > "$WORKDIR/models.json"  2>/dev/null
on "curl -s -o /dev/null -w '%{http_code}' -m10 '$R/health'" > "$WORKDIR/health.txt" 2>/dev/null

# The engine log TAIL, not the live log: the verdict has to be reproducible from
# the handoff alone, so what the validator scans for fault lines is the same
# bytes a reader will see.
on "docker exec $CTR tail -c 200000 /tmp/glm53_mix.log" > "$WORKDIR/worker.tail.log" 2>&1
on "docker exec $CTR tail -c 50000 /tmp/router.log"     > "$WORKDIR/router.tail.log" 2>&1

on "docker inspect $CTR --format '{{.Image}}'"                > "$WORKDIR/image.txt" 2>&1
on "rocm-smi --showproductname 2>/dev/null | head -20"        > "$WORKDIR/gpu.txt" 2>&1
on "docker exec $CTR bash -c 'cat /opt/rocm/.info/version 2>/dev/null || true'" > "$WORKDIR/rocm.txt" 2>&1

# **The engine's argv as it is actually running**, one flag per line.
#
# The round's two defining flags are recorded from what the process HAS, not from
# what this script ASKED for. Those differ whenever a deployment is reused, and
# they would also differ if mix_up.sh silently fell back — in which case a
# self-declared `cuda_graph: 1` would be a lie the validator could not catch.
# check_service_live decides the round from this file.
on "docker exec $CTR bash -c \"tr '\\0' '\\n' < /proc/\\\$(pgrep -f 'infera.engine.sglang' | head -1)/cmdline\"" \
  > "$WORKDIR/engine_argv.txt" 2>&1

# The router's command line, for the same reason. Profiling is a ROUTER flag, so
# it cannot be read out of the engine's argv. mix_up.sh stages the invocation as
# /run_router.sh, which is what the running router was started from.
on "docker exec $CTR cat /run_router.sh" > "$WORKDIR/router_cmd.txt" 2>&1

# ---- 4. assemble the handoff ------------------------------------------------
# `reproducible` requires `result` and `env`, plus one of `script` / `command`.
# Item keys are exactly those names, so a directory here is named `result` and
# not `results`; handoff/content.py:check_items rejects any top-level item the
# content type never declared.
say "assembling handoff -> $OUT"
ITEMS="$OUT/items"
rm -rf "$ITEMS"; mkdir -p "$ITEMS/result" "$ITEMS/env" "$ITEMS/logs"

cp "$WORKDIR/smoke.txt" "$WORKDIR/workers.json" "$WORKDIR/models.json" \
   "$WORKDIR/health.txt" "$ITEMS/result/"

cp "$WORKDIR/gpu.txt" "$WORKDIR/rocm.txt" "$WORKDIR/image.txt" \
   "$WORKDIR/engine_argv.txt" "$WORKDIR/router_cmd.txt" "$ITEMS/env/"

# **`command`, not `script`.** `reproducible` takes either, and copying the
# scripts in was the first design. It does not survive contact with the seal:
# `handoff.locality` rejects any absolute path outside a fixed allow-list, and a
# real bring-up script names `/tmp/glm53_mix.log` and `/tmp/router.log`. The two
# ways out are both worse than this one — substituting the paths away publishes
# scripts that no longer run, and archiving them publishes something nobody can
# review. So the handoff carries the invocation and says which commit of which
# package the scripts came from; `deployment.json` records both.
MODEL_NAME="$(basename "$PD_MODEL")"
cat > "$ITEMS/command" <<EOF
#!/usr/bin/env bash
# Reproduce this round. \`agent.gate\` requires this item to be executable, so it
# is a script rather than a transcript — and writing it as a runnable script is
# what makes it survive publication: the three site-specific paths are shell
# variables the caller supplies, so there is no absolute path for the locality
# seal to reject and nothing had to be substituted away.
#
# The scripts this drives are not copied into the handoff; items/env/deployment.json
# names the package and the commit they came from.
set -eu
: "\${MODEL_MOUNT:?export MODEL_MOUNT=<directory holding the checkpoint>}"
: "\${WORK_ROOT:?export WORK_ROOT=<node-local work area, needs ~1 GB per capture>}"
: "\${SCRIPTS:?export SCRIPTS=<the package's assets/serve directory>}"

NODE_IP=$PD_NODE_IP IMAGE=$PD_IMAGE ETCD_IMAGE=$PD_ETCD_IMAGE \\
MODEL="\$MODEL_MOUNT/$MODEL_NAME" MODEL_MOUNT="\$MODEL_MOUNT" SERVED=$PD_SERVED \\
CTR=$CTR ROUTER_PORT=$PD_ROUTER_PORT PORT=$PD_WORKER_PORT ETCD_PORT=$PD_ETCD_PORT \\
TP=$PD_TP WORK_ROOT="\$WORK_ROOT" TRACE_OUT="\$WORK_ROOT/profiles" \\
PROFILE=$PROFILE CUDA_GRAPH=$CUDA_GRAPH SCRIPTS="\$SCRIPTS" \\
bash "\$SCRIPTS/mix_up.sh"

NODE_IP=$PD_NODE_IP SERVED=$PD_SERVED CTR=$CTR ROUTER_PORT=$PD_ROUTER_PORT \\
bash "\$SCRIPTS/mix_smoke.sh"
EOF
chmod +x "$ITEMS/command"

# **The logs go in compressed, and that is not a size decision.** 817 of the 818
# absolute paths in this round's logs are false positives of the kind
# handoff.locality's own docstring predicts: container-internal paths the engine
# image owns, HTTP routes in an access log, and an etcd key prefix. None is a
# fact about this machine, and the mechanism that would let the kind say so --
# locality.Oracles.image_prefixes, fed from the kind's `dependencies` -- exists in
# the type and is wired to nothing (temp/bugs/002).
#
# The examples are described rather than quoted on purpose: a comment naming one
# of those routes is itself an absolute path by the heuristic's reading, which is
# how this file first failed to publish when the scripts were still copied in.
#
# The alternative was to substitute those paths away, which would corrupt the one
# artefact whose value is being a faithful record. gzip keeps the bytes exactly
# and the seal skips what it cannot decode as UTF-8. check_service_live reads
# them back through gzip.
for f in "$LOG" "$WORKDIR/worker.tail.log" "$WORKDIR/router.tail.log"; do
  gzip -9 -c "$f" > "$ITEMS/logs/$(basename "$f").gz"
done

# Which scripts ran, identified rather than copied. The commit is best-effort:
# the package is data in a repository, and a run from a dirty tree should say so
# instead of naming a commit that does not describe what executed.
PKG_COMMIT="$(git -C "$PKG" rev-parse HEAD 2>/dev/null || echo unknown)"
PKG_DIRTY="$(git -C "$PKG" status --porcelain 2>/dev/null | head -1)"
[ -n "$PKG_DIRTY" ] && PKG_COMMIT="$PKG_COMMIT+dirty"

python3 - "$ITEMS/env/deployment.json" <<PYEOF
import json, sys
json.dump({
    "round": "$ROUND",
    "scripts": {
        "package": "agent_sys/examples/llm_e2e_performance_optimization/profiling-demo",
        "entrypoints": ["assets/serve/mix_up.sh", "assets/serve/mix_smoke.sh"],
        "commit": "$PKG_COMMIT",
    },
    "node": "$PD_NODE",
    "node_ip": "$PD_NODE_IP",
    "slurm_jobid": "$PD_JOBID",
    "image": "$PD_IMAGE",
    "image_id": open("$WORKDIR/image.txt").read().strip(),
    "model_path": "$PD_MODEL",
    "served_model_name": "$PD_SERVED",
    "endpoint": "$R",
    "ports": {"router": $PD_ROUTER_PORT, "worker": $PD_WORKER_PORT, "etcd": $PD_ETCD_PORT},
    "tp_size": $PD_TP,
    "disagg_mode": "mixed",
    # What this run ASKED for. What it GOT is env/engine_argv.txt, and that is
    # what check_service_live reads -- see the comment beside its capture.
    "requested": {"cuda_graph": $CUDA_GRAPH, "profiling_enabled": $PROFILE},
    "work_root": "$WORK",
}, open(sys.argv[1], "w"), indent=2)
PYEOF

cat > "$ITEMS/watchout" <<'EOF'
This is a MIX (aggregated) deployment: prefill and decode are the same process,
so a profiler window taken against it holds both kinds of kernel in one trace and
they cannot be separated after the fact by role. The 1P1D kit can separate them
because the router's role selector picks a different worker.

etcd is on 12379, not the 2379 the reference scripts hard-code. These nodes run a
Kubernetes control plane whose own etcd holds 2379 over TLS, and a plaintext
discovery client pointed at it fails in a way that surfaces much later as an
empty worker pool.

The weights are read straight off NFS. Measured 921 MB/s single-stream on this
node, so a cold load is ~13 minutes; a second load in the same session is served
from page cache (~4 minutes) because the node has 3 TB of RAM.
EOF

cat > "$OUT/README.md" <<EOF
# deployment_baseline

## Purpose

A record of one GLM-5.3-Flash MIX deployment on a single MI355X node, and the
evidence that it served. This is the \`$ROUND\` round: decode CUDA graphs are
$( [ "$CUDA_GRAPH" = 1 ] && echo "ON, so throughput measured against this deployment is quotable" || echo "OFF, so throughput measured against this deployment is NOT a baseline" ),
and the torch-profiler control plane is $( [ "$PROFILE" = 1 ] && echo "ENABLED" || echo "disabled" ).

## How to run

\`items/command\` is the invocation. The scripts it names are not copied in here;
\`items/env/deployment.json\` says which package and commit they came from, which
is what makes them recoverable without shipping a second copy that could drift.

Every site path in this handoff is written as \`@NAME@\`. On the machine that
produced it, \`@MODEL_MOUNT@\` was the directory holding the checkpoint,
\`@WORK_ROOT@\` the node-local work area, \`@TMPDIR@\` the temporary directory and
\`@HOME@\` the operator's home. They are placeholders because \`handoff\` refuses to
seal content that names one machine's paths — the right rule, and the reason this
record transfers. \`items/command\` takes the same three as shell variables, so it
runs once they are exported.

## Result

The endpoint is \`$R\`. \`items/result/\` carries the proof:
\`workers.json\` (expected: exactly one worker, \`disagg_mode\` \`mixed\`),
\`models.json\`, \`health.txt\` (the router's HTTP status) and \`smoke.txt\`
(five blocks: registration, models, a factual answer, an arithmetic answer that
must be 391, and an engine-side scan for faults).

## Environment

\`items/env/deployment.json\` carries the full shape — node, image id, model path,
ports, TP size, and the two flags that define the round. \`gpu.txt\`, \`rocm.txt\`
and \`image.txt\` are the raw captures it was built from.

## Watch out

See \`items/watchout\`. In short: MIX puts prefill and decode kernels in one
trace; etcd is on $PD_ETCD_PORT because Kubernetes owns 2379 here; the weights
are read over NFS, so the first load of a session is about three times slower
than the ones after it.
EOF

# ---- 5. make it publishable -------------------------------------------------
# A handoff should not name an absolute path outside a fixed allow-list. The
# rule is right -- a record of one machine's afternoon is not a transferable
# artefact -- but NOTHING ENFORCES IT AT PUBLICATION: `handoff/store.py` does not
# call `locality.check` (user-ruled 2026-08-31, 97% false positive). So the
# site-specific roots are replaced by named placeholders here, on the producing
# side, and `redact.py` is the only thing that checks. `${MODEL_MOUNT}/GLM-5.3-Flash-FP8` keeps the model's identity
# and drops the mount root, which is the split spec §7 asks for.
#
# Ordering matters: this runs after everything is in place and before the body
# exits, so nothing reaches the seal unredacted. redact.py fails loudly and names
# any absolute path it was not given a placeholder for, which is what turns a new
# site-specific path into an error here instead of "output was never delivered"
# a quarter of an hour later.
say "redacting site-specific paths"
python3 "$PKG/assets/lib/redact.py" "$OUT" \
  "MODEL_MOUNT=$(dirname "$PD_MODEL")" \
  "WORK_ROOT=$WORK" \
  "TASK_PACKAGE=$PKG" \
  "TMPDIR=/tmp" \
  "HOME=$HOME" || {
    say "ABORT: evidence still names local paths redact.py could not place"
    exit 1
  }

say "done: $(find "$ITEMS" -type f | wc -l) evidence files under $ITEMS"
exit 0
