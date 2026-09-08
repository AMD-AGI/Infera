#!/usr/bin/env bash
# Bring one arm of the experiment up and hand back a `reproducible` record of it.
#
# Runs on the LOGIN NODE. Everything that touches a GPU goes through
# assets/lib/remote.sh's `on`, which is `srun --overlap` into the allocation.
#
# The arm is chosen by the caller through E2E_ARM. `stock` mounts nothing;
# `patched` mounts the plan `apply_patch` built. **That is the only difference**,
# and keeping both arms on one implementation is what stops them drifting into
# two deployments that vary in more than the axis under test.
#
# Evidence is captured through stdout rather than read off a shared filesystem,
# for the reason profiling-demo gives: a deployment record that only assembles on
# clusters where the store is reachable from the compute node is a worse artefact
# than one that assembles anywhere.
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
OUT="${E2E_OUTPUT_DIR:?}"
ARM="${E2E_ARM:?}"

. "$PKG/assets/lib/remote.sh"

SERVE="$PKG/assets/serve"
WORK="${E2E_WORK_ROOT:?}"
CTR="${E2E_CONTAINER:?}"
R="http://${E2E_NODE_IP:?}:${E2E_PORT_ROUTER:?}"

WORKDIR="$(pwd)/round.$ARM"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"
LOG="$WORKDIR/mix_up.log"

say() { printf '[%s] %s\n' "$ARM" "$*"; }

# **The neighbour's occupancy, sampled per step.** T32, as m5 refined it: m3's
# `gpu.txt` below is a `rocm-smi` *product-info* dump taken once, so the card
# **set** survives and the **occupancy during the measurement** does not — and
# that second thing is what separates "the artefact is wrong" from "the
# producer's card had a neighbour". Two of 2026-09-04's worst numbers needed it
# and neither was recoverable afterwards: the DELIVERY-NOTE refusal blamed a
# patch for a neighbour, and the sealed arms' `probe` read 2062 s against 37 s
# on the same budget, 56x, on a contended chassis.
#
# One `rocm-smi` per step, both columns from one invocation — two calls would
# sample two moments and report them as one. `assets/lib/neighbour.py` shapes it
# into the `neighbour` field `round_noise.py` already reads, so m5's judging half
# and this collecting half agree by construction rather than by agreement.
#
# **Never fatal.** A missing sample is a gap in evidence about the environment;
# refusing the round over it would discard a real measurement to protect a
# record of the conditions it was taken under.
NEIGHBOUR_LOG="$WORKDIR/neighbour.jsonl"
sample_neighbour() {
  local step="$1"
  on "rocm-smi --showmemuse --showuse --csv" 2>/dev/null \
    | python3 "$PKG/assets/lib/neighbour.py" --ours "${E2E_GPU_DEVICES:-}" \
        --step "$step" --append "$NEIGHBOUR_LOG" 2>/dev/null \
    || say "neighbour sample at '$step' failed; continuing without it"
}

say "node=$E2E_NODE ip=$E2E_NODE_IP jobid=$E2E_JOBID"
say "model=$E2E_MODEL_PATH tp=$E2E_TP cuda_graph=1 container=$CTR"

# ---- 1. preconditions -------------------------------------------------------
require_visible_on_node "$SERVE/mix_up.sh" "staged task package" || exit 1

if ! on "test -r '$E2E_MODEL_PATH/config.json'" >/dev/null 2>&1; then
  say "ABORT: $E2E_MODEL_PATH/config.json is not readable on $E2E_NODE"
  exit 1
fi

# ---- 2. the mount plan ------------------------------------------------------
# Both arms read it. The stock arm does not apply it, and says so: the two
# deployment records then differ in a stated way rather than an implied one, and
# a reader of the stock record can see which mounts were withheld.
OVERLAY="${AGENT_SYS_INPUT_PATCH_OVERLAY:?}/items/result/mounts.json"
[ -r "$OVERLAY" ] || { say "ABORT: no mount plan at $OVERLAY"; exit 1; }

SPEC=""
PRIOR_STEPS=""
if [ "$ARM" = "patched" ]; then
  # The stock arm's own record of what it did, and the reason this task is
  # allowed to tear that deployment down.
  #
  # **This used to be a graph edge and is now a precondition.** In
  # `integration-demo`, `serve_patched` consumed `bench_stock` — not because it
  # needed the numbers but because the edge said "the stock arm has finished".
  # M5.2 merged the five leaves into one task, so there is no edge left to carry
  # that; what is left is this check, against the stock arm's steps record
  # written earlier in this same task. It is weaker than a scheduler constraint
  # and stronger than a sentence in a readme: the patched bring-up refuses to
  # start if the stock arm did not finish, and `check_measurement_order` refuses
  # the pair afterwards if the two arms overlapped in time.
  # No apostrophe in this message, and it is not a style choice: inside
  # `${VAR:?word}` an unpaired `'` opens a single-quoted string that runs to the
  # end of the file, so `arm's` here made bash swallow everything after line 68
  # and report `syntax error: unexpected end of file` at line 396. The whole
  # file failed `bash -n`.
  PRIOR_STEPS="${E2E_PRIOR_STEPS:?E2E_PRIOR_STEPS must name items/env/steps.json from the stock arm}"
  if [ ! -r "$PRIOR_STEPS" ]; then
    say "ABORT: the stock arm's step record is not readable at $PRIOR_STEPS"
    say "  This task tears the stock deployment down and must not run before it was measured."
    exit 1
  fi
  say "stock arm completed: $(python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(' -> '.join(s['step'] for s in d.get('steps', [])))" "$PRIOR_STEPS")"

  SPEC="$WORKDIR/mounts.tsv"
  python3 "$PKG/assets/lib/patchkit.py" mountspec "$OVERLAY" "$WORK" "$SPEC" || exit 1
  # The engine reads these files at import time and the container is started with
  # them, so a source that is not there is a bring-up that silently runs stock
  # code. Checked on the node, because that is where the mount will be resolved.
  while IFS=$'\t' read -r host inside; do
    on "test -r '$host'" >/dev/null 2>&1 || {
      say "ABORT: mount source missing on $E2E_NODE: $host"
      say "  apply_patch staged it under the node-local work root; the allocation may have changed."
      exit 1
    }
  done < "$SPEC"
  say "$(wc -l < "$SPEC") patch mount(s) will be applied"
else
  say "$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['mounts']))" "$OVERLAY") patch mount(s) deliberately NOT applied"
fi
say "preconditions ok"
sample_neighbour preconditions

# ---- 3. bring the stack up --------------------------------------------------
say "deploying (cold-starts the engine; first load off NFS took 819s, later ones ~243s)"
on "NODE_IP='$E2E_NODE_IP' IMAGE='$E2E_IMAGE' ETCD_IMAGE='$E2E_ETCD_IMAGE' \
    MODEL='$E2E_MODEL_PATH' MODEL_MOUNT='$(dirname "$E2E_MODEL_PATH")' SERVED='$E2E_SERVED_NAME' \
    CTR='$CTR' ROUTER_PORT='$E2E_PORT_ROUTER' PORT='$E2E_PORT_WORKER' \
    ETCD_PORT='$E2E_PORT_ETCD' TP='$E2E_TP' \
    WORK_ROOT='$WORK' CUDA_GRAPH=1 SCRIPTS='$SERVE' MOUNT_SPEC='$SPEC' \
    CTX='${E2E_CTX:-262144}' \
    DSA_ARGS='${E2E_DSA_ARGS:---dsa-prefill-backend tilelang --dsa-decode-backend tilelang}' \
    PARSER_ARGS='${E2E_PARSER_ARGS:---reasoning-parser glm45 --tool-call-parser glm47}' \
    bash '$SERVE/mix_up.sh'" 2>&1 | tee "$LOG"
up_rc="${PIPESTATUS[0]}"

if [ "$up_rc" != "0" ] || ! grep -q MIX_UP_OK "$LOG"; then
  say "deployment failed (rc=$up_rc). Last 40 lines:"
  tail -40 "$LOG" >&2
  exit 1
fi
sample_neighbour deployed
say "deployment up at $R"

# ---- 4. evidence ------------------------------------------------------------
say "collecting evidence"
sample_neighbour evidence
# The environment is passed explicitly, not inherited: `srun --export=ALL` copies
# this process's environment, which carries the IT_* block but none of the names
# mix_smoke.sh reads.
on "NODE_IP='$E2E_NODE_IP' ROUTER_PORT='$E2E_PORT_ROUTER' SERVED='$E2E_SERVED_NAME' \
    CTR='$CTR' bash '$SERVE/mix_smoke.sh'" > "$WORKDIR/smoke.txt" 2>&1
on "curl -s -m10 '$R/v1/workers'" > "$WORKDIR/workers.json" 2>/dev/null
on "curl -s -m10 '$R/v1/models'"  > "$WORKDIR/models.json"  2>/dev/null
on "curl -s -o /dev/null -w '%{http_code}' -m10 '$R/health'" > "$WORKDIR/health.txt" 2>/dev/null

# The engine log TAIL, not the live log: the verdict has to be reproducible from
# the handoff alone, so what a validator scans is the same bytes a reader sees.
on "docker exec $CTR tail -c 400000 /tmp/glm53_mix.log" > "$WORKDIR/worker.tail.log" 2>&1
on "docker exec $CTR tail -c 50000 /tmp/router.log"     > "$WORKDIR/router.tail.log" 2>&1
on "docker inspect $CTR --format '{{.Image}}'"          > "$WORKDIR/image.txt" 2>&1
on "rocm-smi --showproductname 2>/dev/null | head -20"  > "$WORKDIR/gpu.txt" 2>&1
on "docker exec $CTR bash -c 'cat /opt/rocm/.info/version 2>/dev/null || true'" > "$WORKDIR/rocm.txt" 2>&1

# **The engine's argv as it is actually running**, one flag per line. The round's
# defining flags are recorded from what the process HAS, not from what this
# script ASKED for; a self-declared `cuda_graph: 1` is exactly no use against the
# failure a validator is looking for.
on "docker exec $CTR bash -c \"tr '\\0' '\\n' < /proc/\\\$(pgrep -f 'infera.engine.sglang' | head -1)/cmdline\"" \
  > "$WORKDIR/engine_argv.txt" 2>&1
on "docker exec $CTR cat /run_router.sh" > "$WORKDIR/router_cmd.txt" 2>&1

# ---- 5. is the patch live? --------------------------------------------------
# Gathered here and not in the validator, because by the time `check_patch_live`
# runs this deployment may already have been torn down by the next step. A
# validator can only read the handoff.
#
# Three independent facts, and the first one is the one people expect to be
# enough and is not: __file__ inside the container is the SAME string on both
# arms, because a bind mount does not change a path. Only the hash distinguishes
# them.
PATCH_LIVE="$WORKDIR/patch_live.json"
if [ "$ARM" = "patched" ]; then
  on "docker inspect $CTR --format '{{json .Mounts}}'" > "$WORKDIR/docker_mounts.json" 2>&1
  : > "$WORKDIR/container_hashes.tsv"
  while IFS=$'\t' read -r host inside; do
    got="$(on "docker exec $CTR sha256sum '$inside' 2>/dev/null | cut -d' ' -f1" | tr -d '[:space:]')"
    printf '%s\t%s\n' "$inside" "${got:-MISSING}" >> "$WORKDIR/container_hashes.tsv"
  done < "$SPEC"

  # Marker hits, counted out of the engine log tail that travels in the handoff,
  # so the count is checkable from the record rather than from a live log.
  : > "$WORKDIR/marker_hits.tsv"
  python3 "$PKG/assets/lib/patchkit.py" markers "$OVERLAY" | while IFS=$'\t' read -r key rx; do
    n="$(grep -Ec "$rx" "$WORKDIR/worker.tail.log" || true)"
    printf '%s\t%s\t%s\n' "$key" "$rx" "${n:-0}" >> "$WORKDIR/marker_hits.tsv"
  done
  say "marker hits: $(tr '\t' ' ' < "$WORKDIR/marker_hits.tsv" | tr '\n' ';')"
else
  echo '[]' > "$WORKDIR/docker_mounts.json"
  : > "$WORKDIR/container_hashes.tsv"
  : > "$WORKDIR/marker_hits.tsv"
fi

# ---- 6. assemble the handoff ------------------------------------------------
# `reproducible` requires `result` and `env`, plus one of `script` / `command`.
# Item keys are exactly those names -- a directory here is `result`, not
# `results`; check_items rejects any top-level item the type never declared.
sample_neighbour assemble
say "assembling handoff -> $OUT"
ITEMS="$OUT/items"
rm -rf "$ITEMS"; mkdir -p "$ITEMS/result" "$ITEMS/env" "$ITEMS/logs"

cp "$WORKDIR/smoke.txt" "$WORKDIR/workers.json" "$WORKDIR/models.json" \
   "$WORKDIR/health.txt" "$ITEMS/result/"
cp "$WORKDIR/gpu.txt" "$WORKDIR/rocm.txt" "$WORKDIR/image.txt" \
   "$WORKDIR/engine_argv.txt" "$WORKDIR/router_cmd.txt" "$ITEMS/env/"
# **Copied only if a sample was taken.** An absent `neighbour.jsonl` says the
# sampling did not run, which is a different statement from an empty one saying
# nothing was next to us — and a `cp` that fails on the first would abort a round
# over a record of the conditions rather than over the measurement.
[ -s "$NEIGHBOUR_LOG" ] && cp "$NEIGHBOUR_LOG" "$ITEMS/env/neighbour.jsonl"
cp "$WORKDIR/docker_mounts.json" "$WORKDIR/container_hashes.tsv" \
   "$WORKDIR/marker_hits.tsv" "$ITEMS/env/"

MODEL_NAME="$(basename "$E2E_MODEL_PATH")"
cat > "$ITEMS/command" <<EOF
#!/usr/bin/env bash
# Reproduce this arm. \`agent.gate\` requires this item to be executable, so it is
# a script rather than a transcript -- and writing it as one is what makes it
# survive publication: every site path is a shell variable the caller supplies,
# so there is no absolute path for the locality seal to reject.
#
# MOUNT_SPEC is what makes this the \`$ARM\` arm. Leave it empty for stock; point
# it at a host<TAB>container file for patched. items/env/deployment.json records
# which one this run used.
set -eu
: "\${MODEL_MOUNT:?export MODEL_MOUNT=<directory holding the checkpoint>}"
: "\${WORK_ROOT:?export WORK_ROOT=<node-local work area>}"
: "\${SCRIPTS:?export SCRIPTS=<the package's assets/serve directory>}"
MOUNT_SPEC="\${MOUNT_SPEC:-}"

NODE_IP=$E2E_NODE_IP IMAGE=$E2E_IMAGE ETCD_IMAGE=$E2E_ETCD_IMAGE \\
MODEL="\$MODEL_MOUNT/$MODEL_NAME" MODEL_MOUNT="\$MODEL_MOUNT" SERVED=$E2E_SERVED_NAME \\
CTR=$CTR ROUTER_PORT=$E2E_PORT_ROUTER PORT=$E2E_PORT_WORKER ETCD_PORT=$E2E_PORT_ETCD \\
TP=$E2E_TP WORK_ROOT="\$WORK_ROOT" CUDA_GRAPH=1 SCRIPTS="\$SCRIPTS" \\
MOUNT_SPEC="\$MOUNT_SPEC" \\
bash "\$SCRIPTS/mix_up.sh"

NODE_IP=$E2E_NODE_IP SERVED=$E2E_SERVED_NAME CTR=$CTR ROUTER_PORT=$E2E_PORT_ROUTER \\
bash "\$SCRIPTS/mix_smoke.sh"
EOF
chmod +x "$ITEMS/command"

# **The logs go in compressed, and that is not a size decision.** Measured on
# profiling-demo, 817 of 818 absolute paths in one round's logs are false
# positives of the kind handoff.locality's own docstring predicts:
# container-internal paths the image owns, HTTP routes in an access log, an etcd
# key prefix. None is a fact about this machine, and the mechanism that would let
# a kind say so is wired to nothing (temp/bugs/002). Substituting them away would
# corrupt the one artefact whose value is being a faithful record; gzip keeps the
# bytes and the seal skips what it cannot decode as UTF-8.
for f in "$LOG" "$WORKDIR/worker.tail.log" "$WORKDIR/router.tail.log"; do
  gzip -9 -c "$f" > "$ITEMS/logs/$(basename "$f").gz"
done

PKG_COMMIT="$(git -C "$PKG" rev-parse HEAD 2>/dev/null || echo unknown)"
PKG_DIRTY="$(git -C "$PKG" status --porcelain 2>/dev/null | head -1)"
[ -n "$PKG_DIRTY" ] && PKG_COMMIT="$PKG_COMMIT+dirty"

python3 - "$ITEMS/env/deployment.json" "$OVERLAY" "$WORKDIR" "${PRIOR_STEPS:-}" <<PYEOF
import json, sys
dest, overlay, workdir = sys.argv[1:4]
prior = sys.argv[4] if len(sys.argv) > 4 else ""
plan = json.load(open(overlay))
# For the patched arm: what the stock arm had already done when this deployment
# was created. The comparison is only valid if the stock measurement finished
# first, and this is where that becomes checkable rather than assumed.
prior_steps = json.load(open(prior)) if prior else None
json.dump({
    "arm": "$ARM",
    "scripts": {
        "package": "agent_sys/examples/llm_e2e_performance_optimization/integration-demo",
        "entrypoints": ["assets/serve/mix_up.sh", "assets/serve/mix_smoke.sh"],
        "commit": "$PKG_COMMIT",
    },
    "node": "$E2E_NODE",
    "node_ip": "$E2E_NODE_IP",
    "slurm_jobid": "$E2E_JOBID",
    "image": "$E2E_IMAGE",
    "image_id": open(workdir + "/image.txt").read().strip(),
    "model_path": "$E2E_MODEL_PATH",
    "served_model_name": "$E2E_SERVED_NAME",
    "endpoint": "$R",
    "container": "$CTR",
    "ports": {"router": $E2E_PORT_ROUTER, "worker": $E2E_PORT_WORKER, "etcd": $E2E_PORT_ETCD},
    "tp_size": $E2E_TP,
    "disagg_mode": "mixed",
    # What this run ASKED for. What it GOT is env/engine_argv.txt, and that is
    # what check_service_live reads.
    "requested": {"cuda_graph": 1, "profiling_enabled": 0},
    # The plan both arms saw, and what this arm did with it.
    "overlay": {
        "operator_id": plan.get("operator_id"),
        "apply_mode": plan.get("apply_mode"),
        "image": plan.get("image"),
        "declared": len(plan.get("mounts", [])),
        "applied": len(plan.get("mounts", [])) if "$ARM" == "patched" else 0,
        "runtime_marker": plan.get("runtime_marker"),
        "mounts": plan.get("mounts", []),
    },
    "preceded_by": prior_steps,
    "work_root": "$WORK",
}, open(dest, "w"), indent=2)
PYEOF

if [ "$ARM" = "patched" ]; then
  cat > "$ITEMS/watchout" <<'EOF'
This is the PATCHED arm. Numbers measured here mean nothing on their own; they
are only interpretable against the stock arm measured in the same session, on
the same node, against the same trace, in the same order. Do not quote one
without the other.

Two facts have to hold before this deployment counts as having tested anything,
and check_patch_live is what decides:

  env/container_hashes.tsv  what each mounted path hashes INSIDE the running
                            container. This is the only static proof that works.
                            A bind mount does not change the path, so __file__
                            reads identically on a patched and a stock arm.
  env/marker_hits.tsv       how often each marker the patch declared appears in
                            the engine log. The import marker says the bytes were
                            compiled; the first-call marker says the code ran.
                            A patch with no markers can only be proven mounted.

etcd is on a port other than 2379. These nodes run a Kubernetes control plane
whose own etcd holds 2379 over TLS, and a plaintext discovery client pointed at
it fails much later as an empty worker pool.
EOF
else
  cat > "$ITEMS/watchout" <<'EOF'
This is the STOCK arm: the mount plan was read and deliberately not applied.
env/deployment.json records which mounts were withheld, so a reader can tell
this deployment apart from one that never had a plan at all.

It is the baseline, and it is measured in the same session as the patched arm
for a measured reason: profiling-demo replayed the same trace against the same
configuration twice and got 631 output tok/s cold against 1004 with the
deployment reused, because a Mooncake trace carries hash_ids and prefix hit rate
decides how much prefill there is to do. A baseline from another session is not
a baseline.

etcd is on a port other than 2379. These nodes run a Kubernetes control plane
whose own etcd holds 2379 over TLS, and a plaintext discovery client pointed at
it fails much later as an empty worker pool.
EOF
fi

cat > "$OUT/README.md" <<EOF
# deployment_$ARM

## Purpose

A record of one GLM-5.3-Flash MIX deployment on a single MI355X node, and the
evidence that it served. This is the **$ARM** arm of a two-arm comparison:
$( [ "$ARM" = patched ] && echo "the patch's files are bind-mounted read-only over their paths in the image" || echo "the mount plan was read and none of it applied" ).

Decode CUDA graphs are on, which is the configuration worth quoting a number
from. Nothing about this deployment is profiled: the profiling control plane is
off and no trace directory is mounted, so the two arms differ in the mounts and
in nothing else.

## How to run

\`items/command\` is the invocation. The scripts it names are not copied in here;
\`items/env/deployment.json\` says which package and commit they came from.

Every site path in this handoff is written as \`@NAME@\`. On the machine that
produced it, \`@MODEL_MOUNT@\` was the directory holding the checkpoint and
\`@WORK_ROOT@\` the node-local work area. They are placeholders because
\`handoff\` refuses to seal content that names one machine's paths — the right
rule, and the reason this record transfers. \`items/command\` takes the same
paths as shell variables, so it runs once they are exported.

## Result

The endpoint is \`$R\`. \`items/result/\` carries the proof: \`workers.json\`
(expected: exactly one worker, \`disagg_mode\` \`mixed\`), \`models.json\`,
\`health.txt\` and \`smoke.txt\` — five blocks ending in an arithmetic answer that
must be 391 and an engine-side scan for faults.

## Environment

\`items/env/deployment.json\` carries the shape: node, image id, model path,
ports, TP size, and the mount plan with a count of how many of its entries this
arm applied. \`engine_argv.txt\` is the engine's command line as the process
actually has it, which is what decides the arm rather than anything declared.
\`container_hashes.tsv\` and \`marker_hits.tsv\` are the patch-live evidence and
are empty on the stock arm by construction.

## Watch out

See \`items/watchout\`.
EOF

# ---- 7. make it publishable -------------------------------------------------
say "redacting site-specific paths"
# The container roots come from the same table that produced the placeholders in
# mounts.json, because the patched arm's evidence names them and nothing else in
# this package does: `docker inspect`'s Mounts and the sha256 taken inside the
# container both report the real destination, and the seal refuses an absolute
# path whether it belongs to this machine or to every copy of the image.
#
# Measured: without this, `serve_patched` wrote its whole handoff and then
# aborted here, on `items/env/container_hashes.tsv` and
# `items/env/docker_mounts.json`.
mapfile -t ROOT_ARGS < <(python3 "$PKG/assets/lib/patchkit.py" redact-args)
python3 "$PKG/assets/lib/redact.py" "$OUT" \
  "MODEL_MOUNT=$(dirname "$E2E_MODEL_PATH")" \
  "WORK_ROOT=$WORK" \
  "TASK_PACKAGE=$PKG" \
  "TMPDIR=/tmp" \
  "HOME=$HOME" \
  "${ROOT_ARGS[@]}" || {
    say "ABORT: evidence still names paths the handoff seal will refuse"
    exit 1
  }

say "done: $(find "$ITEMS" -type f | wc -l) evidence files under $ITEMS"
exit 0
