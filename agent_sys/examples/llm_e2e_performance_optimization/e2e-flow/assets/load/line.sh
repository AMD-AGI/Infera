#!/usr/bin/env bash
# One line of stage 2, end to end: bring an engine up, load it, tear it down.
#
# **Both of m2's lines are this file with two flags flipped**, the way
# `profiling-demo`'s two serve tasks were one `round.sh`. Keeping them on one
# implementation is what stops them drifting into two deployments that differ in
# more than the axis under test, and the axis is exactly two switches:
#
#   E2E_MODE=profiling_mode_off   CUDA graph ON,  profiler detached
#   E2E_MODE=profiling_mode_on    CUDA graph OFF, profiler attached
#
# The numbers that mean something come from the first. The second exists because
# a graph launch hides the kernels the profiler is there to see: with graphs on
# the profiler records one launch instead of the kernels inside it, so its
# throughput is not a control for anything and is not quoted as one.
#
# **There is no `serve_*` task and there is no `deployment_*` handoff** — mission
# M2.3, M2.4 and M2.5: *"agent A 去把服务部署好，agent B 去使用：这是不被允许的"*.
# A task that needs a service brings it up itself and tears it down itself, which
# is this file. m1's `deploy_kit` carries the environment record it brings it up
# against, and this reads that record rather than re-deriving it.
#
# Runs on the LOGIN NODE. Everything that touches a GPU goes through
# `assets/lib/remote.sh`'s `on`, which dispatches on `$E2E_TRANSPORT` — **no
# transport is spelled anywhere in this package's readmes or step files** (M2.1).
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
MODE="${E2E_MODE:?E2E_MODE=profiling_mode_off|profiling_mode_on}"
KIT="${AGENT_SYS_INPUT_DEPLOY_KIT:?}"

. "$PKG/assets/lib/remote.sh"

SERVE="$PKG/assets/serve"
LOAD="$PKG/assets/load"
WORK="${E2E_WORK_ROOT:?}"

say() { printf '[%s] %s\n' "$MODE" "$*"; }

case "$MODE" in
  profiling_mode_off) CUDA_GRAPH=1; PROFILE=0; CAPTURE=0; SUFFIX=pmoff ;;
  profiling_mode_on)  CUDA_GRAPH=0; PROFILE=1; CAPTURE=1; SUFFIX=pmon  ;;
  *) echo "E2E_MODE must be profiling_mode_off or profiling_mode_on, got '$MODE'" >&2; exit 2 ;;
esac

# **Its own container, and its own ports.** CONTRACT §5.2: never `docker rm -f` a
# name you did not create, and `mix_up.sh` starts with an idempotent teardown of
# the name it is given. Reusing m1's name would mean this task destroying m1's
# container; deriving one per mode means each line destroys only what it made,
# and the two lines can be scheduled in either order without colliding.
#
# The port offset is for the same reason and is not about parallelism: with
# `--network=host` every port is a host port, and a line that crashed leaving a
# listener behind must not silently take over the other line's endpoint.
CTR="${E2E_CONTAINER}_${SUFFIX}"
case "$MODE" in
  profiling_mode_off) OFF=0 ;;
  profiling_mode_on)  OFF=10 ;;
esac
ROUTER_PORT=$((E2E_PORT_ROUTER + OFF))
WORKER_PORT=$((E2E_PORT_WORKER + OFF))
ETCD_PORT=$((E2E_PORT_ETCD + OFF))
TRACE_OUT="$WORK/$SUFFIX/profiles"
R="http://${E2E_NODE_IP:?}:$ROUTER_PORT"

WORKDIR="$(pwd)/line.$MODE"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

# ---- 1. what m1 handed us ----------------------------------------------------
# The record, not the variables. `deploy_kit` is a `code` handoff, so its
# environment record is at `items/codes/environment.yaml` (CONTRACT §2). Reading
# it is the whole of M2.4 — the deployment kind was deleted because m1's output
# already carries this.
ENV_IN="$KIT/items/codes/environment.yaml"
[ -r "$ENV_IN" ] || { say "ABORT: deploy_kit carries no environment record at $ENV_IN"; exit 1; }

eval "$(python3 - "$ENV_IN" <<'PY'
import shlex, sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
fixed = doc.get("fixed") or {}
for key in ("node", "image", "image_id", "model_path", "model_name", "served_model_name", "tp_size"):
    print(f"KIT_{key.upper()}={shlex.quote(str(fixed.get(key, '')))}")
PY
)"

# A mismatch here means this line is about to measure a machine the kit does not
# describe, and every number it produces would be filed under the wrong
# environment. `check_environment`'s `require_runtime_agrees_across_inputs` would
# catch it three tasks later; catching it now costs nothing.
if [ -n "$KIT_NODE" ] && [ "$KIT_NODE" != "${E2E_NODE:?}" ]; then
  say "ABORT: deploy_kit was taken on '$KIT_NODE' and this run is pointed at '$E2E_NODE'"
  exit 1
fi
: "${E2E_SERVED_NAME:=${KIT_SERVED_MODEL_NAME:-$E2E_MODEL_NAME}}"
say "kit: node=$KIT_NODE image=$KIT_IMAGE tp=$KIT_TP_SIZE served=$E2E_SERVED_NAME"
say "this line: container=$CTR router=$ROUTER_PORT graph=$CUDA_GRAPH profile=$PROFILE"

require_visible_on_node "$SERVE/mix_up.sh" "staged task package" || exit 1
if ! on "test -r '${E2E_MODEL_PATH:?}/config.json'" >/dev/null 2>&1; then
  say "ABORT: $E2E_MODEL_PATH/config.json is not readable on $E2E_NODE"
  exit 1
fi

# ---- 2. teardown is registered before bring-up, not after --------------------
# A line that fails between bring-up and load must not leave a TP-8 engine
# holding every GPU on a node four other owners share. The trap fires on the
# error paths below as well as on success, and it removes only this line's own
# container.
teardown() {
  local rc=$?
  say "tearing down $CTR"
  on "docker rm -f '$CTR' '${CTR}_etcd' >/dev/null 2>&1; true" >/dev/null 2>&1
  return $rc
}
trap teardown EXIT

# ---- 3. bring the engine up --------------------------------------------------
say "deploying (cold start; the checkpoint load off shared storage dominates)"
on "NODE_IP='$E2E_NODE_IP' IMAGE='${E2E_IMAGE:?}' ETCD_IMAGE='${E2E_ETCD_IMAGE:?}' \
    MODEL='$E2E_MODEL_PATH' MODEL_MOUNT='$(dirname "$E2E_MODEL_PATH")' \
    SERVED='$E2E_SERVED_NAME' CTR='$CTR' \
    ROUTER_PORT='$ROUTER_PORT' PORT='$WORKER_PORT' ETCD_PORT='$ETCD_PORT' \
    TP='${E2E_TP:?}' WORK_ROOT='$WORK/$SUFFIX' \
    CUDA_GRAPH='$CUDA_GRAPH' PROFILE='$PROFILE' TRACE_OUT='$TRACE_OUT' \
    SCRIPTS='$SERVE' CTX='${E2E_CTX:?}' \
    DSA_ARGS='$E2E_DSA_ARGS' PARSER_ARGS='$E2E_PARSER_ARGS' \
    bash '$SERVE/mix_up.sh'" 2>&1 | tee "$WORKDIR/mix_up.log"
up_rc="${PIPESTATUS[0]}"
if [ "$up_rc" != "0" ] || ! grep -q MIX_UP_OK "$WORKDIR/mix_up.log"; then
  say "deployment failed (rc=$up_rc). Last 40 lines:"
  tail -40 "$WORKDIR/mix_up.log" >&2
  exit 1
fi
say "deployment up at $R"

# ---- 4. the load, and the profiler windows inside it -------------------------
E2E_LOAD_ROUND="$MODE" \
E2E_CAPTURE="$CAPTURE" \
E2E_CONTAINER="$CTR" \
E2E_PORT_ROUTER="$ROUTER_PORT" \
E2E_WORK_ROOT="$WORK/$SUFFIX" \
bash "$LOAD/replay.sh" || exit 1

# ---- 5. rank the kernels, profiler-attached line only ------------------------
# One task, not two. The ranking reads the trace this task just produced, and
# splitting it off would mean a second task re-staging 130 MB of traces to do
# three minutes of work against a deployment that no longer exists.
if [ "$CAPTURE" = "1" ]; then
  STAGING="$WORKDIR/kernel_table.staging"
  mkdir -p "$STAGING"
  E2E_INPUT_TORCH_TRACE="${E2E_OUTPUT_TRACE:?}" \
  E2E_OUTPUT_KERNEL_TABLE="$STAGING" \
  bash "$PKG/assets/analyze/scan.sh" || exit 1

  # `profiling_mode_on.kernel_table` is a `structured_text` kind: the record is
  # `items/text.json`, Magpie's export is `items/table.csv`, and the schema
  # travels with it (CONTRACT §3.4). `scan.sh` writes the `reproducible` shape
  # the standalone package used, so the shape is corrected here rather than by
  # forking a 249-line script that is otherwise right.
  python3 "$PKG/assets/lib/m2_reshape.py" kernel_table \
    "$STAGING" "${E2E_OUTPUT_KERNEL_TABLE:?}" || exit 1
fi

# ---- 6. the environment record, on every output ------------------------------
# Mission G5. **Inherited, never rebuilt**: m1 is the sole producer, and a stage
# that re-derived the record could differ from m1's with nothing to notice. The
# runtime half is overridden because this line brought up its own container, and
# that is the fact a later reader needs in order to know which engine these
# numbers came from.
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for out in ${E2E_OUTPUT_AIPERF:-} ${E2E_OUTPUT_TRACE:-} ${E2E_OUTPUT_KERNEL_TABLE:-}; do
  [ -d "$out" ] || continue
  python3 "$PKG/assets/lib/env_render.py" --inherit "$ENV_IN" \
    --content-type "$( [ "$out" = "${E2E_OUTPUT_KERNEL_TABLE:-}" ] && echo structured_text || echo reproducible )" \
    --out "$out" \
    --set "runtime.container=$CTR" \
    --set "runtime.endpoint=$R" \
    --set "runtime.started_at=$NOW" \
    --set "runtime.ports={\"router\":$ROUTER_PORT,\"worker\":$WORKER_PORT,\"etcd\":$ETCD_PORT}" || exit 1
done

say "done"
exit 0
