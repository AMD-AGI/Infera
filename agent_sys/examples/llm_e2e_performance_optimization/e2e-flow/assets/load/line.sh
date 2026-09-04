#!/usr/bin/env bash
# One line of stage 2, end to end: bring an engine up **from m1's kit**, load it,
# tear it down.
#
# **Both of m2's lines are this file with two configuration values changed.**
# Keeping them on one implementation is what stops them drifting into two
# deployments that differ in more than the axis under test, and the axis is
# exactly two switches:
#
#   E2E_MODE=profiling_mode_off   CUDA graph ON,  profiler detached
#   E2E_MODE=profiling_mode_on    CUDA graph OFF, profiler attached
#
# The numbers that mean something come from the first. The second exists because
# a graph launch hides the kernels the profiler is there to see: with graphs on
# the profiler records one launch instead of the kernels inside it, so its
# throughput is not a control for anything and is not quoted as one.
#
# **This file contains no deployment recipe, and that is M2.3/M2.4.** *"module 1
# 的 output 已经包含了如何部署的全量信息"* — so the bring-up is `deploy.sh` out of
# the `deploy_kit` handoff, the readiness criterion is that kit's
# `wait_ready.sh`, and the teardown is its `teardown.sh`. What this file adds is
# the two-line configuration, the load, and the evidence. An earlier version
# drove `assets/serve/mix_up.sh` directly; that worked and was a second copy of
# m1's launch, which is precisely the duplication M2.3 removes.
#
# **There is no `serve_*` task and no `deployment_*` handoff** — M2.5: *"agent A
# 去把服务部署好，agent B 去使用：这是不被允许的"*. A task that needs a service
# brings it up itself and tears it down itself, which is this file.
#
# Runs on the LOGIN NODE. Everything that touches a GPU goes through
# `assets/lib/remote.sh`'s `on`, which dispatches on `$E2E_TRANSPORT` — **no
# transport is spelled anywhere in this package's readmes or step files** (M2.1).
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
MODE="${E2E_MODE:?E2E_MODE=profiling_mode_off|profiling_mode_on}"
KIT="${AGENT_SYS_INPUT_DEPLOY_KIT:?}"

. "$PKG/assets/lib/remote.sh"

LOAD="$PKG/assets/load"
WORK="${E2E_WORK_ROOT:?}"

say() { printf '[%s] %s\n' "$MODE" "$*"; }

case "$MODE" in
  profiling_mode_off) CAPTURE=0; SUFFIX=pmoff; PORT_OFFSET=0  ;;
  profiling_mode_on)  CAPTURE=1; SUFFIX=pmon;  PORT_OFFSET=10 ;;
  *) echo "E2E_MODE must be profiling_mode_off or profiling_mode_on, got '$MODE'" >&2; exit 2 ;;
esac

# **The kit's runtime contract**, `assets/schemas/deploy_kit.layout.yaml`
# §runtime_contract. `check_deploy_kit` refuses a kit that does not read these,
# so m2's dependency on them is a gate on m1's output rather than a hope.
#
# `E2E_KIT_RUN_TAG` is what scopes teardown: every container name, port lock and
# label the kit creates carries it, so `teardown.sh` removes what this
# invocation created **and nothing else** — the host's `docker ps` shows every
# tenant's containers (CONTRACT §5.2).
export E2E_KIT_RUN_TAG="${E2E_CONTAINER}_${SUFFIX}"
export E2E_KIT_PORT_BASE=$((E2E_PORT_ROUTER + PORT_OFFSET))
# **What we ask the kit for, not where things end up.** The kit derives its own
# run directory from this and declares the result in the handshake, and on the
# real kit the two differ by a level: `pick_params.sh` writes `DK_RUN_DIR` into
# `run.env` and `env.sh` sources it after any export, so the mount is
# `${WORK_ROOT}/${TAG}` and no amount of exporting from here wins that race.
# **Neither side of the trace directory is computed from this** — both are read
# out of the handshake in step 4. An earlier version derived the host side here
# and would have collected traces from a directory the engine never wrote to.
export E2E_KIT_WORK_ROOT="$WORK/$SUFFIX"

# The two configuration values that are the whole difference between the lines.
#
# `EXTRA_ARGS` is appended to the worker's argv **last**, so it overrides an
# earlier occurrence of the same flag rather than racing it. `EXTRA_ENV` reaches
# the worker process only and not the container, which is why the profiler
# directory is set here and not exported around it.
export E2E_KIT_ENGINE_EXTRA_ARGS=""
export E2E_KIT_ENGINE_EXTRA_ENV=""
# `--enable-profiling` is a **router** flag and the engine seams reach the
# worker, so it needs its own seam. Without it the admin profile routes answer
# 403 and every capture produces nothing while reporting success.
export E2E_KIT_ROUTER_EXTRA_ARGS=""
if [ "$CAPTURE" = "1" ]; then
  E2E_KIT_ENGINE_EXTRA_ARGS="--disable-cuda-graph"
  E2E_KIT_ROUTER_EXTRA_ARGS="--enable-profiling"
  # **`E2E_KIT_ENGINE_EXTRA_ENV` stays empty, and `SGLANG_TORCH_PROFILER_DIR` is
  # not set at all.** I named it as the example when asking m1 for the seams and
  # it was wrong: the working pipeline never sets it. The engine is told where to
  # write **per capture**, in the `/start_profile` request body's `output_dir`
  # (`../load/capture.sh`) — which is also what makes the two windows able to
  # write to different subdirectories of one round.
  #
  # That removes what would otherwise be a chicken-and-egg: a start-time
  # environment variable would have to name a container-side path that only the
  # handshake reveals, i.e. after the engine is already up.
fi

WORKDIR="$(pwd)/line.$MODE"
rm -rf "$WORKDIR"; mkdir -p "$WORKDIR"

# ---- 1. locate the kit -------------------------------------------------------
# A `code` handoff holds exactly one packup directory under `items/codes/`.
# Both other cardinalities are real failures: zero means the producer wrote its
# kit somewhere the content type does not put it, and two means a consumer has
# to guess which one worked.
ENV_IN="$KIT/items/codes/environment.yaml"
[ -r "$ENV_IN" ] || { say "ABORT: deploy_kit carries no environment record at $ENV_IN"; exit 1; }

SCRIPTS="$(python3 - "$KIT" <<'PY'
import sys
from pathlib import Path
codes = Path(sys.argv[1]) / "items" / "codes"
found = sorted(p for p in codes.iterdir() if p.is_dir() and (p / "scripts").is_dir())
if len(found) != 1:
    print(f"expected one packup directory with scripts/ under items/codes, found "
          f"{[p.name for p in found]}", file=sys.stderr)
    raise SystemExit(1)
print(found[0] / "scripts")
PY
)" || { say "ABORT: could not locate the kit's scripts"; exit 1; }
say "kit scripts: $SCRIPTS"

for script in deploy.sh wait_ready.sh teardown.sh; do
  [ -r "$SCRIPTS/$script" ] || { say "ABORT: the kit has no $script"; exit 1; }
done
require_visible_on_node "$SCRIPTS/deploy.sh" "staged deploy_kit" || exit 1

# The record, not the variables. `deploy_kit` is a `code` handoff, so its
# environment record is at `items/codes/environment.yaml` (CONTRACT §2).
eval "$(python3 - "$ENV_IN" <<'PY'
import shlex, sys, yaml
doc = yaml.safe_load(open(sys.argv[1]))
fixed = doc.get("fixed") or {}
for key in ("node", "image", "image_id", "model_name", "served_model_name", "tp_size"):
    print(f"KIT_{key.upper()}={shlex.quote(str(fixed.get(key, '')))}")
# `gpu_devices` is a **list** in the record and a comma string to the kit.
devs = fixed.get("gpu_devices") or []
if not isinstance(devs, (list, tuple)):
    devs = [devs]
print(f"KIT_GPU_DEVICES={shlex.quote(','.join(str(d) for d in devs))}")
# **Provenance, carried into the numbers rather than into a message.** A kit
# whose `runtime.replayed_from` is set was replayed rather than produced by a
# bring-up on this node, so the scripts are real and the record was re-rendered.
# The distinction decides whether a number here is comparable to the deployment
# the kit originally described, and a reader meets the number long after the
# message that qualified it.
rt = doc.get("runtime") or {}
print(f"KIT_REPLAYED_FROM={shlex.quote(str(rt.get('replayed_from') or ''))}")
PY
)"

# **Which cards — the seventh kit variable, and the one that was missing.**
#
# m1's kit defaults `: "${E2E_KIT_GPU_DEVICES:=0,1,2,3}"`, a hardcoded literal,
# and **nothing in the kit reads a card**: no `rocm-smi`, no selection logic,
# and its preflight covers ports and container names only. Until this block m2
# exported the other six kit variables and not this one, so **both bring-ups
# took cards 0–3 whatever was actually free.** On a node where 0–3 are held by a
# co-tenant that is an OOM after a full bring-up — 249 was such a node this
# morning, and 235 became one four minutes after a hold landed on it.
#
# **The escape hatch that saved rung 1 does not exist here.** m1 could route
# "take only 4–7" through `E2E_INSTRUCTION` because their leaf is an agent that
# reads prose. This leaf is a program. That makes RUN-PLAN's *"no `--var` names
# a GPU set, only a count"* strictly worse one stage over, and it is why this
# needs a variable rather than a note.
#
# Precedence, and the middle rule is the one worth arguing for:
#   1. `$E2E_GPU_DEVICES` when it names a list — the operator said which, and
#      `none` means "choose freely" rather than "no cards" (`shared.yaml`).
#   2. otherwise **the set the kit records having taken**. m2 measures the
#      deployment m1 described, so m1's own cards are the right default: that
#      field is deliberately *what was taken* and not *what was free*, because
#      free-at-preflight is a measurement of a moment and stale when read.
#   3. otherwise leave it unset and let the kit decide — today's behaviour, and
#      no worse than it was.
if [ "${E2E_GPU_DEVICES:-none}" != "none" ] && [ -n "${E2E_GPU_DEVICES:-}" ]; then
  export E2E_KIT_GPU_DEVICES="$E2E_GPU_DEVICES"
  say "cards: $E2E_KIT_GPU_DEVICES (from --var gpu_devices)"
elif [ -n "${KIT_GPU_DEVICES:-}" ]; then
  export E2E_KIT_GPU_DEVICES="$KIT_GPU_DEVICES"
  say "cards: $E2E_KIT_GPU_DEVICES (the set the deploy_kit records taking)"
else
  say "cards: not named here; the kit's own default applies"
fi

# A mismatch here means this line is about to measure a machine the kit does not
# describe, and every number it produces would be filed under the wrong
# environment. `check_environment`'s `require_runtime_agrees_across_inputs`
# would catch it three tasks later; catching it now costs nothing.
if [ -n "$KIT_NODE" ] && [ "$KIT_NODE" != "${E2E_NODE:?}" ]; then
  say "ABORT: deploy_kit was taken on '$KIT_NODE' and this run is pointed at '$E2E_NODE'"
  exit 1
fi
# **Exported, not merely assigned.** `replay.sh` runs as a child and reads this
# under `set -u`. It worked only because `shared.yaml` ships
# `E2E_SERVED_NAME: '${served_name:-}'`, so the variable arrives *exported and
# empty* and `:=` preserves the export attribute — i.e. correctness depended on
# how the variable happened to arrive rather than on anything this file does.
# Found by running the real path against a stub kit, where the variable was
# absent instead of empty and `replay.sh` died with `unbound variable`.
: "${E2E_SERVED_NAME:=${KIT_SERVED_MODEL_NAME:-$E2E_MODEL_NAME}}"
export E2E_SERVED_NAME
say "kit: node=$KIT_NODE image=$KIT_IMAGE tp=$KIT_TP_SIZE served=$E2E_SERVED_NAME"
export KIT_REPLAYED_FROM
[ -n "${KIT_REPLAYED_FROM:-}" ] && say "kit was REPLAYED from $KIT_REPLAYED_FROM — numbers prove the path, not that deployment"
say "this line: run_tag=$E2E_KIT_RUN_TAG port_base=$E2E_KIT_PORT_BASE"
say "  engine argv += '${E2E_KIT_ENGINE_EXTRA_ARGS:-<none>}'"
say "  engine env  += '${E2E_KIT_ENGINE_EXTRA_ENV:-<none>}'"
say "  router argv += '${E2E_KIT_ROUTER_EXTRA_ARGS:-<none>}'"

# **No preflight of our own.** The kit owns that, and duplicating it here is the
# duplication M2.3 removes — a second opinion about whether the checkpoint is
# readable is a second thing to keep in step with the deployment it describes.

# ---- 2. teardown is registered before bring-up, not after --------------------
# A line that fails between bring-up and load must not leave a TP-8 engine
# holding every GPU on a node four other owners share. The trap fires on the
# error paths below as well as on success. It is the kit's own teardown, scoped
# by the same run tag, so it removes what this invocation created and nothing a
# co-tenant owns.
KIT_ENV_PREFIX="E2E_KIT_RUN_TAG='$E2E_KIT_RUN_TAG' \
  E2E_KIT_PORT_BASE='$E2E_KIT_PORT_BASE' \
  E2E_KIT_WORK_ROOT='$E2E_KIT_WORK_ROOT'"

teardown() {
  local rc=$?
  # **Reclaim before teardown, because reclaim needs the container alive.**
  # CONTRACT §5.0: the engine and the AIPerf container run as root, so what they
  # write into this invocation's work root — the traces under `profiles/` and
  # AIPerf's own export under `aiperf/` — is root-owned, and only a process
  # inside the same container can give it back.
  #
  # m2's *handoffs* are not affected: `replay.sh` and `scan.sh` **copy** into
  # `$AGENT_SYS_OUTPUT_<KIND>` with `cp` run as the zone's user, so the copies
  # are user-owned and only the originals on the node are root's. That is why
  # this is about the node's disk rather than about the seal — but §5.0 says to
  # call it in a `finally` without deciding first whether it will work, and it
  # is idempotent and a no-op once the container is gone, so it is called
  # unconditionally rather than behind a judgement that could be wrong.
  #
  # `$HS_WORK_ROOT_IN_CONTAINER` is unset if we failed before the handshake; in
  # that case nothing was written and there is nothing to reclaim.
  if [ -n "${HS_WORK_ROOT_IN_CONTAINER:-}" ] && [ -n "${CTR:-}" ]; then
    on "bash '$PKG/assets/lib/reclaim.sh' '$CTR' '$HS_WORK_ROOT_IN_CONTAINER'" \
      >"$WORKDIR/reclaim.log" 2>&1 \
      || say "WARN: reclaim exited non-zero; see $WORKDIR/reclaim.log"
  fi
  say "tearing down run_tag=$E2E_KIT_RUN_TAG"
  on "$KIT_ENV_PREFIX bash '$SCRIPTS/teardown.sh'" >"$WORKDIR/teardown.log" 2>&1 \
    || say "WARN: teardown exited non-zero; see $WORKDIR/teardown.log"
  return $rc
}
trap teardown EXIT

# ---- 3. bring the engine up, from the kit ------------------------------------
say "deploying (cold start; the checkpoint load off shared storage dominates)"
on "$KIT_ENV_PREFIX \
    E2E_KIT_ENGINE_EXTRA_ARGS='$E2E_KIT_ENGINE_EXTRA_ARGS' \
    E2E_KIT_ENGINE_EXTRA_ENV='$E2E_KIT_ENGINE_EXTRA_ENV' \
    E2E_KIT_ROUTER_EXTRA_ARGS='$E2E_KIT_ROUTER_EXTRA_ARGS' \
    bash '$SCRIPTS/deploy.sh'" 2>&1 | tee "$WORKDIR/deploy.log"
deploy_rc="${PIPESTATUS[0]}"
if [ "$deploy_rc" != "0" ]; then
  say "deploy.sh failed (rc=$deploy_rc). Last 40 lines:"
  tail -40 "$WORKDIR/deploy.log" >&2
  exit 1
fi

# **`deploy.sh` returning 0 says the launch commands were accepted; this says the
# service answered.** They are separate on purpose, and this is the criterion: a
# readiness wait that exits 0 when it gave up turns "the model never loaded"
# into "the benchmark measured nothing", discovered three stages later.
say "waiting for the service to answer"
on "$KIT_ENV_PREFIX bash '$SCRIPTS/wait_ready.sh'" 2>&1 | tee "$WORKDIR/wait_ready.log"
if [ "${PIPESTATUS[0]}" != "0" ]; then
  say "wait_ready.sh timed out; the deployment never answered"
  tail -40 "$WORKDIR/wait_ready.log" >&2
  exit 1
fi

# ---- 4. the handshake --------------------------------------------------------
# JSON on the node's local disk, so it is read through the transport rather than
# off a shared path. `endpoint` is the **product** endpoint — the router, not
# the engine's own port.
on "cat '$E2E_KIT_WORK_ROOT/deployment.json'" > "$WORKDIR/deployment.json" 2>/dev/null \
  || { say "ABORT: deploy.sh wrote no handshake at $E2E_KIT_WORK_ROOT/deployment.json"; exit 1; }

eval "$(python3 - "$WORKDIR/deployment.json" <<'PY'
import json, shlex, sys
doc = json.load(open(sys.argv[1]))
# **Both sides of the work root are required, and neither is derived.** The kit
# chooses where it mounts — the proven kit uses `/workdir` — so a consumer that
# assumed "same path inside" would hand the engine a path it cannot write. And
# the host side is not `$E2E_KIT_WORK_ROOT` either: the kit appends its own run
# directory, so a consumer that inferred it collects from a directory the engine
# never wrote to. Measured by m1 on the real kit; both keys exist so that
# neither inference has to be made.
missing = [k for k in ("endpoint", "container", "run_tag",
                       "work_root_in_container", "work_root_on_host")
           if not doc.get(k)]
if missing:
    print(f"the handshake is missing {missing}", file=sys.stderr)
    raise SystemExit(1)
for key in ("endpoint", "container", "run_tag", "work_root_in_container",
            "work_root_on_host", "engine_endpoint"):
    print(f"HS_{key.upper()}={shlex.quote(str(doc.get(key, '')))}")
PY
)" || { say "ABORT: the handshake is not usable"; exit 1; }

if [ "$HS_RUN_TAG" != "$E2E_KIT_RUN_TAG" ]; then
  say "ABORT: the handshake carries run_tag='$HS_RUN_TAG' and this line asked for"
  say "  '$E2E_KIT_RUN_TAG'. Either a stale deployment.json is being read, or the"
  say "  kit ignored the tag — and teardown is scoped by it."
  exit 1
fi
R="$HS_ENDPOINT"
CTR="$HS_CONTAINER"
# One directory, two names, **both declared and neither computed**.
# `capture.sh` needs both: the host name to create and collect, the container
# name for the mount check and for the profiler's own `output_dir`.
TRACE_OUT="$HS_WORK_ROOT_ON_HOST/profiles"
TRACE_OUT_IN_CONTAINER="$HS_WORK_ROOT_IN_CONTAINER/profiles"
say "deployment up at $R in $CTR"
say "work root: $HS_WORK_ROOT_ON_HOST (host) = $HS_WORK_ROOT_IN_CONTAINER (in $CTR)"
if [ "$HS_WORK_ROOT_ON_HOST" != "$E2E_KIT_WORK_ROOT" ]; then
  # Expected on the real kit, and worth printing rather than silently absorbing:
  # it is the difference between what we asked for and where the kit put it.
  say "  (asked for $E2E_KIT_WORK_ROOT; the kit placed its run directory below it)"
fi

# ---- 5. the profiling control plane, profiler-attached line only -------------
# Probed with a role that cannot exist: the engine checks the 403 gate BEFORE it
# validates the role, so 400 means profiling is on and 403 means it is not, and
# neither touches a running profile. **Fatal**, because a capture against a 403
# control plane produces no trace and no error the caller sees.
if [ "$CAPTURE" = "1" ]; then
  code=$(on "curl -s -o /dev/null -w '%{http_code}' -m 10 \
             -X POST '$R/v1/admin/profile/start?role=__probe__'" | tr -d ' \r\n')
  case "$code" in
    400) say "profiling control plane ON (probe -> 400 invalid role)" ;;
    403) say "ABORT: the profiling control plane is OFF. The router was brought up"
         say "  without --enable-profiling, so every capture below would produce"
         say "  nothing and report success. Check E2E_KIT_ROUTER_EXTRA_ARGS reached it."
         exit 1 ;;
    *)   say "WARN: unexpected probe status '$code'; continuing and letting the"
         say "  capture's own CAPTURE_OK be the criterion" ;;
  esac
fi

# ---- 6. the load, and the profiler windows inside it -------------------------
E2E_LOAD_ROUND="$MODE" \
E2E_CAPTURE="$CAPTURE" \
E2E_CONTAINER="$CTR" \
E2E_PORT_ROUTER="$E2E_KIT_PORT_BASE" \
E2E_WORK_ROOT="$HS_WORK_ROOT_ON_HOST" \
E2E_TRACE_OUT_IN_CONTAINER="$TRACE_OUT_IN_CONTAINER" \
bash "$LOAD/replay.sh" || exit 1

# ---- 7. rank the kernels, profiler-attached line only ------------------------
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

# ---- 8. the environment record, on every output ------------------------------
# Mission G5. **Inherited, never rebuilt**: m1 is the sole producer, and a stage
# that re-derived the record could differ from m1's with nothing to notice. The
# runtime half is overridden because this line brought up its own deployment,
# and that is the fact a later reader needs in order to know which engine these
# numbers came from — the two lines are two bring-ups by design, so `container`
# legitimately differs between them and `check_profiling_evidence` compares
# node and image digest across the lines rather than the container.
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for out in ${E2E_OUTPUT_AIPERF:-} ${E2E_OUTPUT_TRACE:-} ${E2E_OUTPUT_KERNEL_TABLE:-}; do
  [ -d "$out" ] || continue
  python3 "$PKG/assets/lib/env_render.py" --inherit "$ENV_IN" \
    --content-type "$( [ "$out" = "${E2E_OUTPUT_KERNEL_TABLE:-}" ] && echo structured_text || echo reproducible )" \
    --out "$out" \
    --set "runtime.container=$CTR" \
    --set "runtime.endpoint=$R" \
    --set "runtime.started_at=$NOW" \
    --set "runtime.ports={\"router\":$E2E_KIT_PORT_BASE}" || exit 1
done

say "done"
exit 0
