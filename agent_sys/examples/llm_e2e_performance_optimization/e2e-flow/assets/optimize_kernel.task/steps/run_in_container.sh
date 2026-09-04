#!/usr/bin/env bash
# Run a command where m4's work actually belongs: inside m1's container.
#
#     run_in_container.sh [--workdir DIR] <command ...>
#
# **`docker exec`, not `docker run`, and the difference is the contract.**
# CONTRACT §5: *"Modules 1–4 share one container on one held node. m1 brings it
# up and records it in `environment.runtime`; m2, m3 and m4 exec into it."* So
# m4 does not get to start a container — it joins the one the record names. That
# is also why nothing here ever removes one: **never `docker rm -f` a container
# you did not create** (§5.2), and by construction m4 created none.
#
# m3's `assets/build_workset.task/measure_in_container.sh` is the model for
# everything below and `docker run`s instead, correctly: it serves the *mock*
# case, where m1's container does not exist because m1's `deploy_kit` is sealed
# bytes naming a container nobody brought up. m4's real path has a live one.
#
# ## Why this file exists at all
#
# Measured 2026-09-04, standalone against real inputs: every m4 step that needs
# more than JSON needs the container, and none of them entered one.
#
#   * `KFO_PYTHON` defaults to `/opt/venv/bin/python3` — **a path inside the
#     image** — and STEPs 3 and 7 ran it on the host, dying `rc=127` with a bare
#     `not found`. (`_shlib.sh:kfo_python` now refuses loudly instead.)
#   * STEPs 4 and 5 need torch, and `spur exec <job> python3 -c "import torch"`
#     is `ModuleNotFoundError` — the node's *host* has no torch, only the
#     containers do. m3 measured that and answered it with their script.
#   * STEP 3's campaign needs the GPU and the ROCm stack.
#   * STEP 6 hashes the stock engine file, which is in the container tree.
#
# So "point `KFO_PYTHON` at a better interpreter" was never the fix; there is no
# such interpreter on any host in this cluster.
#
# ## What is NOT verified here
#
# Written and exercised on the login node, which has no docker daemon: argument
# handling, the record read, the ambient-vs-record refusal and every diagnostic
# below are measured. **The `docker exec` itself is not.** First real use is
# rung 4, with the leader watching — stated rather than glossed.
set -euo pipefail

WORKDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --workdir) WORKDIR="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: run_in_container.sh [--workdir DIR] <command ...>" >&2; exit 1; }
COMMAND="$*"

HERE=$(cd "$(dirname "$0")" && pwd)
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
# shellcheck source=/dev/null
. "$HERE/_shlib.sh"
PY=$(kfo_python) || exit 1

# **The record, from the `deploy_kit`, and never rebuilt here.** Same source
# `_lib.load_environment()` uses, so the container this execs into is the one
# every other m4 step is talking about. A script that re-derived the node or the
# container could differ from m1's and nothing would notice.
KIT="${AGENT_SYS_INPUT_DEPLOY_KIT:?this task declares deploy_kit as an input; the record lives in it}"
RECORD="$KIT/content/items/codes/environment.yaml"
[ -f "$RECORD" ] || RECORD="$KIT/items/codes/environment.yaml"
[ -f "$RECORD" ] || { echo "run_in_container: no environment.yaml under $KIT" >&2; exit 1; }

_field() {
  "$PY" - "$RECORD" "$1" <<'PY' 2>/dev/null || true
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
section, key = sys.argv[2].split(".")
print((d.get(section) or {}).get(key) or "")
PY
}

# **An ambient variable may not silently outvote the record.** m3's rule, and it
# is theirs because they paid for it: an inherited `E2E_JOBID` beat the record
# and the work ran on a node the artefact did not name, with nothing noticing.
# Absent is fine — the record fills it, which is the case for anything running
# without the agent's `E2E_*` block. A disagreement is refused, not resolved.
_agree_or_die() {  # name  ambient  from_record
  if [ -n "$2" ] && [ -n "$3" ] && [ "$2" != "$3" ]; then
    echo "run_in_container: $1 is '$2' in the environment and '$3' in the record." >&2
    echo "  Refusing rather than picking one: the record is what this stage's evidence" >&2
    echo "  will claim, and working somewhere else makes it about a different machine." >&2
    exit 1
  fi
  [ -n "$2" ] && printf '%s' "$2" || printf '%s' "$3"
}
E2E_NODE=$(_agree_or_die node "${E2E_NODE:-}" "$(_field fixed.node)")
E2E_JOBID=$(_agree_or_die slurm_jobid "${E2E_JOBID:-}" "$(_field runtime.slurm_jobid)")
E2E_TRANSPORT=$(_agree_or_die transport "${E2E_TRANSPORT:-}" "$(_field runtime.transport)")
export E2E_NODE E2E_JOBID E2E_TRANSPORT

CONTAINER=$(_field runtime.container)
[ -n "$CONTAINER" ] || {
  echo "run_in_container: the environment record names no runtime.container." >&2
  echo "  m1 records it when it brings the container up (CONTRACT section 5). A kit without" >&2
  echo "  one is either a replayed/sealed kit or a bring-up that did not finish." >&2
  exit 1
}

# **Never defaulted to 0.** Five owners share these nodes and cards 0-3 are a
# co-tenant's; a default here puts two runs on one card to blame each other's
# noise. CONTRACT section 5.2: every identifier bound on a shared host is a var.
: "${HIP_VISIBLE_DEVICES:=${E2E_MEASURE_GPU:-}}"
[ -n "$HIP_VISIBLE_DEVICES" ] || {
  echo "run_in_container: HIP_VISIBLE_DEVICES is empty and this host is shared." >&2
  echo "  Pass --var gpu=<n>. Cards 0-3 are another tenant's; 4-7 were free on 2026-09-04." >&2
  exit 1
}

# shellcheck source=/dev/null
. "$PKG/assets/lib/remote.sh"

# **Is the container actually running?** Checked before `docker exec`, because
# without it the failure is docker's own `No such container`, which reads as a
# docker problem rather than as "the record describes a bring-up that is gone".
# Mirrors m3's image-presence check and for the same reason: the record is the
# claim, and this checks the claim against the machine.
if ! on "docker inspect -f '{{.State.Running}}' '$CONTAINER' 2>/dev/null | grep -qx true"; then
  echo "run_in_container: the record names container" >&2
  echo "    $CONTAINER" >&2
  echo "  and it is not running on ${E2E_NODE:-the node}. Modules 1-4 share ONE container and" >&2
  echo "  m1 owns its lifetime; m4 does not start one (CONTRACT section 5)." >&2
  echo "  Containers that ARE running there:" >&2
  on "docker ps --format '    {{.Names}}   {{.Image}}'" >&2 || true
  echo "  Either m1's bring-up has not run for this record, or it has been torn down." >&2
  exit 1
fi

# `PYTHONDONTWRITEBYTECODE=1`: root-owned `__pycache__` inside a handoff is
# CONTRACT section 5.0 arriving early. `reclaim.sh` handles the rest, from
# inside this same container, which is the only context with the privilege.
EXEC_ENV="-e PYTHONDONTWRITEBYTECODE=1 -e HIP_VISIBLE_DEVICES='$HIP_VISIBLE_DEVICES'"
for name in KFO_SCRATCH_ROOT TRITON_CACHE_DIR KNOWLEDGE_LOCAL_ROOT TMPDIR \
            KFO_MAX_HOURS KFO_FORGE_MODEL KFO_SNR_THRESHOLD KFO_REPORT_FLAG KFO_IMPL_FLAG \
            AGENT_SYS_TASK_PACKAGE AGENT_SYS_DEMO_PACKAGE; do
  eval "value=\${$name:-}"
  [ -n "$value" ] && EXEC_ENV="$EXEC_ENV -e $name='$value'"
done
# Inside the image the venv is real, which is the whole point of entering it.
EXEC_ENV="$EXEC_ENV -e KFO_PYTHON='${KFO_PYTHON:-/opt/venv/bin/python3}'"

WORKDIR_ARG=""
[ -n "$WORKDIR" ] && WORKDIR_ARG="-w '$WORKDIR'"

echo "run_in_container: exec into $CONTAINER on ${E2E_NODE:-the node}, GPU $HIP_VISIBLE_DEVICES" >&2
echo "run_in_container: the next line comes from inside the container" >&2

rc=0
# shellcheck disable=SC2086
on "docker exec $EXEC_ENV $WORKDIR_ARG '$CONTAINER' bash -lc '$COMMAND'" || rc=$?

# CONTRACT section 5.0, in a `finally`: idempotent, and a no-op when there is
# nothing root-owned, so this does not decide first whether it will be needed.
if [ -n "${AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION:-}" ]; then
  sh "$PKG/assets/lib/reclaim.sh" "$CONTAINER" "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION" 2>/dev/null || true
fi
[ -n "${KFO_SCRATCH_ROOT:-}" ] && \
  sh "$PKG/assets/lib/reclaim.sh" "$CONTAINER" "$KFO_SCRATCH_ROOT" 2>/dev/null || true

exit "$rc"
