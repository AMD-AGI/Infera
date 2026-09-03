#!/usr/bin/env bash
# Run the workset's own entrypoints where the real path runs them: inside a
# container on a GPU node.
#
#     measure_in_container.sh <workset content dir> [command ...]
#
# With no command it runs both entrypoints and writes `evidence/`, which is what
# `build_workset`'s mock branch needs. With a command it runs that instead, in
# the same container from the same image — which is what `check_workset_runs`
# needs for its single-shape re-measurement. **One instrument, two callers**: a
# validator that re-measured through a different arrangement than the producer
# used would not be re-measuring the same thing.
#
# **Why this exists, and it is not "the mock needs a GPU".** The mock's own
# header has always said `evidence/` is a *measurement* and cannot be
# fabricated. What was wrong was *where* the measurement happened: the mock ran
# the entrypoints wherever the runner sat, and the plan said to run rung 0 from
# a host that has torch. Measured by the leader:
#
#     spur exec 106253 python3 -c "import torch"   ->  ModuleNotFoundError
#
# **The node's host environment has no torch. Only the containers do.** So there
# was no host anywhere in this cluster that satisfied the old wiring — the mock
# was not merely unsatisfied, it was unsatisfiable.
#
# The real path already measures in a container: `build_workset` is a `kind: ai`
# closure running inside the shared container of CONTRACT §5, so its STEP 7 and
# STEP 8 execute there. The 142.5 dB and 0.5–1.2% rsd in `run_correctness.sh`'s
# body came from `rocm/sgl-dev:v0.5.18-rocm720-mi35x`, not from a host.
#
# So this is m4's principle rather than a different machine: **the mock
# exercises the real wiring instead of a parallel one.** A mock that measured
# host-side would be testing an arrangement production never uses — and, as it
# turned out, one nothing can satisfy.
#
# **What it does not do: fabricate `evidence/`.** MOCK-MAP forbids it and it
# would gut `check_workset_runs`, whose entire job is that the numbers were
# measured on this hardware. The measurement still happens; it happens in the
# right place.
#
# ## The one honest difference from the real path
#
# The real path runs inside **m1's** container, shared by modules 1–4. In mock
# mode that container does not exist — m1's `deploy_kit` is sealed bytes and
# names a container nobody brought up. So this starts its own, from the same
# `environment.fixed.image`, and tears it down. Same image, same device, same
# entrypoints; a different container instance. Stated rather than glossed,
# because "the mock ran in a container" and "the mock ran in *the* container"
# are not the same claim and only the first is true here.
set -eu

CONTENT="${1:?usage: measure_in_container.sh <workset content dir> [command ...]}"
shift || true
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
# **The node facts come from the handoff when the caller has none.** A task body
# gets the agent's `E2E_*` block; a *validator* body gets the GLOBAL environment
# row and none of it. But the workset carries `environment.yaml`, which names
# the node, the job and the transport — so a validator reads them out of the
# artefact it is grading rather than needing them injected. That is the same
# rule as everything else here: the record travels with the handoff.
_from_record() {
  "${AGENT_SYS_DEMO_PYTHON:-python3}" - "$1" "$2" <<'PY' 2>/dev/null || true
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
section, key = sys.argv[2].split(".")
print((d.get(section) or {}).get(key) or "")
PY
}

# shellcheck source=/dev/null
. "$PKG/assets/lib/remote.sh"

ROOT="$CONTENT/items/codes"
[ -d "$ROOT" ] || { echo "measure_in_container: no $ROOT" >&2; exit 1; }

# The image is the one the environment record names, not one this script picks.
# A mock that measured under a different image than the record claims would be
# evidence about a machine the handoff does not describe.
IMAGE=$("${AGENT_SYS_DEMO_PYTHON:-python3}" - "$ROOT/environment.yaml" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
print((d.get("fixed") or {}).get("image") or "")
PY
)
[ -n "$IMAGE" ] || { echo "measure_in_container: environment.yaml names no fixed.image" >&2; exit 1; }

: "${E2E_NODE:=$(_from_record "$ROOT/environment.yaml" fixed.node)}"
: "${E2E_JOBID:=$(_from_record "$ROOT/environment.yaml" runtime.slurm_jobid)}"
: "${E2E_TRANSPORT:=$(_from_record "$ROOT/environment.yaml" runtime.transport)}"
export E2E_NODE E2E_JOBID E2E_TRANSPORT

# **Every identifier bound on a shared host is a parameter** (CONTRACT §5.2).
# The card especially: five owners share two nodes and m1 holds GPU 0.
: "${E2E_MEASURE_GPU:=4}"
: "${E2E_MEASURE_CONTAINER:=yihou_m3_measure_$$}"

# The workset must be visible from the node. On this cluster "remote" *is*
# `/shared_nfs`, so a run root anywhere else cannot be measured — say so here
# rather than let `docker run` fail with a confusing empty mount.
case "$ROOT" in
  /shared_nfs/*) ;;
  *) echo "measure_in_container: $ROOT is not under /shared_nfs, so the node cannot see it." >&2
     echo "  On this cluster the shared filesystem is the transport; point the run root there." >&2
     exit 1 ;;
esac

# The default is the producer's pair; a caller may substitute its own.
if [ "$#" -eq 0 ]; then
  COMMAND="./run_correctness.sh --json evidence/correctness.json && \
           ./run_performance.sh --json evidence/performance.json"
else
  COMMAND="$*"
fi

echo "measure_in_container: $IMAGE on GPU $E2E_MEASURE_GPU as $E2E_MEASURE_CONTAINER"

# `--rm` so nothing is left behind, and a name nothing else owns: **never
# `docker rm -f` a name you did not create**, and both held nodes carry other
# tenants' containers. `PYTHONDONTWRITEBYTECODE=1` keeps root-owned `.pyc` out
# of the handoff; `reclaim.sh` (CONTRACT §5.0) handles what is still root-owned
# after, because the container writes `evidence/` as root by construction.
on "docker run --rm --name '$E2E_MEASURE_CONTAINER' \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
      -e HIP_VISIBLE_DEVICES='$E2E_MEASURE_GPU' \
      -e E2E_NODE='${E2E_NODE:-}' -e PYTHONDONTWRITEBYTECODE=1 \
      -v /shared_nfs:/shared_nfs -w '$ROOT' '$IMAGE' \
      bash -c '$COMMAND'"

sh "$PKG/assets/lib/reclaim.sh" "$CONTENT" 2>/dev/null || true
echo "measure_in_container: evidence written by the same entrypoints the real path runs"
