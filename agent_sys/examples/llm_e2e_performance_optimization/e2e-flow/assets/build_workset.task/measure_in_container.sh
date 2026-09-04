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
# gets the agent's `E2E_*` block — *the agent's*, which is the whole point and
# was not true when this line was written: `workset_builder` declared no `env`,
# so this script ran with none of it and was correct anyway, by reading the
# record. A *validator* body gets the GLOBAL environment row and none of it
# either. The workset carries `environment.yaml`, which names
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

# **An ambient variable may not silently outvote the record.**
#
# This was `: "${E2E_JOBID:=<from record>}"`, so an inherited `E2E_JOBID` won
# and the measurement could happen on a node the record does not name — with
# nothing noticing, because the evidence records where it *ran* and the premise
# records where it *claims*. Found by simulating the post-fix graph: the record
# said node 061 / job 106250, my producer run carried E2E_JOBID=106253, and it
# measured on 031 while the artefact said 061.
#
# That is the whole failure this script exists to prevent, one level up from
# the image: evidence about a machine the handoff does not describe. So a
# disagreement is refused rather than resolved. Absent is fine — the record
# fills it, which is the validator's case, since a validator body gets none of
# the E2E_* block.
_agree_or_die() {  # name  ambient  from_record
  if [ -n "$2" ] && [ -n "$3" ] && [ "$2" != "$3" ]; then
    echo "measure_in_container: $1 is '$2' in the environment and '$3' in the record." >&2
    echo "  Refusing rather than picking one: the record is what the evidence will claim," >&2
    echo "  and measuring somewhere else makes it evidence about a different machine." >&2
    exit 1
  fi
  [ -n "$2" ] && printf '%s' "$2" || printf '%s' "$3"
}
E2E_NODE=$(_agree_or_die node "${E2E_NODE:-}" "$(_from_record "$ROOT/environment.yaml" fixed.node)")
E2E_JOBID=$(_agree_or_die slurm_jobid "${E2E_JOBID:-}" "$(_from_record "$ROOT/environment.yaml" runtime.slurm_jobid)")
E2E_TRANSPORT=$(_agree_or_die transport "${E2E_TRANSPORT:-}" "$(_from_record "$ROOT/environment.yaml" runtime.transport)")
export E2E_NODE E2E_JOBID E2E_TRANSPORT

# **Every identifier bound on a shared host is a parameter** (CONTRACT §5.2).
# The card especially: five owners share the held nodes, and which cards are
# free changes between rungs — this line used to name GPU 0 as m1's, and by the
# time it was read the held set was 0-3 with 4-7 free. A site fact asserted in a
# comment is a site fact nothing validates (m1's T19). `rocm-smi` immediately
# before use is the only reading that is current.
#
# The default is 4 and both `E2E_MEASURE_GPU` and `E2E_MEASURE_CONTAINER` are
# now declared on `workset_builder`, so the `:=` below is the *body's* fallback
# for a standalone invocation rather than the only value the graph can produce.
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
  # **Progress between the two, because the stretch is otherwise silent.**
  # A container start, a torch import, three shapes of correctness and then
  # three shapes x five groups x ten iterations of timing is minutes with
  # nothing on stdout. A watchdog that keys on silence cannot tell that from a
  # wedge, and neither can a person reading the log — the first thing anyone
  # asks about a stalled stage is whether it is doing anything.
  COMMAND="echo '  [1/2] correctness' && ./run_correctness.sh --json evidence/correctness.json && \
           echo '  [2/2] performance' && ./run_performance.sh --json evidence/performance.json"
else
  COMMAND="$*"
fi

# **Is that image actually on the node?** Checked before `docker run`, because
# the two failure modes without this are both bad: docker tries to *pull* a
# 90 GB image it may not have credentials for, or fails with a message about a
# manifest that sends the reader to a registry rather than to the record.
#
# This is a live seam, not a hypothetical. The sealed `deploy_kit` names
# `infera/engine-sglang:gfx950-local`, which is on **neither** held node — `031`
# carries `rocm/sgl-dev:*`, `infera/engine-vllm:test-local*` and m1's newer
# build; `061` carries no sglang or infera image at all. So a mock run that
# inherits the sealed record will land here, and it should land here saying
# exactly that rather than somewhere less obvious.
#
# The image is still taken from the record and **not** guessed or overridden.
# The record is the claim; this checks the claim against the machine and reports
# the disagreement. Substituting a different image would make the evidence be
# about something the handoff does not describe.
if ! on "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
  echo "measure_in_container: the environment record names image" >&2
  echo "    $IMAGE" >&2
  echo "  and it is not present on ${E2E_NODE:-the node}. Not pulling: it would be tens of GB" >&2
  echo "  and the record, not this script, is what has to be right." >&2
  echo "  Images that ARE on that node:" >&2
  on "docker images --format '    {{.Repository}}:{{.Tag}}'" >&2 || true
  echo "  Either the kit's record should name one of those, or that image should be" >&2
  echo "  built/pulled onto the node before this stage runs." >&2
  exit 1
fi

echo "measure_in_container: $IMAGE on GPU $E2E_MEASURE_GPU as $E2E_MEASURE_CONTAINER"
echo "measure_in_container: starting the container; the next line comes from inside it"

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
