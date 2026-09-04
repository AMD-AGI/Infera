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

# **The workset must be visible from the node, and that is asked rather than
# asserted.** This was `case "$ROOT" in /shared_nfs/*)`, which encoded a site
# fact as a literal path: correct where it was written, and silently wrong one
# mount change later. `/shared_nfs` went read-only on the login node, the run
# root moved to `/home/yihou/agent_sys_runroot`, and rung 1's `build_workset`
# refused after eleven seconds — a workset that would have measured fine, since
# `/home` is NFS from the same server and the compute nodes mount it too
# (measured on node 243: the run root is there and writable).
#
# The property wanted was never "under /shared_nfs"; it is "a path the node can
# also see". A second literal for `/home` would be the same defect with a
# longer list, so this asks the node.
#
# **And the helper already existed.** `remote.sh:require_visible_on_node` has
# done exactly this since before I wrote the literal, with six callers across
# four other owners' scripts and a header comment that already says the run root
# being NFS is "a fact about this cluster, not about agent_sys, so it is checked
# rather than assumed". CONTRACT §4.1: shared things are shared, not re-solved.
#
# **Visibility is probed; what may be *mounted* is a measured list**, and the
# block below says why the two are not the same question. Being able to see a
# path and being allowed to bind-mount it are separate permissions here.
require_visible_on_node "$ROOT" "workset" || exit 1

# **What the container mounts follows from where the workset is**, and the
# daemon has an opinion about it. `-v /shared_nfs:/shared_nfs` alone would pass
# the check above and then hand `docker run` an empty mount — the confusing
# failure the old guard existed to prevent, reintroduced one directory over.
#
# **The obvious derivation is refused by this cluster.** Taking the run root's
# top-level component gives `/home` for `/home/<user>/agent_sys_runroot`, and
# the leader measured the daemon's answer on node 243:
#
#     Error response from daemon: authorization denied by plugin spur-authz:
#     denied [BH]: /home:/home -- mount your own directory instead,
#     e.g. -v $HOME:$HOME (or -v /home/<you>:/home/<you>)
#
# So the plugin's rule is about *whose* directory, not about depth: a shared
# top-level directory is refused and `/shared_nfs` is allowed. That is not
# derivable from the path, and the previous attempt here derived it anyway —
# the same string-surgery-instead-of-asking mistake as the guard above, made
# twice in one file.
#
# **These two forms are what was measured working, and nothing else is
# guessed.** Anything outside them refuses here, naming both, rather than
# arriving as an authorization denial in the middle of a measurement.
REMOTE_HOME="${E2E_REMOTE_HOME:-$HOME}"
case "$ROOT/" in
  "$REMOTE_HOME"/*)  MOUNTS="-v $REMOTE_HOME:$REMOTE_HOME" ;;
  /shared_nfs/*)     MOUNTS="-v /shared_nfs:/shared_nfs" ;;
  *) echo "measure_in_container: $ROOT is on neither filesystem this cluster's docker" >&2
     echo "  authorization plugin allows a measurement to mount. Measured on node 243:" >&2
     echo "    -v $REMOTE_HOME:$REMOTE_HOME   (a run root under your home)   OK" >&2
     echo "    -v /shared_nfs:/shared_nfs                                    OK" >&2
     echo "    -v /home:/home            denied [BH] by plugin spur-authz" >&2
     echo "  Point --demo-root at one of the two, or extend this case with a form you" >&2
     echo "  have seen the daemon accept — not one you expect it to." >&2
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

# **`--rm` is not a teardown, and `41c8540` is why.** A cancelled Slurm hold does
# not reclaim its GPUs: containers talk to the *host* daemon, so they outlive the
# job. Job 109192 was cancelled 28 minutes into 8 hours and fifteen minutes later
# all four containers were still `Up` and serving. `--rm` covers the case where
# docker exits on its own; it covers nothing if this script is killed, if the
# `spur exec` is interrupted, or if the runner's stall detector ends the task
# while timing is in progress — and holds were cancelled four times today, at
# 5 h, 1 h 21 m and 28 minutes. An abandoned measurement container then holds a
# card for the rest of someone else's reservation.
#
# So teardown is a trap and not a following line, and the flag is what keeps the
# standing rule intact: **never `docker rm -f` a container you did not create**.
# `STARTED` is set immediately before the `run` and nowhere else, so an operator
# who points `E2E_MEASURE_CONTAINER` at a name that already exists cannot have it
# removed by this script — it only ever tears down a container it just started.
#
# `reclaim.sh` (CONTRACT §5.0) moves into the trap for the same reason: the
# container writes `evidence/` as root by construction, and a run that dies
# half-way leaves the *most* root-owned files, which is exactly when the old
# placement skipped it.
STARTED=0
_teardown() {
  # Disarm first: a signal fires the handler *and then* EXIT, so without this
  # the teardown runs twice. Both halves are idempotent, so the duplicate was
  # harmless — but a `docker rm -f` printed twice in a transcript reads as a
  # retry, and a retry reads as something having gone wrong.
  trap - EXIT HUP INT TERM
  [ "$STARTED" = 1 ] && on "docker rm -f '$E2E_MEASURE_CONTAINER' >/dev/null 2>&1 || true" || true
  sh "$PKG/assets/lib/reclaim.sh" "$CONTENT" 2>/dev/null || true
}
trap _teardown EXIT HUP INT TERM

# `PYTHONDONTWRITEBYTECODE=1` keeps root-owned `.pyc` out of the handoff.
STARTED=1
on "docker run --rm --name '$E2E_MEASURE_CONTAINER' \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
      -e HIP_VISIBLE_DEVICES='$E2E_MEASURE_GPU' \
      -e E2E_NODE='${E2E_NODE:-}' -e PYTHONDONTWRITEBYTECODE=1 \
      $MOUNTS -w '$ROOT' '$IMAGE' \
      bash -c '$COMMAND'"

echo "measure_in_container: evidence written by the same entrypoints the real path runs"
