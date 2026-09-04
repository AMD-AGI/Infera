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
# **No default, and this is T19 closed on my side.** It was `:=4`, and card 4
# is a real card on every node this package has touched. m4 refused to declare
# one for the same reason and was right: *a card as a package default makes a
# consumer's refuse-when-empty guard unreachable*, and mine was the consumer
# that defaulted where theirs refused.
#
# The measured cost of defaulting: 2026-09-04, all eight cards on 006 were at
# 75% under a bring-up that declared no `HIP_VISIBLE_DEVICES`, and this stage
# would have measured on card 4 regardless. **The bad outcome is the quiet
# one** — not an OOM, which is loud, but timings contaminated by a co-tenant.
# `check_workset_runs` re-measures on the same card, gets the same
# contamination, and *agrees*: two honest measurements, both wrong, gate green.
# Nothing in the chain looks at neighbours, so a default here is the one knob
# that can produce a confidently wrong number.
#
# Refusing costs a `--var` and buys the property that a measurement never
# happens on a card nobody chose.
if [ -z "${E2E_MEASURE_GPU:-}" ]; then
  echo "measure_in_container: no measurement card, and this is not defaulted on" >&2
  echo "  purpose. Five owners share these nodes, and a card someone else is serving" >&2
  echo "  from does not fail — it returns slower numbers that check_workset_runs" >&2
  echo "  re-measures on the same card and agrees with. Cards on this node now:" >&2
  on "rocm-smi --showmemuse 2>/dev/null | grep 'VRAM%'" >&2 2>/dev/null \
    || echo "    (could not read rocm-smi on ${E2E_NODE:-the node})" >&2
  # **The action goes last, because the consumer keeps the tail.**
  # `check_workset_runs` reports `stderr.splitlines()[-N:]`, so a message whose
  # instruction is at the top arrives with the instruction cut off. Measured on
  # this very message.
  echo "  FIX: pass --var measure_gpu=<n> to the run, or set E2E_MEASURE_GPU=<n>." >&2
  exit 1
fi
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
# **Establish the transport before asking it a question**, because
# `require_visible_on_node` cannot tell the two apart.
#
# Measured 2026-09-04: with a deliberately bad `E2E_TRANSPORT`, that helper
# reports `the workset is not visible on : /home/yihou` — for a path that
# plainly is. It runs `on "test -e …" >/dev/null 2>&1` and treats **any**
# non-zero as absence, so an unset `E2E_JOBID`, a dead allocation, a spur error
# and a genuinely missing path all produce the same sentence. That is the day's
# own pattern one level in: **a plausible explanation attached to a failure it
# did not produce**, and it is worse than no message, because it sends the
# reader to fix a run root that is already correct. It sent the leader there.
#
# The helper is `assets/lib/remote.sh`, shared with six callers and not mine to
# change. What is mine is not relaying its guess: a reachability probe first,
# with the transport's **actual** stderr, so "the node is unreachable" and "the
# node cannot see this path" are two different messages again.
if ! _probe=$(on "true" 2>&1); then
  echo "measure_in_container: cannot reach the node at all, so nothing below was tested." >&2
  echo "  transport='${E2E_TRANSPORT:-}' jobid='${E2E_JOBID:-}' node='${E2E_NODE:-}'" >&2
  echo "  (all three come from the record when the caller has none; a validator body" >&2
  echo "   gets no E2E_* block, which is the usual reason they are empty here)" >&2
  echo "  the transport said:" >&2
  printf '    %s\n' "${_probe:-<no output>}" >&2
  exit 1
fi
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
# **Derived from `$ROOT`, never from `$HOME`.** This read
# `${E2E_REMOTE_HOME:-$HOME}` and the mount came out `-v /home:/home`, the one
# form the plugin explicitly refuses — and the refusal quoted my own instruction
# to *extend this case with a form you have seen the daemon accept*.
#
# **This comment used to say `$HOME` is `/home` in a validation zone. That was a
# fitted parameter, not a measurement**, and it is corrected rather than deleted
# because the correction is the useful part. I never echoed `$HOME` anywhere; I
# picked the value that would explain the mount string I had. `HOME=/home` does
# explain it — and `validator/environment.py:235` sets a validator's `HOME` to
# `<zone>/home`, under which this code would have taken the *refusal* branch
# instead. **So the two do not reconcile and the real value is still unknown**:
# either that denial came from a task body rather than the validator, or the
# harness config puts `HOME` back over the zone's. `todo.md` T42 carries the
# open question.
#
# None of which changes the fix, and that is the point: the derivation below
# reads `$ROOT` and no ambient value, so it is correct under every candidate
# answer. **A fix that does not depend on the disputed fact is worth more than
# winning the dispute.**
#
# **I built a bound identifier out of an ambient value**, which is the mistake
# this file already carries two other corrections for: `_agree_or_die` exists
# because an ambient `E2E_JOBID` outvoted the record, and the guard above exists
# because a literal path outlived its mount. `$HOME` is the same class — it
# describes whoever is running, not the artefact, and a validator runs as
# nobody in particular.
#
# `$ROOT` is a fact about the workset and is right in every caller. `/home/<user>`
# from it is the form measured accepted on 006 (`b9849a7`, `torch 2.9.1` back
# from inside the container) and again on 234 and 249.
#
# `E2E_REMOTE_HOME` is still honoured when **explicitly set** — an operator
# naming a home is stating a fact; `$HOME` defaulting to `/home` was not.
case "$ROOT" in
  /shared_nfs/*) MOUNT_AT="/shared_nfs" ;;
  /home/*/*)     MOUNT_AT="/home/$(printf '%s' "${ROOT#/home/}" | cut -d/ -f1)" ;;
  *)
    if [ -n "${E2E_REMOTE_HOME:-}" ] && [ "${ROOT#"${E2E_REMOTE_HOME}"/}" != "$ROOT" ]; then
      MOUNT_AT="$E2E_REMOTE_HOME"
    else
      echo "measure_in_container: cannot derive a mount this cluster's docker authorization" >&2
      echo "  plugin will accept from $ROOT. Measured forms:" >&2
      echo "    -v /home/<user>:/home/<user>   OK   (a run root under one user's home)" >&2
      echo "    -v /shared_nfs:/shared_nfs     OK" >&2
      echo "    -v /home:/home                 denied [BH] by plugin spur-authz" >&2
      echo "  Point --demo-root at one of the two, or set E2E_REMOTE_HOME, or extend this" >&2
      echo "  case with a form you have SEEN the daemon accept — not one you expect it to." >&2
      exit 1
    fi ;;
esac
MOUNTS="-v $MOUNT_AT:$MOUNT_AT"

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
# **The reclaim was wrong three ways and `2>/dev/null` hid all three.** It read
# `sh "$PKG/assets/lib/reclaim.sh" "$CONTENT" 2>/dev/null || true`, and
# `reclaim.sh`'s contract is `<container-name> <path-inside-container> …`:
#
# 1. `$CONTENT` is a host path, passed where a container name goes, so
#    `docker inspect` missed and it exited 0 with "nothing to do";
# 2. it ran **locally**, on a login node with no reach to the node's daemon;
# 3. it ran **after** `docker rm -f`, and `reclaim.sh` works by `docker exec`
#    into a *running* container — with `--rm` there is no window at all.
#
# Found by trying to edit `evidence/performance.json` and getting `Permission
# denied`: `-rw-r--r-- root root`. Every workset this stage has produced has
# root-owned evidence, which is the failure `reclaim.sh`'s own header predicts —
# 644 so every reader and the seal work fine, and the zone's user cannot clean
# up, so the symptom lands on a later run.
#
# Two defences, because they cover different failures:
#
# * **The payload chowns as its last act** (below). The container is root and
#   owns the files, so this needs no `docker exec` and no lifecycle window. It
#   covers the normal path, where `--rm` removes the container the instant the
#   command returns.
# * **The trap reclaims *before* removing.** If the run is killed mid-measure
#   the payload's chown never happens, and this is the only remaining chance —
#   the container is still up at that point, which is exactly what `reclaim.sh`
#   needs. Ordering matters and the old code had it backwards.
STARTED=0
_teardown() {
  # Disarm first: a signal fires the handler *and then* EXIT, so without this
  # the teardown runs twice. Both halves are idempotent, so the duplicate was
  # harmless — but a `docker rm -f` printed twice in a transcript reads as a
  # retry, and a retry reads as something having gone wrong.
  trap - EXIT HUP INT TERM
  [ "$STARTED" = 1 ] || return 0
  # On the node, not here, and while the container is still alive. Errors are
  # shown rather than swallowed: a reclaim that cannot run is the thing that
  # went unnoticed, and "nothing to do" printed once is cheaper than a tree
  # nobody can delete.
  on "sh '$PKG/assets/lib/reclaim.sh' '$E2E_MEASURE_CONTAINER' '$ROOT'" || true
  on "docker rm -f '$E2E_MEASURE_CONTAINER' >/dev/null 2>&1 || true" || true
}
trap _teardown EXIT HUP INT TERM

# **The payload travels base64, because `bash -c '$COMMAND'` silently ate it.**
#
# Measured on node 006, 2026-09-04, the first time this `docker run` ever
# executed: a payload containing a single quote produced **no output, exit 0,
# and the success line below**. The default payload contains four —
# `echo '  [1/2] correctness'` and its pair — so the real measurement path was
# the broken one. The command crosses `spur exec bash -lc`, then this string,
# then `bash -c '...'`; the first `'` in the payload closed the quote the
# wrapper opened, and the remainder was re-parsed by the node's shell instead
# of the container's.
#
# The differential that settled it, same script and same everything else: a
# payload with single quotes printed nothing, and the identical payload written
# without them printed `2.9.1+rocm7.2.0.git7e1940d4` from inside the container.
#
# Base64 has no shell metacharacters, so it survives all three layers with one
# pair of double quotes and no escaping to get right. `check_workset_runs` also
# drives this script with its own `--shape` command, so the custom-command path
# is not a convenience — a validator's payload was subject to the same defect.
# **And the payload hands the files back before it exits.** Same reason
# `reclaim.sh` exists (CONTRACT §5.0) and the same chown it performs, done from
# inside rather than through `docker exec` — the container is root, owns what it
# just wrote, and is the only context with the privilege. `id -u`/`id -g` here
# because that is the identity the zone belongs to; `spur exec` measured as the
# same uid, so the node agrees with this host.
#
# `rc=$?` around it so the measurement's exit status is what leaves the
# container: a reclaim that swallowed a failing measurement would turn this into
# the silent-success class twice over.
# **Spaces around the pipes, and they are load-bearing.** Measured 2026-09-04
# across two nodes: `echo <b64>|base64 -d|bash` returns 255 with no output on
# crsuse2-m2m-047 and works on 006; the identical command with spaces works on
# both. Same `spur exec bash -lc`, same image, no docker in the minimal case —
# so it is a property of that node's shell path, not of this package. Why is
# unknown; the spaced form is the one *seen* to work on both, which is the
# standard this file already applies to the mount list.
CMD_B64=$(printf '%s' "{ $COMMAND ; } ; rc=\$? ; chown -R $(id -u):$(id -g) '$ROOT' 2>/dev/null || true ; exit \$rc" \
          | base64 | tr -d '\n')

# `PYTHONDONTWRITEBYTECODE=1` keeps root-owned `.pyc` out of the handoff.
#
# **`TMPDIR` and `TRITON_CACHE_DIR` are m4's guard, adopted after measuring
# that I do not currently need it.** Their reason: Triton defaults to
# `$HOME/.triton`, and on a host where `$HOME` is an NFS mount a container
# user's writes there fail *silently*; and a `TMPDIR` naming a directory that
# does not exist segfaults every HIP launch while `torch.cuda.is_available()`
# still returns True.
#
# Measured inside this image: `HOME=/root`, on the container's own overlay,
# writable — so neither reaches NFS and the failure cannot occur here. **Set
# anyway, because that is a property of this image and not of the contract.**
# I mount `/home/<user>`, so an image whose `HOME` sat under that mount would
# put the Triton cache on NFS and I would find out as unexplained compile time
# inside warmup, which is invisible. Two flags cost nothing and remove the
# dependency on an image's environment.
#
# `/tmp` and `/tmp/triton` both resolve inside the container. Naming a path
# that does not exist is the segfault above, so these are deliberately not
# parameterised to anything host-side.
STARTED=1
on "docker run --rm --name '$E2E_MEASURE_CONTAINER' \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
      -e HIP_VISIBLE_DEVICES='$E2E_MEASURE_GPU' \
      -e E2E_NODE='${E2E_NODE:-}' -e PYTHONDONTWRITEBYTECODE=1 \
      -e TMPDIR=/tmp -e TRITON_CACHE_DIR=/tmp/triton \
      $MOUNTS -w '$ROOT' '$IMAGE' \
      bash -c \"echo $CMD_B64 | base64 -d | bash\""

echo "measure_in_container: evidence written by the same entrypoints the real path runs"
