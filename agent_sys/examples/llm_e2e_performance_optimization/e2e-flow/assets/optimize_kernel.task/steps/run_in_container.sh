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
# ## Verified end to end, 2026-09-04
#
# Argument handling, the record read, the ambient-vs-record refusal and every
# diagnostic below were measured on the login node. **The `docker exec` was
# closed on the node**, into m1's `yihou_e2e_sgl_m1real-20260904` on
# `crsuse2-m2m-249`, card 4:
#
#     torch 2.11.0+rocm7.2
#     6b727fcde1724924c71c1148d89005500195527e827fe7ec8d51eef43d92a762  …/srt/layers/sampler.py
#     EXIT=0
#
# Both match m1's independent probes. m1 named that container after the record
# on purpose, so the `runtime.container` lookup, `_agree_or_die` and the
# running-state probe all ran rather than being skipped.
#
# The hash is the one `apply.py`'s gate expects from that image, so the
# resolve -> hash path m5's stale `base_sha256` came from is closed too. What is
# still unexercised is the *campaign* — this wrapper has never carried a
# multi-hour command.
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

# **Never defaulted.** Five owners share these nodes and cards 0-3 have been a
# co-tenant's; a default here puts two runs on one card to blame each other's
# noise. CONTRACT section 5.2: every identifier bound on a shared host is a var.
#
# **One name for the card, and it is `HIP_VISIBLE_DEVICES` (`--var gpu=`).**
# This used to fall back to m3's `measure_gpu`. Two names for one identifier is
# CONTRACT section 4.3's shape, and the fallback also made the refusal below
# unreachable, so it went.
#
# **The reason I gave at the time has since expired, and the decision has not.**
# It was: `shared.yaml` declares `${measure_gpu:-4}`, *a real card*, so importing
# it picks card 4 silently on a shared host. As of `7c2d501` that default is gone
# — `E2E_MEASURE_GPU: '${measure_gpu:-}'`, empty — so the silent-card-4 argument
# no longer holds. **The one-name argument does**, and it is the one that was
# load-bearing. Corrected rather than left standing: a justification that has
# outlived its premise reads as a live reason and is how the next person
# re-litigates a settled decision from a false start.
: "${HIP_VISIBLE_DEVICES:=}"
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
#
# **A sentinel in stdout, not the exit status, and this file had the bug the
# sentinel exists to prevent.** `remote.sh:require_visible_on_node` was rewritten
# earlier the same day because `on "test -e …"` read any non-zero as absence, so
# a dead allocation and a missing path produced one sentence — and that sentence
# blamed the wrong thing and cost a rung. This check was written after that fix
# and repeated its shape: `on "docker inspect … | grep -qx true"` is non-zero
# both when the node answers "no such container" and when the node is never
# reached, and the message below then tells the reader m1 tore their container
# down. Reproduced 2026-09-04 against m1's real record for
# `yihou_e2e_flow_sgl_e2e-main-20260904` on `crsuse2-m2m-217`, twice, once with
# `SPUR_CONTROLLER_ADDR` present and once with `env -i` stripping it: **byte
# for byte the same six lines**, though only the first had tested anything. The
# stripped case is not exotic — `remote.sh:92-97` records that the closed
# validator environment is exactly where that variable is absent.
#
# Only the far side can print `E2E_STATE=`, so its presence is what separates
# "the node answered" from "nothing was asked".
CSTATE="$(on "docker inspect -f 'E2E_STATE={{.State.Running}} E2E_ID={{.Id}} E2E_CREATED={{.Created}} E2E_STARTED={{.State.StartedAt}} E2E_RESTARTS={{.RestartCount}} E2E_ENV={{json .Config.Env}}' '$CONTAINER' 2>/dev/null || echo E2E_STATE=absent" 2>/dev/null)" || true
RECORD_CONTAINER_UP=1
case "$CSTATE" in
  *E2E_STATE=true*) ;;
  *E2E_STATE=*)
    RECORD_CONTAINER_UP=0
    ;;
  *)
    echo "run_in_container: could not reach ${E2E_NODE:-the node}, so NOTHING was learned" >&2
    echo "  about container '$CONTAINER'. This is not evidence that m1's bring-up is gone." >&2
    echo "  transport='${E2E_TRANSPORT:-}' jobid='${E2E_JOBID:-}' node='${E2E_NODE:-}'" >&2
    if [ -n "${SPUR_CONTROLLER_ADDR:-}" ]; then _addr="set"; else _addr="UNSET -- the usual cause"; fi
    echo "  SPUR_CONTROLLER_ADDR is $_addr; a closed" >&2
    echo "  environment does not carry it (remote.sh:92-97)." >&2
    exit 2
    ;;
esac

# **When the record's container is not up, measure in an ephemeral one of our
# own — and say so in the artefact.**
#
# Until 2026-09-04 this refused, on CONTRACT §5: *m1 owns the container's
# lifetime; m4 execs into it.* Obeying that exactly is what stopped rung 0 —
# **in a mock chain nobody brings the deployment up**, so the container the
# record names has never existed, and a check that can only run after a real
# deployment cannot be part of the mock e2e, which is the deliverable. The
# leader ruled §5 conflates two things and is amending it: *the deployment*,
# whose lifetime is m1's, and *the measurement apparatus*, which belongs to
# whoever measures and is gone when they finish.
#
# m3 hit this first and their shape is copied rather than re-derived
# (`build_workset.task/measure_in_container.sh:156`): self-named, `--rm`,
# started from the image the record names, torn down in a trap, and **never a
# name we did not create** — `STARTED` gates the teardown so a caller who points
# `KFO_MEASURE_CONTAINER` at an existing name cannot have it removed by us.
#
# **The two are not interchangeable and the report must not let a reader guess.**
# A speedup measured inside the live deployment and one measured in a fresh
# container off the same image are different claims — the first carries the
# engine's actual state, the second only the image's. Which was used is printed
# and written into `observed_runtime.json` as `mode`.
MODE=record
if [ "$RECORD_CONTAINER_UP" = 0 ]; then
  MODE=ephemeral
  IMAGE="$(_field fixed.image)"
  [ -n "$IMAGE" ] || {
    echo "run_in_container: the record's container '$CONTAINER' is not running on" >&2
    echo "  ${E2E_NODE:-the node}, and the record names no fixed.image to fall back to." >&2
    echo "  Containers that ARE running there:" >&2
    PS_OUT="$(on "docker ps --format '{{.Names}}   {{.Image}}'" 2>/dev/null)" || PS_OUT=""
    # **Print `none` rather than nothing.** An empty line here is
    # indistinguishable from a listing that failed, and the sentence that
    # follows is a conclusion the reader draws from *seeing* what is there.
    # Same class as `f103fe0`, where a redirection order emptied this list.
    if [ -n "$PS_OUT" ]; then printf '%s\n' "$PS_OUT" | sed 's/^/    /' >&2
    else echo "    none (the node answered; there are no containers running)" >&2; fi
    exit 1
  }
  : "${KFO_MEASURE_CONTAINER:=yihou_m4_measure_$$}"
  echo "run_in_container: the record's container '$CONTAINER' is not running." >&2
  echo "  Measuring in an ephemeral container of my own, '$KFO_MEASURE_CONTAINER'," >&2
  echo "  from the image the record names: $IMAGE" >&2
  echo "  This carries the IMAGE's state, not the deployment's. The handoff records" >&2
  echo "  mode=ephemeral so no reader has to infer which of the two produced the number." >&2
  CONTAINER="$KFO_MEASURE_CONTAINER"
fi

# **What the record claims and what is on the node are two different facts, and
# the join between them was unchecked.** m1 found the case on 2026-09-04: their
# record said `runtime.started_at: 2026-09-04T09:03:51Z` while docker reported
# `Created == StartedAt == 09:37:18Z` with `RestartCount: 0` — a restart keeps
# its creation time and bumps the counter, so this was a *different container
# wearing the same name*. `_agree_or_die` above compares node, jobid and
# transport and they were all still true; the container is looked up by name and
# the name resolved. Every field validated and the join was still wrong.
#
# For the exec it does not bite — the name reached a live container of the right
# image on the right node. For provenance it does, and this stage's handoff
# carries `premise.run_environment`, which `10_read_inputs.py:137` fills with
# `lib.load_environment()`: m1's record, verbatim, no observation. So the one
# line below is the only place in m4 where the container that actually ran the
# work identifies itself. Printed rather than folded into the handoff, because
# `run_environment`'s reader is the premise gate and changing what that field
# means is a contract decision, not a wrapper's.
# Only in `record` mode is there anything to compare: `CSTATE` describes the
# container the record names, and in `ephemeral` mode that container does not
# exist, so every field below would be empty. **An empty observation printed
# beside a claim reads as an observation that matched**, which is the one thing
# `observed_runtime` exists to prevent.
#
# Written as `if` rather than `[ … ] && echo` for readability only. **I first
# claimed that shape was what aborted the ephemeral run; it was not** — bash
# does not apply `set -e` to a non-final command in an `&&` list, and two
# pre-existing lines in this file rely on that. The abort was `grep` exiting 1
# under `pipefail`, twenty lines down. Left corrected rather than deleted,
# because a plausible wrong cause recorded as fact is how the next reader
# "fixes" the wrong line.
if [ "$MODE" = record ]; then
  echo "run_in_container: observed $(printf '%s' "$CSTATE" | tr ' ' '\n' | grep -E '^E2E_(ID|CREATED|STARTED|RESTARTS)=' | tr '\n' ' ')" >&2
fi
echo "run_in_container: record claims started_at=$(_field runtime.started_at)" >&2

# **The card this exec asks for must be one the container was actually given.**
#
# Measured 2026-09-04 in m1's own kit, `start_container.sh:37-44`: the container
# is started with `--device /dev/kfd --device /dev/dri`, which exposes **every**
# card on the host, and is pinned only by `--env
# HIP_VISIBLE_DEVICES=${E2E_KIT_GPU_DEVICES}`. So the pin is an environment
# variable, not a device whitelist — and `docker exec -e HIP_VISIBLE_DEVICES=4`
# **overrides it**. An exec into a deployment pinned to `0,1,2,3` asking for
# card 4 does not fail: it runs, on a card that deployment was never allocated,
# and returns a number.
#
# **That is worse than any refusal chased today, because it produces a value
# rather than a stop** — and the value lands in a `cost: gpu_hours` validator
# whose output someone will believe. m1 warned about exactly this on 217
# (*"the cards you'd get are mine (0-3), not 4-7"*), and until now the only
# thing standing between that warning and a wrong number was remembering it.
#
# **Refuse, never correct.** Silently substituting a card the container does
# own would be this same defect wearing a fix: the caller asked a question about
# one card and would get an answer about another.
#
# An unpinned container (no `HIP_VISIBLE_DEVICES` in its env) constrains
# nothing, so there is nothing to check and this says nothing. `--device`
# whitelisting is deliberately **not** read: m3's rule is to extend a case with
# a form you have SEEN, and every container this package has met pins by env.
#
# **Demonstrated against a real pinned deployment**, m1's
# `yihou_e2e_flow_sgl_m1r1` on `crsuse2-m2m-006`, 2026-09-04, `docker inspect`
# only and no exec: `HIP_VISIBLE_DEVICES=0,1,2,3` in `Config.Env` with
# `HostConfig.Devices` = `/dev/kfd`, `/dev/dri` — the whole card set present and
# only the variable narrowing it, confirming on a live container what
# `start_container.sh:37-44` says on disk. Asking for card 4 and card 7 each
# refused with exit 1, and the command the wrapper was given (an `echo` that
# would have proved an exec happened) never ran.
#
# **The coupling, for whoever changes the bring-up:** if m1 ever pins by
# per-card `--device /dev/dri/renderD<N>` and drops the env var, this check
# finds no pin, takes the branch above and waves the exec through. That is still
# *safe* — a card outside the whitelist genuinely is not there — but safe by
# absence rather than by refusal, and the failure arrives as a deep HIP error
# instead of a sentence. Keep both: the variable for this to read, the whitelist
# for enforcement. m1 has this as a constraint in T27 and will say before
# switching.
# **The opening quote is load-bearing. Do not relax this pattern.**
#
# `.Config.Env` is a JSON array of `"NAME=value"` strings, so an unanchored
# match also fires on the tail of a *different* variable that merely ends in
# this name: `ROCR_HIP_VISIBLE_DEVICES=7` would be read as the pin. Matching
# `"NAME=` requires the element to start there.
#
# **The failure is not "the check stops working" — it is a check that gets it
# wrong in both directions at once.** With `7` mistaken for the pin of a
# container really pinned to `0,1,2,3`, a request for card 0 is *refused* though
# it is correct, and a request for card 7 is *passed* though it is not. A gate
# that refuses right answers and admits wrong ones is worse than no gate, and it
# would present as flakiness rather than as a bug in this line.
#
# Found only because the case was fabricated; no container in this cluster
# happens to carry such a variable, so nothing would have surfaced it in use.
# **`|| true` because `grep` exits 1 on no match and `pipefail` is on.** With no
# pin in the env — every ephemeral container, and every unpinned one — this
# pipeline legitimately finds nothing, `grep` reports that as failure, and
# `set -euo pipefail` then killed the whole script *at this assignment*. The
# symptom was the wrapper announcing the ephemeral fallback and exiting 1 with
# no message, which reads as the fallback failing rather than as a successful
# search for something absent. Found by `bash -x`; my first guess was a
# different line and was wrong.
CPIN_ALL="$(printf '%s' "$CSTATE" | sed -n 's/.*E2E_ENV=//p' \
  | grep -o '"HIP_VISIBLE_DEVICES=[^"]*' | head -1 | sed 's/^"HIP_VISIBLE_DEVICES=//' || true)"
if [ -n "$CPIN_ALL" ]; then
  case ",$CPIN_ALL," in
    *",$HIP_VISIBLE_DEVICES,"*) ;;
    *)
      echo "run_in_container: this exec asks for GPU $HIP_VISIBLE_DEVICES, but container" >&2
      echo "    $CONTAINER" >&2
      echo "  was brought up pinned to HIP_VISIBLE_DEVICES=$CPIN_ALL." >&2
      echo "  The pin is an env var and the container holds /dev/dri whole, so this exec would" >&2
      echo "  NOT fail -- it would run on card $HIP_VISIBLE_DEVICES and return a number for a card this" >&2
      echo "  deployment was never allocated. Refusing rather than substituting one of" >&2
      echo "  {$CPIN_ALL}: you asked about a specific card and an answer about a different" >&2
      echo "  one is not a smaller answer, it is a wrong one." >&2
      echo "  Pass --var gpu=<one of $CPIN_ALL>, or measure in a container of your own." >&2
      exit 1
      ;;
  esac
fi

# **In the artefact, not only in the log** — the leader's ruling on T34,
# 2026-09-04: keep `premise.run_environment` meaning exactly what it means
# today (m1's record, carried faithfully) and put the observation *beside* it,
# so a later premise gate can compare the two instead of a reader comparing two
# log lines. A field that silently changed meaning would be worse than one that
# is honestly narrow.
#
# **The destination has to be writable from HERE, and `KFO_SCRATCH_ROOT` is
# not.** This wrote to `$KFO_SCRATCH_ROOT/observed_runtime.json` with the mkdir
# swallowed by `|| true` — and that root is node-local
# (`/mnt/m2m_nobackup/...`), so on the login node where every caller of this
# script actually runs, the mkdir failed, the write failed, and **both failures
# were silent**. The same locality mistake `e747653` fixed one layer up, still
# sitting here. An observation nobody can read is not a record.
#
# `KFO_OBSERVED_RUNTIME` names the file outright, and the caller picks somewhere
# it can read back — the validator uses its zone. The old spelling stays as a
# fallback but is now *tested* for writability instead of assumed, and says so
# when it is not.
OBSERVED="${KFO_OBSERVED_RUNTIME:-}"
if [ -z "$OBSERVED" ] && [ -n "${KFO_SCRATCH_ROOT:-}" ]; then
  if mkdir -p "$KFO_SCRATCH_ROOT" 2>/dev/null; then
    OBSERVED="$KFO_SCRATCH_ROOT/observed_runtime.json"
  else
    echo "run_in_container: cannot record the container observation under" >&2
    echo "  KFO_SCRATCH_ROOT=$KFO_SCRATCH_ROOT (not writable from this host; it is" >&2
    echo "  node-local). Pass KFO_OBSERVED_RUNTIME=<a path this host can write>." >&2
  fi
fi
if [ -n "$OBSERVED" ]; then
  mkdir -p "$(dirname "$OBSERVED")" 2>/dev/null || true
  _get() { printf '%s' "$CSTATE" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1; }
  cat > "$OBSERVED" <<JSON || true
{
  "mode": "$MODE",
  "_mode_means": "record = measured inside the container the deploy_kit names, carrying the deployment's own engine state. ephemeral = that container was not running, so this measured in a throwaway started from the image the record names, which carries the IMAGE's state and not the deployment's. They are different claims.",
  "container": "$CONTAINER",
$([ "$MODE" = record ] && cat <<REC
  "container_id": "$(_get E2E_ID)",
  "created": "$(_get E2E_CREATED)",
  "started_at": "$(_get E2E_STARTED)",
  "restart_count": "$(_get E2E_RESTARTS)",
REC
)
  "node": "${E2E_NODE:-}",
  "gpu_pin": "$CPIN_ALL",
  "gpu_used": "$HIP_VISIBLE_DEVICES",
  "record_claims_started_at": "$(_field runtime.started_at)",
  "observed_by": "run_in_container.sh"
}
JSON
fi

# **Single-quote for the shell that will re-parse this, escaping embedded
# quotes.** `on` hands its argument to a `bash -lc`, so everything assembled
# below is parsed by a shell one more time. Wrapping in bare `'...'` breaks the
# moment a value contains a quote — and the FIRST real use of this script was
# going to be
#
#     python3 -c 'import torch; print(torch.__version__)'
#
# which contains two, and which died with `syntax error near unexpected token`
# from the outer shell rather than anything to do with the container. Caught
# 2026-09-04 by assembling the real probe against a stub docker before the node
# window opened, which is the only reason it did not burn it.
_sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# `PYTHONDONTWRITEBYTECODE=1`: root-owned `__pycache__` inside a handoff is
# CONTRACT section 5.0 arriving early. `reclaim.sh` handles the rest, from
# inside this same container, which is the only context with the privilege.
EXEC_ENV="-e PYTHONDONTWRITEBYTECODE=1 -e HIP_VISIBLE_DEVICES=$(_sq "$HIP_VISIBLE_DEVICES")"
for name in TRITON_CACHE_DIR KNOWLEDGE_LOCAL_ROOT TMPDIR; do
  eval "value=\${$name:-}"
  [ -n "$value" ] && EXEC_ENV="$EXEC_ENV -e $name=$(_sq "$value")"
done

# **`AGENT_SYS_*`, `KFO_*` and `E2E_*` by PREFIX, never by name — and the
# enumeration that used to be here was itself the defect.**
#
# Two failures on this wrapper's first real use, 2026-09-04, both from the same
# hand-maintained list:
#
#   * `AGENT_SYS_INPUT_<KIND>` / `AGENT_SYS_OUTPUT_<KIND>` were absent, so STEP 1
#     inside the container refused with *"AGENT_SYS_INPUT_OPERATOR_WORKSET is
#     unset; this task does not have operator_workset as an input"* — which reads
#     as a wiring fault in the task and was a gap in the boundary crossing.
#   * `KFO_MOCK` was absent, so `--var forge_mock=1` did not arrive and
#     `30_run_forge.sh` took the **real campaign** branch. It failed on git
#     rather than running for an hour, which is luck, not design.
#
# The kind is part of the variable name, so a fixed list would have to duplicate
# the kind list from `steps/m4_kernel_opt.yaml` — CONTRACT §4.3's shape, one
# authority and two readers, and the second reader silently narrower. A prefix
# has no list to fall behind.
#
# **The wrapper IS the boundary: anything it does not carry does not arrive**,
# and it arrives as a fault in whatever runs on the far side.
while IFS='=' read -r name value; do
  [ -n "$value" ] && EXEC_ENV="$EXEC_ENV -e $name=$(_sq "$value")"
done <<EOF
$(env | grep -E '^(AGENT_SYS|KFO|E2E)_' || true)
EOF
# Inside the image the venv is real, which is the whole point of entering it.
EXEC_ENV="$EXEC_ENV -e KFO_PYTHON=$(_sq "${KFO_PYTHON:-/opt/venv/bin/python3}")"

WORKDIR_ARG=""
[ -n "$WORKDIR" ] && WORKDIR_ARG="-w $(_sq "$WORKDIR")"

# Name the verb, because the two are different actions on different
# containers and this line is what a transcript reader anchors on.
if [ "$MODE" = record ]; then _VERB="exec into"; else _VERB="run a throwaway"; fi
echo "run_in_container: $_VERB $CONTAINER on ${E2E_NODE:-the node}, GPU $HIP_VISIBLE_DEVICES" >&2
# **"the next line comes from inside the container" is printed by whichever
# branch is about to cross, not here.** Announcing it up front is a promise made
# before the last thing that can refuse — the mount-form check below — so a
# refusal arrived with that sentence directly above it and nothing from inside
# following. A message that describes what did not happen is the same defect as
# an empty listing.

# **Create the scratch roots on the far side, because only the far side can.**
# `TMPDIR` and `TRITON_CACHE_DIR` point at node-local storage
# (`/mnt/m2m_nobackup/...`), which is an NVMe volume on the compute node; the
# login node's copy of that path is not writable by us, so a caller cannot
# create them before calling and rung 0 died on 2026-09-04 trying
# (`check_speedup_substantiated` → `PermissionError`).
#
# **An absent `TMPDIR` is not a harmless default here.** A `TMPDIR` naming a
# directory that does not exist makes every HIP kernel launch SIGSEGV with no
# output while `torch.cuda.is_available()` still returns `True` — the trap that
# cost the 2026-09-02 run 25 minutes. So forwarding the variable without
# creating the directory is the worst of the three options: it looks configured
# and it crashes in the kernel.
#
# Prepended to the command rather than run as a separate `docker exec`, so it
# cannot succeed in one exec and be missing in the next.
MKSCRATCH=""
for name in TMPDIR TRITON_CACHE_DIR KNOWLEDGE_LOCAL_ROOT KFO_SCRATCH_ROOT; do
  eval "value=\${$name:-}"
  [ -n "$value" ] && MKSCRATCH="$MKSCRATCH mkdir -p $(_sq "$value") || exit 1;"
done
[ -n "$MKSCRATCH" ] && COMMAND="$MKSCRATCH $COMMAND"

rc=0
if [ "$MODE" = record ]; then
  echo "run_in_container: the next line comes from inside the container" >&2
  # shellcheck disable=SC2086
  on "docker exec $EXEC_ENV $WORKDIR_ARG $(_sq "$CONTAINER") bash -lc $(_sq "$COMMAND")" || rc=$?
else
  # **`STARTED` gates the teardown, and that is m3's rule, not a nicety.** A
  # caller who points `KFO_MEASURE_CONTAINER` at a name that already exists must
  # not have it removed by us: this only ever tears down a container it started
  # itself, which is the file-wide rule *never `docker rm -f` something you did
  # not create* expressed in a lifecycle.
  #
  # `--rm` and the trap are both here because they cover different failures:
  # `--rm` handles the normal return, the trap handles being killed mid-measure.
  # The trap **reclaims before removing** — `reclaim.sh` works by `docker exec`
  # into a *running* container, so the other order leaves no window at all, and
  # m3 records that getting this backwards is what left root-owned evidence in
  # every workset the stage produced.
  #
  # The mounts are the three identity forms m1's `runtime_contract`
  # (`a32f06d`) records the daemon accepting, and identity-mapped for the reason
  # `_remeasure` needs: the apparatus path is computed on this host and handed
  # to a shell inside the container, so the two must agree.
  STARTED=0
  _teardown() {
    trap - EXIT HUP INT TERM   # disarm: a signal fires the handler and then EXIT
    [ "$STARTED" = 1 ] || return 0
    on "sh '$PKG/assets/lib/reclaim.sh' '$CONTAINER' '$(dirname "${OBSERVED:-/tmp}")'" || true
    on "docker rm -f '$CONTAINER' >/dev/null 2>&1 || true" || true
  }
  trap _teardown EXIT HUP INT TERM

  # **The mounts come from the paths that actually cross, never from `$HOME`.**
  #
  # This read `for m in /shared_nfs "$HOME" "$KFO_SCRATCH_ROOT"`, and `$HOME` is
  # the wrong source: a **validation zone redefines it** to `<zone>/home`
  # (measured — every `validation-*/` directory in a run has a `home/` inside
  # it). So the ephemeral container mounted a subdirectory of the zone, the
  # apparatus at `<zone>/substantiate-XXXX/seed` was never mounted at all, and
  # rung 0's fourth attempt died with `./run_performance.sh: No such file or
  # directory` — a container that came up perfectly and could not see its own
  # script.
  #
  # `--workdir` and the kit are the paths this exec genuinely needs to resolve
  # inside, so they are what decide the mounts. Each is reduced to the accepted
  # top-level form from m1's `runtime_contract.measurement_visible` (`a32f06d`),
  # identity-mapped because `_remeasure` computes these paths here and hands
  # them to a shell over there.
  #
  # **A path in no accepted form is refused, not guessed.** m3's rule: extend
  # this with a form you have SEEN the daemon accept. Silently skipping it would
  # reproduce exactly the failure above — a container that starts and cannot see
  # what it was given.
  _mount_root() {  # path -> the accepted mount root containing it, or empty
    case "$1" in
      /shared_nfs/*|/shared_nfs)   printf '/shared_nfs' ;;
      /mnt/m2m_nobackup/*)         printf '%s' "$(printf '%s' "$1" | cut -d/ -f1-4)" ;;
      /home/*)                     printf '%s' "$(printf '%s' "$1" | cut -d/ -f1-3)" ;;
      *)                           printf '' ;;
    esac
  }
  MOUNTS=""
  for m in "$WORKDIR" "$KIT" "${TMPDIR:-}" "${KFO_SCRATCH_ROOT:-}" "${AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION:-}"; do
    [ -n "$m" ] || continue
    r="$(_mount_root "$m")"
    if [ -z "$r" ]; then
      echo "run_in_container: '$m' is under no mount form this cluster is known to accept" >&2
      echo "  (/shared_nfs, /home/<user>, /mnt/m2m_nobackup/<user>). Refusing rather than" >&2
      echo "  starting a container that cannot see it -- that failure arrives as 'No such" >&2
      echo "  file or directory' from inside and reads as a broken workset." >&2
      exit 1
    fi
    case " $MOUNTS " in *" -v $r:$r "*) continue ;; esac
    MOUNTS="$MOUNTS -v $r:$r"
  done
  STARTED=1
  echo "run_in_container: the next line comes from inside the container" >&2
  # shellcheck disable=SC2086
  on "docker run --rm --name $(_sq "$CONTAINER") \
        --device /dev/kfd --device /dev/dri --group-add video \
        --ipc=host --shm-size 16g --security-opt seccomp=unconfined \
        $MOUNTS $EXEC_ENV $WORKDIR_ARG $(_sq "$IMAGE") bash -lc $(_sq "$COMMAND")" || rc=$?
  _teardown
fi

# CONTRACT section 5.0, in a `finally`: idempotent, and a no-op when there is
# nothing root-owned, so this does not decide first whether it will be needed.
# Only meaningful in `record` mode — an ephemeral container is gone by now, and
# its payload handed the files back before it exited.
if [ "$MODE" = record ] && [ -n "${AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION:-}" ]; then
  sh "$PKG/assets/lib/reclaim.sh" "$CONTAINER" "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION" 2>/dev/null || true
fi
[ "$MODE" = record ] && [ -n "${KFO_SCRATCH_ROOT:-}" ] && \
  sh "$PKG/assets/lib/reclaim.sh" "$CONTAINER" "$KFO_SCRATCH_ROOT" 2>/dev/null || true

exit "$rc"
