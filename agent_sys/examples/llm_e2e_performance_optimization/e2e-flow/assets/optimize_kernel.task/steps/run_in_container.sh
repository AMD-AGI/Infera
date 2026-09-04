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
case "$CSTATE" in
  *E2E_STATE=true*) ;;
  *E2E_STATE=*)
    echo "run_in_container: the record names container" >&2
    echo "    $CONTAINER" >&2
    echo "  and ${E2E_NODE:-the node} answered that it is not running there. Modules 1-4 share" >&2
    echo "  ONE container and m1 owns its lifetime; m4 does not start one (CONTRACT section 5)." >&2
    echo "  Containers that ARE running there:" >&2
    # **`>&2` BEFORE `2>/dev/null`, and the other order silently emptied this
    # list.** Redirections apply left to right, so `2>/dev/null >&2` points fd2
    # at /dev/null and then duplicates it into fd1 — the listing goes to
    # /dev/null and the diagnostic prints a blank where the containers should
    # be, which reads as "there are none". Caught 2026-09-04 by the stubkit's
    # falsification run: the wrapper correctly refused a container that did not
    # exist and reported an empty node, while `yihou_m4_standalone_20260904` was
    # up on it. Exactly the failure this whole file keeps re-learning — an empty
    # result and a discarded one are indistinguishable to the reader.
    on "docker ps --format '    {{.Names}}   {{.Image}}'" >&2 2>/dev/null || true
    echo "  Either m1's bring-up has not run for this record, or it has been torn down." >&2
    exit 1
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
echo "run_in_container: observed $(printf '%s' "$CSTATE" | tr ' ' '\n' | grep -E '^E2E_(ID|CREATED|STARTED|RESTARTS)=' | tr '\n' ' ')" >&2
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
CPIN_ALL="$(printf '%s' "$CSTATE" | sed -n 's/.*E2E_ENV=//p' \
  | grep -o '"HIP_VISIBLE_DEVICES=[^"]*' | head -1 | sed 's/^"HIP_VISIBLE_DEVICES=//')"
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
# `KFO_SCRATCH_ROOT` because it is the one directory this wrapper already owns
# and already reclaims below. Absent, the observation is logged and not
# recorded, which is the pre-existing behaviour rather than a failure: a body
# that does not declare a scratch root has nowhere for this to live and should
# not have one invented for it.
if [ -n "${KFO_SCRATCH_ROOT:-}" ]; then
  mkdir -p "$KFO_SCRATCH_ROOT" 2>/dev/null || true
  _get() { printf '%s' "$CSTATE" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1; }
  cat > "$KFO_SCRATCH_ROOT/observed_runtime.json" <<JSON || true
{
  "container": "$CONTAINER",
  "container_id": "$(_get E2E_ID)",
  "created": "$(_get E2E_CREATED)",
  "started_at": "$(_get E2E_STARTED)",
  "restart_count": "$(_get E2E_RESTARTS)",
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

echo "run_in_container: exec into $CONTAINER on ${E2E_NODE:-the node}, GPU $HIP_VISIBLE_DEVICES" >&2
echo "run_in_container: the next line comes from inside the container" >&2

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
# shellcheck disable=SC2086
on "docker exec $EXEC_ENV $WORKDIR_ARG $(_sq "$CONTAINER") bash -lc $(_sq "$COMMAND")" || rc=$?

# CONTRACT section 5.0, in a `finally`: idempotent, and a no-op when there is
# nothing root-owned, so this does not decide first whether it will be needed.
if [ -n "${AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION:-}" ]; then
  sh "$PKG/assets/lib/reclaim.sh" "$CONTAINER" "$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION" 2>/dev/null || true
fi
[ -n "${KFO_SCRATCH_ROOT:-}" ] && \
  sh "$PKG/assets/lib/reclaim.sh" "$CONTAINER" "$KFO_SCRATCH_ROOT" 2>/dev/null || true

exit "$rc"
