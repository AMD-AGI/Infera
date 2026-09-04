#!/usr/bin/env bash
# Run a command on the compute node this package was pointed at.
#
# This is the seam the 1P1D kit calls `$SSH_CMD` and documents as
# `export SSH_CMD="<your-scheduler> exec"`. A login node cannot ssh to a compute
# node on either cluster this package has run on, so the transport is always
# "step into an allocation that is already holding the node".
#
# **There are two such transports and which one exists is a fact about the
# cluster, not about this package**, so `on` dispatches rather than hard-coding
# one. `PD_TRANSPORT` picks it; `auto` (the default) probes.
#
#   srun  Slurm. `srun --jobid=<id> --overlap -N1 -n1 -w <node> --export=ALL`.
#         `--overlap` is required, not decorative: without it srun waits for the
#         job's own step to release the resources, and that step is
#         `sleep infinity`, so the call hangs forever with no output.
#
#   local agent_sys is itself running on the compute node, so `on` is a plain
#         `bash -lc`. Explicit only -- `auto` never picks it, because "neither
#         transport binary is present" does not prove "I am on the node", and
#         guessing wrong runs every GPU command on the login node. Worth having:
#         the login node here is shared by ~170 users at 62 GB, and an
#         `agent-sys` driving a long graph from it can be OOM-killed mid-run.
#
#   spur  The AMD crsuse2-m2m cluster. `spur exec <jobid> bash -lc`. The
#         controller proxies to whichever node holds the job, so **no node name
#         is passed** -- `PD_NODE` stays a fact this package records rather than
#         one the transport consumes.
#
# On the spur cluster `srun` is NOT Slurm: `/usr/local/bin/srun` is a spur
# re-implementation that does not accept `--export` at all and exits 128 with
# "raw mode unavailable (stdin is not a TTY)" when driven non-interactively.
# So probing for the `srun` binary is not enough to choose it -- `spur` wins the
# probe wherever it exists.
#
# The command arrives as one string and is handed to `bash -lc`. Passing it to
# the transport directly would make it look for a binary whose name is the whole
# string. Measured against both transports: exit codes propagate exactly
# (`exit 42` arrives as 42), nested quoting survives, pipes work, and two calls
# may overlap -- `assets/load/replay.sh` backgrounds one and needs that.
#
# **The environment is NOT carried across, and nothing here needs it to be.**
# `srun --export=ALL` did copy this process's environment, but no script this
# seam invokes on the far side reads a `PD_*` or `AGENT_SYS_*` name: every call
# site passes what its script reads as an explicit `VAR='...'` prefix. That is
# deliberate and predates this change -- see the comment at
# `assets/serve/round.sh:93`, which records an early run that appeared to work
# only because the operator's login shell still had the names exported. `spur
# exec` carries no environment at all (measured: `MARK=x spur exec <id> bash -lc
# 'echo $MARK'` prints empty), so relying on inheritance would have been a
# latent fault on Slurm and is an immediate one here.

: "${PD_TRANSPORT:=auto}"

# Resolved once, at source time, so a probe does not run per call.
if [ "$PD_TRANSPORT" = "auto" ]; then
  if command -v spur >/dev/null 2>&1; then
    PD_TRANSPORT=spur
  else
    PD_TRANSPORT=srun
  fi
fi

on() {
  case "$PD_TRANSPORT" in
    local)
      # agent_sys is already running ON the compute node, so there is nothing to
      # step into. Never selected by `auto`: "no transport binary is present" is
      # not the same fact as "I am on the node", and guessing wrong would run
      # every GPU command on the login node. Ask for it explicitly.
      bash -lc "$*"
      ;;
    spur)
      spur exec "${PD_JOBID:?PD_JOBID is unset}" bash -lc "$*"
      ;;
    srun)
      srun --jobid="${PD_JOBID:?PD_JOBID is unset}" --overlap -N1 -n1 \
        -w "${PD_NODE:?PD_NODE is unset}" --export=ALL bash -lc "$*"
      ;;
    *)
      echo "PD_TRANSPORT=$PD_TRANSPORT is not one of: auto, local, spur, srun" >&2
      return 2
      ;;
  esac
}

# Assert that a path this side can see resolves on the node too.
#
# The bodies run scripts out of the staged task package by absolute path, which
# only works because the run root is on a filesystem both hosts mount. On the
# Slurm cluster that was $HOME (NFS from psnfs01); on the spur cluster it is
# /shared_nfs. That is a fact about a cluster, not about agent_sys, so it is
# checked rather than assumed -- if the run root ever moves to local disk the
# failure should name the reason instead of arriving as "No such file".
require_visible_on_node() {
  local path="$1" what="$2"
  on "test -e '$path'" >/dev/null 2>&1 && return 0
  echo "the $what is not visible on ${PD_NODE:-the compute node}: $path" >&2
  echo "This package runs its bodies on the node by absolute path, which needs" >&2
  echo "the run root to be on a filesystem both hosts mount. Point --demo-root" >&2
  echo "at a shared path (/shared_nfs here; \$HOME on the Slurm cluster)." >&2
  return 1
}
