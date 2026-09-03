#!/usr/bin/env bash
# Run a command on the compute node this package was pointed at.
#
# This is the seam the 1P1D kit calls `$SSH_CMD` and documents as
# `export SSH_CMD="<your-scheduler> exec"`. On this cluster the login node
# cannot ssh to a compute node, so the transport is `srun` into a Slurm
# allocation that is already running and holding the node.
#
# `--overlap` is required, not decorative: without it srun waits for the job's
# own step to release the resources, and that step is `sleep infinity`, so the
# call hangs forever with no output.
#
# The command arrives as one string and is handed to `bash -lc`. Passing it to
# srun directly would make srun look for a binary whose name is the whole
# string.
#
# `--export=ALL` carries this process's environment across, which is how the
# remote side sees `AGENT_SYS_OUTPUT_*` and the `E2E_*` block from the agent spec.
#
# **The transport is a variable, because it is a fact about the scheduler and not
# about this package.** `E2E_TRANSPORT=srun` is the Slurm-allocation form above.
# `E2E_TRANSPORT=spur` is `spur exec <jobid> bash -lc`, which is what the
# `amd-spur` partition offers instead: its login node holds no allocation an
# `srun --overlap` could attach to. Left unset, the transport is detected from
# which binary is on PATH, so neither cluster has to be told.
#
# Two differences make `spur exec` more than a command swap, both measured
# 2026-09-02 and both silent failures if ignored:
#
#   1. **It carries none of the caller's environment.** `--export=ALL` has no
#      equivalent; `spur exec <job> bash -lc 'echo $FOO'` prints nothing for an
#      exported FOO. So `_env_prelude` re-exports the `E2E_*` and `AGENT_SYS_*`
#      blocks explicitly, quoted with `printf %q`. Most call sites here already
#      pass what they need inline (`on "NODE_IP='$E2E_NODE_IP' … bash mix_up.sh"`)
#      and do not depend on this, but the prelude keeps the two transports
#      behaving alike rather than leaving a difference for a future call site to
#      fall into.
#   2. **It starts at `pwd=/` with `HOME=/opt/spur`**, where srun starts in the
#      caller's cwd with the caller's HOME. Both are restored below.

# `</dev/null` is not decoration, and the 1P1D kit's equivalent has it for the
# same two reasons.
#
# It stops srun reading the caller's stdin, which matters because several call
# sites here sit inside `while read … done < file` loops: without it srun
# swallows the loop's input and the loop runs once.
#
# And it detaches srun from the body's stdin so nothing downstream is holding a
# descriptor the framework is waiting on. `agent/backends/program.py` drains the
# body's stdout on a thread; a grandchild that keeps the write end open after the
# body exits leaves that drain blocked. Observed on 2026-08-31: `serve_patched`
# wrote its complete handoff at 17:21:55 and the framework still reported the
# task `running` and the handoff `generating` seventeen minutes later, until the
# settle budget expired.

# Which transport to use. Explicit `E2E_TRANSPORT` wins.
#
# Otherwise **spur is preferred over srun where both are on PATH**, which looks
# backwards and is not. On the `amd-spur` cluster `/usr/local/bin/srun` is a spur
# re-implementation, not Slurm's: it rejects `--export` outright and exits 128
# with "raw mode unavailable (stdin is not a TTY)" even without it. A body has no
# TTY, so detecting srun first would pick a transport that cannot work, and the
# failure would arrive as an exit code with no hint that the wrong seam was used.
# Presence of `spur` is the positive signal that this is that cluster.
_transport() {
  if [ -n "${E2E_TRANSPORT:-}" ]; then echo "$E2E_TRANSPORT"; return; fi
  if command -v spur >/dev/null 2>&1; then echo spur
  elif command -v srun >/dev/null 2>&1; then echo srun
  else echo spur; fi
}

# `export`s for the variables a remote body may read, as one `printf %q`-quoted
# string. Only the two prefixes this package actually puts on the wire: exporting
# the caller's whole environment would carry the login node's PATH and HOME onto
# a machine where neither is right.
_env_prelude() {
  local name val out=''
  # **`E2E_` and not `IT_`.** This file arrived from `integration-demo`, whose
  # variables are `IT_*`; this package's are `E2E_*` (CONTRACT.md 6). Left as
  # carried, this loop forwarded a prefix nothing in this package sets, so NO
  # `E2E_*` variable reached the remote side at all -- and the symptom would
  # have been an unset variable on the far end of an `spur exec`, naming
  # neither this line nor the rename.
  for name in $(compgen -v | grep -E '^(E2E_|AGENT_SYS_)' | sort); do
    eval "val=\${$name-}"
    out+="export $name=$(printf '%q' "$val"); "
  done
  printf '%s' "$out"
}

on() {
  case "$(_transport)" in
    srun)
      srun --jobid="${E2E_JOBID:?E2E_JOBID is unset}" --overlap -N1 -n1 \
        -w "${E2E_NODE:?E2E_NODE is unset}" --export=ALL bash -lc "$*" </dev/null
      ;;
    spur)
      # `cd` first: spur starts at `/`. `|| cd /` rather than failing, because a
      # caller whose cwd is a login-only path still deserves to run its command.
      spur exec "${E2E_JOBID:?E2E_JOBID is unset}" bash -lc \
        "export HOME=$(printf '%q' "${E2E_REMOTE_HOME:-$HOME}"); \
         cd $(printf '%q' "$PWD") 2>/dev/null || cd /; \
         $(_env_prelude) $*" </dev/null
      ;;
    *)
      echo "unknown E2E_TRANSPORT: ${E2E_TRANSPORT:-} (want 'srun' or 'spur')" >&2
      return 2
      ;;
  esac
}

# Assert that a path this side can see resolves on the node too.
#
# This package needs it in both directions: the bodies run staged scripts out of
# the task package by absolute path, and `apply_patch` has the node write files
# into the attempt zone with `docker cp` and reads them back. Both work because
# $HOME here is NFS from psnfs01 and the compute nodes mount the same export --
# a fact about this cluster, not about agent_sys, so it is checked rather than
# assumed. If the run root ever moves to local disk the failure should name the
# reason instead of arriving as "No such file".
require_visible_on_node() {
  local path="$1" what="$2"
  on "test -e '$path'" >/dev/null 2>&1 && return 0
  echo "the $what is not visible on $E2E_NODE: $path" >&2
  echo "This package runs its bodies on the node by absolute path and exchanges" >&2
  echo "files through the zone, which needs the run root to be on a filesystem" >&2
  echo "both hosts mount. Point --demo-root at a shared path (\$HOME here is NFS" >&2
  echo "and works)." >&2
  return 1
}
