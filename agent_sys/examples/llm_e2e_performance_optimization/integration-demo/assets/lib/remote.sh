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
# remote side sees `AGENT_SYS_OUTPUT_*` and the `IT_*` block from the agent spec.

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
on() {
  srun --jobid="${IT_JOBID:?IT_JOBID is unset}" --overlap -N1 -n1 \
    -w "${IT_NODE:?IT_NODE is unset}" --export=ALL bash -lc "$*" </dev/null
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
  echo "the $what is not visible on $IT_NODE: $path" >&2
  echo "This package runs its bodies on the node by absolute path and exchanges" >&2
  echo "files through the zone, which needs the run root to be on a filesystem" >&2
  echo "both hosts mount. Point --demo-root at a shared path (\$HOME here is NFS" >&2
  echo "and works)." >&2
  return 1
}
