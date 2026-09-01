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
# remote side sees `AGENT_SYS_OUTPUT_*` and the `PD_*` block from the agent spec.

on() {
  srun --jobid="${PD_JOBID:?PD_JOBID is unset}" --overlap -N1 -n1 \
    -w "${PD_NODE:?PD_NODE is unset}" --export=ALL bash -lc "$*"
}

# Assert that a path this side can see resolves on the node too.
#
# The bodies run scripts out of the staged task package by absolute path, which
# only works because $HOME here is NFS from psnfs01 and the compute nodes mount
# the same export. That is a fact about this cluster, not about agent_sys, so it
# is checked rather than assumed -- if the run root ever moves to local disk the
# failure should name the reason instead of arriving as "No such file".
require_visible_on_node() {
  local path="$1" what="$2"
  on "test -e '$path'" >/dev/null 2>&1 && return 0
  echo "the $what is not visible on $PD_NODE: $path" >&2
  echo "This package runs its bodies on the node by absolute path, which needs" >&2
  echo "the run root to be on a filesystem both hosts mount. Point --demo-root" >&2
  echo "at a shared path (\$HOME here is NFS and works)." >&2
  return 1
}
