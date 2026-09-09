#!/usr/bin/env bash
# Turn a patch set into a set of per-file bind mounts.
#
# Runs first among the expensive steps' predecessors and costs seconds, which is
# the point: a patch that will not apply should fail here rather than after the
# stock arm has spent twenty minutes measuring a baseline nobody will use.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

exec python3 "$PKG/assets/apply_patch.task/apply.py" \
  --patch-dir "${AGENT_SYS_INPUT_KERNEL_PATCH:?AGENT_SYS_INPUT_KERNEL_PATCH is unset}" \
  --out "${AGENT_SYS_OUTPUT_PATCH_OVERLAY:?AGENT_SYS_OUTPUT_PATCH_OVERLAY is unset}" \
  --package "$PKG"
