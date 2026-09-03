#!/bin/sh
# Turn m4's optimised kernel into a set of per-file read-only bind mounts.
#
# **First among the expensive steps' predecessors, and it costs seconds.** That
# is the point of its position in the graph: a patch that will not apply should
# fail here rather than after the stock arm has spent twenty minutes measuring a
# baseline nobody is going to use.
#
# A program and not an AI leaf, because m4's `apply` block was written against
# the workset's declared integration point (M5.1.1) — the judgement about which
# file in the engine the optimised kernel replaces was made two stages ago, and
# what is left is extract, hash, apply, compile, copy.
#
# `/bin/sh` and `set -eu`, not bash and not `pipefail`: agent_sys never reads the
# shebang, it invokes `["/bin/sh", entry]`, and `/bin/sh` here is dash, which
# exits 2 on `set -o pipefail` before line 2 (CONTRACT.md §3.2a).
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `|| rc=$?` and not `; rc=$?`: under `set -e` a simple command exiting
# non-zero kills the script before the assignment runs, so the branch below
# was never reached and a real-mode task died with the mock's status.
# A `||` puts it in a condition context, where `set -e` does not fire.
rc=0
bash "$PKG/assets/lib/mock.sh" stage5-integration patch_overlay || rc=$?
# 0 = mocked and written; 3 = this stage is not mocked, fall through to the
# real work; anything else is a mock that failed and must not be read as
# either.
if [ "$rc" -eq 0 ]; then
  # Adaptation (A): the sealed handoff predates `environment.yaml`, so the mock
  # renders one from m1's record. Without this the copy fails
  # `check_environment` — which is the schema doing its job rather than an
  # oversight, and is why nothing here synthesises the record's *contents*.
  exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/lib/env_render.py" \
    --inherit "${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml" \
    --content-type reproducible \
    --out "${AGENT_SYS_OUTPUT_PATCH_OVERLAY:?}"
fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi

exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/apply_patch.task/apply.py" \
  --kernel-optimization "${AGENT_SYS_INPUT_KERNEL_OPTIMIZATION:?AGENT_SYS_INPUT_KERNEL_OPTIMIZATION is unset}" \
  --deploy-kit "${AGENT_SYS_INPUT_DEPLOY_KIT:?AGENT_SYS_INPUT_DEPLOY_KIT is unset}" \
  --operator-workset "${AGENT_SYS_INPUT_OPERATOR_WORKSET:-}" \
  --out "${AGENT_SYS_OUTPUT_PATCH_OVERLAY:?AGENT_SYS_OUTPUT_PATCH_OVERLAY is unset}" \
  --package "$PKG"
