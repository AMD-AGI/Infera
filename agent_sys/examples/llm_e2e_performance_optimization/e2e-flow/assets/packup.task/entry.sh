#!/usr/bin/env bash
# Assemble the flow's one export: eight handoffs spanning all five stages, laid
# out the way the `experiment-result-packup` skill's `deliverable_layout.md` says.
#
# **`content_type: code`, deliberately.** Laying a packup into a `reproducible`
# kind renames `results/` to `items/result` and leaves `REPRODUCE.md` with no
# item to be, which destroys exactly what `check_packup_shape` exists to check.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

rc=0
bash "$PKG/assets/lib/mock_m5.sh" packup \
  "${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml" || rc=$?
if [ "$rc" -eq 0 ] && [ -n "$(ls -A "${AGENT_SYS_OUTPUT_E2E_PACKUP:?}" 2>/dev/null)" ]; then
  exit 0
fi
if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then exit "$rc"; fi

python3 "$PKG/assets/packup.task/packup.py" \
  --out "${AGENT_SYS_OUTPUT_E2E_PACKUP:?AGENT_SYS_OUTPUT_E2E_PACKUP is unset}" \
  --package "$PKG"

# G5: every handoff carries the environment record, and a `code` kind carries it
# at `items/codes/environment.yaml`. Inherited from m1 rather than re-derived.
exec python3 "$PKG/assets/lib/env_render.py" \
  --inherit "${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml" \
  --content-type code \
  --out "$AGENT_SYS_OUTPUT_E2E_PACKUP"
