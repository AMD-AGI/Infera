#!/bin/sh
# Assemble the flow's one export: eight handoffs spanning all five stages, laid
# out the way the `experiment-result-packup` skill's `deliverable_layout.md` says.
#
# **`content_type: code`, deliberately.** Laying a packup into a `reproducible`
# kind renames `results/` to `items/result` and leaves `REPRODUCE.md` with no
# item to be, which destroys exactly what `check_packup_shape` exists to check.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
ENVYAML="${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml"

rc=0
bash "$PKG/assets/lib/mock_m5.sh" packup "$ENVYAML" || rc=$?
if [ "$rc" -eq 0 ]; then exit 0; fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi

"${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/packup.task/packup.py" \
  --out "${AGENT_SYS_OUTPUT_E2E_PACKUP:?AGENT_SYS_OUTPUT_E2E_PACKUP is unset}" \
  --package "$PKG"

# G5: every handoff carries the environment record, and a `code` kind carries it
# at `items/codes/environment.yaml`. Inherited from m1 rather than re-derived.
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/lib/env_render.py" \
  --inherit "$ENVYAML" \
  --content-type code \
  --out "$AGENT_SYS_OUTPUT_E2E_PACKUP"
