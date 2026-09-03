#!/usr/bin/env bash
# SKELETON. The owner replaces `check.py` with the real body.
#
# **Both environment fallbacks, deliberately.** A validator's INPUT phase gets
# the GLOBAL row and never `AGENT_SYS_TASK_PACKAGE`; only the PRODUCER row
# exports it. Writing one of the two has already cost a run.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
exec python3 "$PKG/assets/check_measurement_order.validator/check.py" "$@"
