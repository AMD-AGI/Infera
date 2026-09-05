#!/bin/sh
# SKELETON. The owner replaces `check.py` with the real body.
#
# **Both environment fallbacks, deliberately.** A validator's INPUT phase gets
# the GLOBAL row and never `AGENT_SYS_TASK_PACKAGE`; only the PRODUCER row
# exports it. Writing one of the two has already cost a run.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
# `"${AGENT_SYS_DEMO_PYTHON:-python3}"` and not a bare `python3`.
# `cli/main.py:668` exports the run's own interpreter under that name, and a
# bare `python3` resolves through the zone's PATH instead — which on this
# cluster is /usr/bin/python3, where `referencing` is not installed. Measured:
# every validator that loads `assets/lib/schema.py` died on
# `ModuleNotFoundError: No module named 'referencing'` and wrote no verdict,
# so the phase reported "nothing was decided" rather than a pass or a fail.
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/check_measurement_order.validator/check.py" "$@"
