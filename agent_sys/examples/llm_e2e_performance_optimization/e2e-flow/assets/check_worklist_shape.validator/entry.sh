#!/bin/sh
# SKELETON. The owner replaces `check.py` with the real body.
#
# **Both environment fallbacks, deliberately.** A validator's INPUT phase gets
# the GLOBAL row and never `AGENT_SYS_TASK_PACKAGE`; only the PRODUCER row
# exports it. Writing one of the two has already cost a run.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
# **`$AGENT_SYS_DEMO_PYTHON` and not a bare `python3`.** `cli/main.py:668`
# exports the interpreter the run itself is using; a validation zone gets a
# policy-derived `PATH` on which `python3` resolves to `/usr/bin/python3`,
# which on this host has no `referencing` and therefore cannot import
# `assets/lib/schema.py`. Measured: the body dies with `ModuleNotFoundError`
# before writing `verdict.json`, and the phase reports "nothing was decided"
# rather than a verdict -- a validator that cannot start looks like one that
# was never asked. Found by m5 driving their leaves through the graph.
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$PKG/assets/check_worklist_shape.validator/check.py" "$@"
