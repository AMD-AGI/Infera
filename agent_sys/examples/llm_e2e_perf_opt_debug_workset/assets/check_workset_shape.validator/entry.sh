#!/bin/sh
# A SHIM, for the same reason as the task body's: the check under test must be
# the real one. `check.py` resolves its `zone` helper from its own `__file__`,
# so running it from the real package picks up the real `assets/lib/zone.py`
# too — nothing here needs a second copy of that either.
set -eu
: "${KFO_REAL_PACKAGE:?pass --var real_package=<abs path to .../llm_e2e_performance_optimization/kernel-opt-demo>}"
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" "$KFO_REAL_PACKAGE/assets/check_workset_shape.validator/check.py"
