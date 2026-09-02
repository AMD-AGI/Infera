#!/bin/sh
# The body of `check_optimization_shape`. Same shape as
# `check_workset_shape`'s entry and for the same reason: the two §8.2
# environment rows are disjoint rather than nested, so a body naming only one
# dies on the other phase.
#
# **Today this kind reaches only the PRODUCER row**, because
# `kernel_optimization` has no consumer — `optimize_kernel` is the graph's
# `is_end` and nothing takes its output as an input. The GLOBAL fallback costs
# nothing and removes the trap for stage 5, which is exactly the task that will
# consume this kind.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_optimization_shape.validator/check.py"
