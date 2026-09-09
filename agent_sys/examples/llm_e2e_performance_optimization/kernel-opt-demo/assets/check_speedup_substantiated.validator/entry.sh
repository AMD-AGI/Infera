#!/bin/sh
# The body of `check_speedup_substantiated`.
#
# **This body needs a GPU, and the PRODUCER row is what gives it one.** A
# validator body gets a *closed* environment — `os.environ` is deliberately not
# inherited (`validator/environment.py:84-90`) — so `HIP_VISIBLE_DEVICES` does
# not arrive by accident. It arrives because this kind is validated on
# `optimize_kernel`'s **output** phase, whose row is PRODUCER, whose source is
# that task's resolved config, which is where the agent spec's `env:` block
# ends up. `HIP_VISIBLE_DEVICES`, `TRITON_CACHE_DIR` and `KFO_SCRATCH_ROOT` all
# reach this script by that route and by no other.
#
# The GLOBAL fallback below is for the package variable only. If this validator
# ever ran on an input phase it would have no GPU variable at all, and the
# measurement would silently land on card 0 — which on a shared host is
# somebody else's. It does not run on an input phase today: `kernel_optimization`
# has no consumer.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_speedup_substantiated.validator/check.py"
