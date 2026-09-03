#!/bin/sh
# The body of `check_capabilities_genuine`. Run by `validator.ScriptBodyRunner`
# as `/bin/sh <package_root>/assets/check_capabilities_genuine.validator/entry.sh`,
# with `cwd` set to a freshly allocated validation zone holding `args.json`,
# `inputs.json` and `materials.json`.
#
# **Both environment rows, in one line** — `AGENT_SYS_TASK_PACKAGE` is the
# PRODUCER row's and `AGENT_SYS_DEMO_PACKAGE` is the GLOBAL row's, and they are
# disjoint rather than nested (`validator` spec §8.2).
#
# This body needs the producer row for a second reason the shape check does not:
# `ENVCHK_NONCE` reaches it only from `Prepared.environment`, into which
# `env_mgr.material.deploy` merged the agent spec's `env` block. On the global
# row it is absent, and `check.py` says so by name rather than recomputing seven
# tokens from an empty string.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_capabilities_genuine.validator/check.py"
