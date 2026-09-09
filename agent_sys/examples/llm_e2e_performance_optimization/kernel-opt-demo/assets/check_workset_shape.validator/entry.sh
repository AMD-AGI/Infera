#!/bin/sh
# The body of `check_workset_shape`. Run by `validator.ScriptBodyRunner` with
# `cwd` set to a freshly allocated validation zone holding `args.json`,
# `inputs.json` and `materials.json`.
#
# **Both environment rows, in one line, and this validator really does hit
# both.** `AGENT_SYS_TASK_PACKAGE` is what the PRODUCER row exports and
# `AGENT_SYS_DEMO_PACKAGE` is what the GLOBAL row exports; they are disjoint,
# not nested (`validator` spec §8.2), so a body naming only one dies on the
# other phase.
#
# The `workset` kind is validated **twice**: in `publish_workset`'s output phase
# (PRODUCER row) and again in `optimize_kernel`'s input phase (GLOBAL row —
# §8.2's CONSUMER row has no source and cannot have one, because `env.prepare`
# runs strictly after `INPUT_VALIDATING`). Unlike the template's, this fallback
# is not insurance against a future consumer: it is load-bearing today.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_workset_shape.validator/check.py"
