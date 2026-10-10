#!/bin/sh
# **This closure runs under `kind: ai` (`e2e_integrator`), so this file is not
# the body — `readme.md` is.** The STEPS section there is the method; the AI
# sequences it and decides what to do when a step is ambiguous (M5.2, G4.2.1).
#
# It is kept, and it is not dead weight, for one case: a pure-mock run.
#
#     --var m5_agent=runner --var mock_stages=all
#
# swaps in the shared program agent, which does run this file, which mocks —
# producing the three handoffs from the sealed evidence with no model call and no
# GPU. **Without that switch a mocked m5 is not mocked at all**: a `kind: ai`
# task never runs `entry.sh`, so promoting this closure to `ai` took it off the
# mock path, and the first full mock run sat at `integrate_and_verify: running`
# while an agent prepared to bring a model up for real. The default is the real
# agent, so the mock is the thing you have to ask for.
#
# It deliberately does NOT attempt the real measurement. A shell script cannot do
# what steps 2 to 8 of the readme ask for — decide that four minutes of
# `Health check failed` is a JIT compile rather than a hang, invent correctness
# cases that could not have been prepared for — and one that pretended to would
# be the "reported PASS over a run in which every result was zero" failure this
# package is built against.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
ENVYAML="${AGENT_SYS_INPUT_DEPLOY_KIT:?}/items/codes/environment.yaml"

rc=0
bash "$PKG/assets/lib/mock_m5.sh" arms "$ENVYAML" || rc=$?
if [ "$rc" -eq 0 ]; then
  exec bash "$PKG/assets/lib/mock_m5.sh" report "$ENVYAML"
fi
if [ "$rc" -ne 3 ]; then exit "$rc"; fi

cat >&2 <<'MSG'
integrate_and_verify: this stage is not mocked, and this entry.sh is not its body.

The body is assets/integrate_and_verify.task/readme.md, run by the `e2e_integrator`
agent (kind: ai). Either:
  - set --var mock_stages to include m5, or
  - run the closure under its declared AI agent.
MSG
exit 1
