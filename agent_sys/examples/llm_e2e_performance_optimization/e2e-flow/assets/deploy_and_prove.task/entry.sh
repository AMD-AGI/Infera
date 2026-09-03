#!/bin/sh
# The mock path for `deploy_and_prove`, and only that.
#
# This task's agent is `kind: ai` (`../../steps/m1_deploy.yaml`), so the AI
# backend never runs this file — the brief in `readme.md` is the task, and its
# STEPS section step 0 tells the agent to call `mock.sh` itself. This script
# exists so a person, or a wiring run driven by `agent: runner`, can take the
# mock path without reading the readme.
#
# POSIX: bodies are invoked as `["/bin/sh", entry]` (`agent/backends/program.py:83`),
# so the shebang is never consulted and `set -o pipefail` would be a hard exit 2
# under dash.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `bash`, explicitly: `mock.sh` uses `${!var}` indirect expansion, which dash
# does not have and which fails as `Bad substitution`.
bash "$PKG/assets/lib/mock.sh" stage1-deploy deploy_kit && exit 0

echo "deploy_and_prove: stage1-deploy is not in E2E_MOCK_STAGES, and there is no" >&2
echo "program body for a real bring-up: it is judgement work and belongs to the" >&2
echo "kind: ai agent. Run the closure with its declared agent, or set" >&2
echo "--var mock_stages=all to take the mock path." >&2
exit 1
