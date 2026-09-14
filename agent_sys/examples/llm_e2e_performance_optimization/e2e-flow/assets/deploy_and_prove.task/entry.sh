#!/bin/sh
# The mock path for `deploy_and_prove`.
#
# This task's default agent is `kind: ai` (`../../steps/m1_deploy.yaml`), and an
# AI backend never runs this file — the brief in `readme.md` is the task. This
# script is what runs when the closure is driven by the program agent instead
# (`--var m1_agent=runner`), which is how the mock e2e run drives stage 1.
#
# POSIX: bodies are invoked as `["/bin/sh", entry]`
# (`agent/backends/program.py:83`), so the shebang is never consulted and
# `set -o pipefail` would be a hard exit 2 under dash.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `bash`, explicitly, for both: `mock.sh` uses `${!var}` indirect expansion and
# `mock_adapt.sh` uses arrays and `${var:=}` defaulting that dash does not have.
#
# `|| rc=$?` and not `; rc=$?`: under `set -e` a simple command exiting non-zero
# kills the script *before* the assignment runs, so the `; ` form never reaches
# its own branches. `mock.sh` exits 0 when it mocked, and 3 when the stage is not
# in `$E2E_MOCK_STAGES`.
rc=0
bash "$PKG/assets/lib/mock.sh" stage1-deploy deploy_kit || rc=$?

if [ "$rc" = 0 ]; then
  # MOCK-MAP (A) and (I). The sealed bytes are correct and incomplete: they
  # predate `codes/environment.yaml` and the runtime contract, and both are
  # things this package requires of a kit rather than things that run.
  # `check_deploy_kit` fails the unadapted copy on exactly those two, which is
  # the validator working — so the adaptation runs here, after the copy, and the
  # same script builds `gate.sh`'s positive fixture.
  #
  # **Teed to a file, because a task body's stderr goes nowhere a person can
  # read.** Measured: this adaptation failed inside a run and the only visible
  # symptom was `check_deploy_kit` reporting a missing `environment.yaml` two
  # phases later — the cause named nothing and was three layers from the effect.
  # The log is outside the handoff on purpose: it is diagnostics about producing
  # the artefact, not part of it.
  #
  # **`|| rc=$?`, never `if ! cmd; then rc=$?`.** `!` inverts the status, so `$?`
  # inside the `then` branch is the *negated* one — 0 — in both dash and bash.
  # This block had that shape, and the effect was the worst available: the
  # failure was caught, logged in full, and reported as **success**, so the task
  # was marked succeeded and handed on a `deploy_kit` with no environment
  # record. `check_environment` then refused it two phases later and the graph
  # blamed the handoff instead of the task — precisely the three-layers-from-the
  # -effect problem the log above exists to prevent. Found by m2 sweeping every
  # body in the package; one site, this one.
  log="${TMPDIR:-/tmp}/m1_mock_adapt.$$.log"
  arc=0
  bash "$PKG/assets/deploy_and_prove.task/mock_adapt.sh" \
        "${AGENT_SYS_OUTPUT_DEPLOY_KIT:?}" >"$log" 2>&1 || arc=$?
  if [ "$arc" != 0 ]; then
    echo "deploy_and_prove: mock_adapt.sh failed (rc=$arc). Its output:" >&2
    cat "$log" >&2
    exit "$arc"
  fi
  cat "$log" >&2
  exit 0
fi

if [ "$rc" = 3 ]; then
  echo "deploy_and_prove: stage1-deploy is not in E2E_MOCK_STAGES, and there is no" >&2
  echo "program body for a real bring-up: it is judgement work and belongs to the" >&2
  echo "kind: ai agent. Run the closure with its declared agent (drop" >&2
  echo "--var m1_agent=runner), or set --var mock_stages=all to take the mock path." >&2
  exit 1
fi

echo "deploy_and_prove: mock.sh failed with rc=$rc" >&2
exit "$rc"
