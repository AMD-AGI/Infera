#!/bin/sh
# The mock path for `optimize_kernel`, and only that.
#
# This task's agent is `kind: ai` (`../../steps/m4_kernel_opt.yaml`), so the AI
# backend never runs this file — the brief in `readme.md` is the task. It exists
# so a person, or a wiring run driven by `agent: runner`, can take the mock path
# without reading the readme.
#
# POSIX: bodies are invoked as `["/bin/sh", entry]`
# (`agent/backends/program.py:83`), so the shebang is never consulted and
# `set -o pipefail` would be a hard exit 2 under dash.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"

# `bash`, explicitly: `mock.sh` uses `${!var}` indirect expansion, which dash
# does not have and which fails as `Bad substitution`.
#
# **Three outcomes, not two, and `|| rc=$?` is what keeps them apart.** `mock.sh`
# exits 0 having copied, **3** when this stage is not in `E2E_MOCK_STAGES`, and
# 1 when the copy itself failed. An `if …; then` would fold 1 and 3 together and
# report a genuine mock failure as "not selected, run it for real" — which under
# `set -eu` then exits 1 with the wrong sentence attached, and the wrong sentence
# is what sends somebody looking in the wrong place.
rc=0
bash "$PKG/assets/lib/mock.sh" stage4-kernel-opt kernel_optimization || rc=$?

if [ "$rc" -eq 1 ]; then
  echo "optimize_kernel: mock.sh failed to copy the sealed handoff; not falling through to" >&2
  echo "the real path, because a mock that half-copied is not an unmocked run." >&2
  exit 1
fi

if [ "$rc" -eq 0 ]; then
  # --- MOCK-MAP adaptation (G) ---------------------------------------------
  #
  # The sealed stage-4 artefact predates two fields this contract requires, and
  # the gap is structural rather than an oversight: it was a `KFO_MOCK=1` run,
  # so `apply` had nothing to apply and the premise question had not been asked
  # yet (M4.3.5 and M5.1.1 are both new). It also carries no `workset.yaml`
  # snapshot, because the merged `operator_workset` kind did not exist.
  #
  # So the mock **renders** what is missing from the workset that is actually
  # staged as this task's input, exactly as adaptation (A) renders
  # `environment.yaml` and (D) writes `env/steps.json`. Nothing is synthesised:
  # every number stays the sealed run's, and the rendered fields are copies of
  # the staged workset's own.
  #
  # A consequence worth stating, because it is the difference between the mock
  # running and the mock aborting: the premise now **holds**. Both sides of the
  # comparison are the environment record m1 minted for this run, so
  # `fixed.gpu_arch` matches itself. In the sealed run it did not — that was a
  # gfx942 workset against a gfx950 host — and a verbatim copy of it would
  # correctly abort m4 and stop the graph. `--var mock_premise=mismatched`
  # reproduces that on purpose, once, as the only cheap test of the abort path
  # we have.
  exec "${KFO_PYTHON:-python3}" "$PKG/assets/optimize_kernel.task/mock_adapt.py" \
      --handoff "${AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION:?}"
fi

echo "optimize_kernel: stage4-kernel-opt is not in E2E_MOCK_STAGES, and there is no" >&2
echo "program body for a real campaign: driving KernelForge, judging whether it" >&2
echo "degraded and deciding whether a result is real is judgement work and belongs" >&2
echo "to the kind: ai agent. Run the closure with its declared agent, or set" >&2
echo "--var mock_stages=all to take the mock path." >&2
exit 1
