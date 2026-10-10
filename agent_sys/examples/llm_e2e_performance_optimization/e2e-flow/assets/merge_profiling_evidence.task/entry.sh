#!/bin/sh
# Fold both lines into stage 2's single export (M2.9).
#
# **No mock path, on purpose** — MOCK-MAP (H). The four inputs are mocked
# upstream when stage 2 is mocked, so this runs for real in every mode. Graded
# against a hand-shaped stand-in, `check_profiling_evidence`'s cross-part rules
# would be testing that the stand-in was shaped correctly, which is a test of
# the mock and not of the validator; `require_same_environment` in particular
# can only mean anything over parts that arrived separately.
#
# `#!/bin/sh` + `set -eu`, per CONTRACT §3.2a.
set -eu
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
exec python3 "$PKG/assets/merge_profiling_evidence.task/merge.py"
