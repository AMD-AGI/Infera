#!/usr/bin/env bash
# Fold both lines into stage 2's single export (M2.9).
#
# **No mock path, on purpose.** The four inputs are mocked upstream when stage 2
# is mocked, so this runs for real in every mode. `MOCK-MAP.md` maps
# `profiling_evidence` onto the sealed `profile_packup`, which is a `code`
# handoff and would need reshaping into a `reproducible` one — but a hand-shaped
# export would make `check_profiling_evidence`'s cross-part rules grade a
# stand-in, and those rules are the only reason this handoff exists rather than
# four. Merging for real over mocked inputs exercises them instead.
set -euo pipefail
PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
exec python3 "$PKG/assets/merge_profiling_evidence.task/merge.py"
