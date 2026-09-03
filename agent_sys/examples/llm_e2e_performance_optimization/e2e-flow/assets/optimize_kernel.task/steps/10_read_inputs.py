#!/usr/bin/env python3
"""STEP 1 — read the inputs and pin the run. Nothing here is chosen.

Every later step reads `inputs.json` rather than the workset, so that "which
operator", "which entrypoint" and "which shapes" are decided exactly once. A
step that re-derived them could re-derive them differently, and the failure
would be a correctness suite and a timing loop running over different shapes —
which produces numbers and no error.

Acceptance (the readme's, enforced here):
  * exactly one operator,
  * a correctness entrypoint and a performance entrypoint,
  * at least `--min-shapes` performance shapes.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib as lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--operator", default=os.environ.get("KFO_WORKSET_OPERATOR") or None)
    ap.add_argument("--min-shapes", type=int, default=3)
    a = ap.parse_args()

    workset = lib.load_workset()
    operator = lib.pick_operator(workset, a.operator)
    operator_id = str(operator.get("operator_id"))
    entry = lib.entrypoints(workset, operator)

    problems: list[str] = []
    for role in ("correctness", "performance"):
        if not (entry.get(role) or {}).get("cmd"):
            problems.append(f"the workset declares no {role} entrypoint for {operator_id}")

    performance_shapes = lib.shapes(operator, "performance")
    correctness_shapes = lib.shapes(operator, "correctness")
    if len(performance_shapes) < a.min_shapes:
        problems.append(
            f"{len(performance_shapes)} performance shapes, the workset contract requires "
            f">= {a.min_shapes}"
        )
    if not correctness_shapes:
        problems.append("the workset declares no correctness shapes")

    # The baseline the whole stage divides by. Read here so that a workset whose
    # evidence is missing fails now, free, rather than after a campaign.
    evidence = workset.get("evidence") or {}
    baseline_rel = evidence.get("performance_report")
    baseline: dict[str, float] = {}
    if not baseline_rel:
        problems.append("the workset carries no evidence.performance_report; there is no ground truth to divide by")
    else:
        path = lib.workset_root() / str(baseline_rel)
        if not path.is_file():
            problems.append(f"the workset's performance report is missing at {baseline_rel}")
        else:
            baseline = lib.report_per_case_ms(lib.load_json(path), operator_id)
            missing = sorted(c for c in performance_shapes if c not in baseline)
            if missing:
                problems.append(f"the workset's own baseline has no figure for {missing}")

    if not isinstance((workset.get("ground_truth") or {}).get("dtypes"), dict):
        problems.append(
            "the workset carries no ground_truth.dtypes; dtype is on M4.3.5's abort list and "
            "cannot be compared without it"
        )
    if not operator.get("integration"):
        problems.append("the operator declares no `integration`; m5 cannot be a program without it (M5.1.1)")
    if not isinstance(operator.get("noise_floor"), (int, float)):
        problems.append(
            "the operator declares no numeric `noise_floor`. m4 must not pick one: a consumer "
            "choosing its own floor is a consumer choosing when to call its own result significant"
        )
    apparatus = list(operator.get("apparatus") or [])
    if not apparatus:
        problems.append("the operator declares no `apparatus`; the files that must travel are unknown")
    for relative in apparatus:
        if not (lib.workset_root() / str(relative)).is_file():
            problems.append(f"apparatus names {relative!r}, which is not in the workset")

    forge = operator.get("forge") or {}
    one_line = forge.get("one_line")
    if one_line and not (lib.workset_root() / str(one_line)).is_file():
        problems.append(f"the workset names a forge one-liner at {one_line} that is not there")

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    lib.write_json(
        Path(a.out),
        {
            "pinned_at": datetime.now(timezone.utc).isoformat(),
            "operator_id": operator_id,
            "workset_root": str(lib.workset_root()),
            "workset_id": workset.get("workset_id"),
            "entrypoints": {
                "correctness": entry["correctness"],
                "performance": entry["performance"],
            },
            "protocol": workset.get("protocol"),
            "ground_truth": workset.get("ground_truth"),
            "shapes": {"correctness": correctness_shapes, "performance": performance_shapes},
            "baseline_per_case_ms": baseline,
            "baseline_report": baseline_rel,
            "edit_target": operator.get("edit_target"),
            # M5.1.1's declared integration point. Distinct from `edit_target`:
            # that says where an optimiser edits, this says where a replacement
            # is installed and what it may not change.
            "integration": operator.get("integration"),
            # Declared by the workset and NEVER defaulted here -- a consumer
            # with a fallback floor is one that silently picks its own
            # significance threshold when the workset forgot to state one.
            "noise_floor": operator.get("noise_floor"),
            # The files that must travel byte-identically for the entrypoints to
            # run outside the handoff. A list, not a directory guess.
            "apparatus": list(operator.get("apparatus") or []),
            "gates": operator.get("gates"),
            "forge_one_line": one_line,
            "run_environment": lib.load_environment(),
        },
    )
    print(
        f"ok: operator {operator_id}, {len(performance_shapes)} performance shape(s), "
        f"{len(correctness_shapes)} correctness shape(s), baseline from {baseline_rel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
