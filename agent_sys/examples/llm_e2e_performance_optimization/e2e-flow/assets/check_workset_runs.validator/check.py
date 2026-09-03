#!/usr/bin/env python3
"""`check_workset_runs` — trustworthiness, strong.

The workset's own one-click correctness and performance entrypoints run, on this
hardware, and agree with the numbers it prints.

**This validator is the whole trust chain of the package** (CONTRACT.md §4.0),
so it is worth saying exactly what it holds up. m4 is told to take its ground
truth *strictly* from the workset and to **abort** rather than re-measure when
the premise does not hold — M4.3.5, reversed from the previous round's "do not
trust the workset's printed number". That instruction is only safe because
something has already run the workset's own tests on this hardware.

`build_workset` builds **and** measures — there is no separate `verify_workset`
task, because splitting build from measure across two agents is what M2.5
forbids. So the evidence in the workset is the producer's own claim about
itself. Grading the shape of that claim would make the chain a claim about a
claim.

**Therefore this validator re-measures.** It runs at least `reverify_shapes`
shapes through the workset's own `--shape` selector and checks its own number
against the recorded one. `build_workset` asserts a baseline; this confirms the
assertion; m4 may then divide by it. The same move `check_no_regression` makes
one stage later by recomputing instead of reading a verdict field.

Cost is `gpu_hours` and honest: one shape, five groups. `--shape` exists on the
entrypoints precisely so this check does not double the workset's GPU bill at
every seal.

**Two failures that are not failures of the workset, and are still failures.**
A node too busy to give a stable measurement fails `max_rsd`, and a host on
which the entrypoints refuse to run fails outright. Neither is a defect in the
artefact and both are correct verdicts: the artefact's claim is *"these numbers
hold on this hardware"*, and neither case establishes it. `todo.md` T10 records
that this and `min_pass_ratio` express opposite philosophies about forgiveness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import workset_io as W  # noqa: E402
import zone  # noqa: E402

#: How far this validator's own re-measurement may sit from the recorded one
#: before the record is called wrong rather than the node called busy. Wider
#: than `max_rsd` on purpose: the two runs are minutes and one co-tenant apart,
#: which is the very effect `todo.md` T7 measured at 12% between two arms that
#: were 1.1% apart under matched load.
_REVERIFY_TOLERANCE = 0.25


def _fail(problems: list[str], message: str) -> bool:
    problems.append(message)
    return False


def _check_reports(content: Path, document: dict, args: dict, problems: list[str], notes: list[str]) -> list[dict]:
    """Grade the recorded evidence and return the per-operator performance rows.

    Returns `[]` when the evidence cannot be read at all, which is a failure the
    caller has already recorded.
    """
    evidence = document.get("evidence")
    if not evidence:
        _fail(problems, "the workset carries no evidence block; there is nothing to confirm")
        return []
    try:
        perf = W.load_report(content, evidence["performance_report"])
        correct = W.load_report(content, evidence["correctness_report"])
    except (json.JSONDecodeError, OSError) as error:
        _fail(problems, f"the evidence does not load: {error}")
        return []

    if perf.get("impl") != "baseline":
        _fail(problems, f"the recorded performance report has impl {perf.get('impl')!r}, expected 'baseline'")
    if not correct.get("passed"):
        _fail(problems, "the recorded correctness report did not pass; a timing of a wrong kernel is not evidence")

    correct_by_id = {o["operator_id"]: o for o in correct.get("operators") or []}
    min_groups = W.arg_num(args, "min_groups", 5, int)
    min_iters = W.arg_num(args, "min_iters_per_group", 10, int)
    max_rsd = W.arg_num(args, "max_rsd", 0.10)

    for operator in perf.get("operators") or []:
        label = operator["operator_id"]
        if not operator.get("ran"):
            notes.append(f"{label}: did not run ({operator.get('failure') or 'no reason recorded'})")
            continue

        companion = correct_by_id.get(label)
        if companion is None:
            _fail(problems, f"{label}: timed, but the correctness report does not mention it")
        elif not companion.get("passed"):
            _fail(problems, f"{label}: timed, but its correctness failed ({companion.get('failure')})")

        for shape in operator["shapes"]:
            where = f"{label}/{shape['case_id']}"
            groups = shape["groups"]
            per_group = shape["per_group_ms"]
            if groups < min_groups:
                _fail(problems, f"{where}: {groups} measurement group(s), the protocol floor is {min_groups}")
            if len(per_group) != groups:
                _fail(problems, f"{where}: per_group_ms has {len(per_group)} entries for {groups} groups")
            if shape["iters_total"] < min_groups * min_iters:
                _fail(problems, f"{where}: {shape['iters_total']} iterations, below {min_groups}x{min_iters}")

            # Recompute rather than trust. The producer and this import the same
            # `weighted_mean`, so a stored figure that disagrees with the raw
            # numbers cannot come from a different formula — only from the record
            # having been edited after it was measured.
            recomputed = W.weighted_mean(per_group, shape["iters_total"])
            stored = shape["weighted_mean_ms"]
            if stored > 0 and abs(recomputed - stored) / stored > 0.01:
                _fail(problems, f"{where}: weighted_mean_ms is {stored:.6f}, the per-group figures give "
                                f"{recomputed:.6f}; the record disagrees with itself")
            recomputed_rsd = W.rsd(per_group)
            if abs(recomputed_rsd - shape["rsd"]) > 0.01:
                _fail(problems, f"{where}: rsd is {shape['rsd']:.4f}, the per-group figures give {recomputed_rsd:.4f}")
            if recomputed_rsd > max_rsd:
                _fail(problems, f"{where}: run-to-run spread {recomputed_rsd:.4f} exceeds {max_rsd}. The machine was "
                                f"not quiet, and an optimiser working against this baseline would chase noise")

    return list(perf.get("operators") or [])


def _reverify(content: Path, document: dict, recorded: list[dict], args: dict,
              problems: list[str], notes: list[str]) -> None:
    """Re-measure `reverify_shapes` shapes here, and compare.

    This is the step that makes the chain evidence rather than a claim about a
    claim. Picking the **primary** shape of each operator in turn: it is the one
    a headline number refers to, and the one m4 will divide by first.
    """
    wanted = W.arg_num(args, "reverify_shapes", 1, int)
    if wanted <= 0:
        _fail(problems, "reverify_shapes is 0; that turns this validator into a reader of the producer's own claim")
        return

    root = W.workset_root(content)
    entry = document["entrypoints"]["performance"]
    by_id = {o["operator_id"]: o for o in recorded}

    picked: list[tuple[str, str]] = []
    for operator in document["operators"]:
        primary = next((s for s in operator["shapes"] if s.get("is_primary")), None)
        if primary is not None:
            picked.append((operator["operator_id"], primary["case_id"]))
    picked = picked[:wanted]
    if not picked:
        _fail(problems, "no operator declares a primary shape; there is nothing to re-measure")
        return

    for operator_id, case_id in picked:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reverify.json"
            command = [*entry["cmd"].split(), "--operator", operator_id, "--shape", case_id, "--json", str(out)]
            try:
                finished = subprocess.run(  # noqa: S603 — the command comes from the artefact under test
                    command, cwd=root, capture_output=True, text=True,
                    timeout=int(entry.get("timeout_s") or 1800),
                )
            except subprocess.TimeoutExpired:
                _fail(problems, f"{operator_id}/{case_id}: the performance entrypoint timed out")
                continue
            except OSError as error:
                _fail(problems, f"{operator_id}/{case_id}: the performance entrypoint would not start: {error}")
                continue

            if finished.returncode != 0:
                tail = (finished.stderr or finished.stdout or "").strip().splitlines()[-3:]
                _fail(problems, f"{operator_id}/{case_id}: the performance entrypoint exited "
                                f"{finished.returncode}: {' | '.join(tail)}")
                continue
            if not out.is_file():
                _fail(problems, f"{operator_id}/{case_id}: the entrypoint exited 0 and wrote no --json report")
                continue

            try:
                fresh = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                _fail(problems, f"{operator_id}/{case_id}: the re-run's report is not valid JSON: {error}")
                continue

            measured = _one_shape(fresh, operator_id, case_id)
            claimed = _one_shape({"operators": [by_id.get(operator_id) or {}]}, operator_id, case_id)
            if measured is None:
                _fail(problems, f"{operator_id}/{case_id}: the re-run's report does not contain this shape")
                continue
            if claimed is None:
                _fail(problems, f"{operator_id}/{case_id}: the recorded evidence does not contain this shape")
                continue

            drift = abs(measured - claimed) / claimed if claimed > 0 else 1.0
            line = (f"{operator_id}/{case_id}: recorded {claimed:.4f} ms, re-measured {measured:.4f} ms "
                    f"({drift * 100:.1f}% apart)")
            if drift > _REVERIFY_TOLERANCE:
                _fail(problems, line + f" — beyond {_REVERIFY_TOLERANCE * 100:.0f}%. The recorded baseline does not "
                                       f"hold on this hardware, and m4 is about to divide by it")
            else:
                notes.append(line)


def _one_shape(report: dict, operator_id: str, case_id: str) -> float | None:
    for operator in report.get("operators") or []:
        if not operator or operator.get("operator_id") != operator_id:
            continue
        for shape in operator.get("shapes") or []:
            if shape.get("case_id") == case_id:
                return float(shape.get("weighted_mean_ms") or 0.0)
    return None


def _check(content: Path, args: dict, problems: list[str], notes: list[str]) -> bool:
    try:
        document = W.load_workset(content)
    except Exception as error:  # noqa: BLE001
        return _fail(problems, f"items/codes/workset.yaml does not load: {error}")

    recorded = _check_reports(content, document, args, problems, notes)
    if not recorded:
        return False
    _reverify(content, document, recorded, args, problems, notes)

    # `min_pass_ratio` forgives an operator that legitimately cannot run yet —
    # `identify`'s `agent_recovered` case — without letting the whole step block
    # the operators that are ready.
    ran = [o for o in recorded if o.get("ran")]
    ratio = len(ran) / len(recorded)
    floor = W.arg_num(args, "min_pass_ratio", 0.5)
    if ratio < floor:
        _fail(problems, f"{len(ran)} of {len(recorded)} operator(s) measured (ratio {ratio:.2f}, floor {floor})")

    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        notes: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems, notes)
        for note in notes:
            print(f"{hid}: {note}")
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_workset_runs: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
