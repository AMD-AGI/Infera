#!/usr/bin/env python3
"""`check_scores` — completeness, strong.

Every (student, problem) pair is present, and **every published number is
recomputed from the numbers published beside it** under the weights the document
itself declares, to within `args.tolerance`.

That is the whole design of the check. It has no opinion about whether 70/20/10
is a good weighting; it establishes that the figures are the ones those weights
produce from those parts. A score sheet nobody can recompute is a claim.

Writes one boolean per handoff id in `inputs.json`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

COMPONENTS = ("correctness", "within_time", "review")


def declared_pairs() -> set[tuple[str, str]] | None:
    """Every pair the harness said it could run, or `None` if out of reach.

    `store.declared_dir("harness")` is `AGENT_SYS_INPUT_HARNESS`, exported
    because the producing task, `score`, consumes `harness`. It resolves on the
    PRODUCER row and not on the GLOBAL one — see `readme.md` beside this file.

    `None` is not a pass: the caller falls back to the pairs `per_case` and
    `per_student_problem` agree on, which is a weaker question honestly asked.
    """
    content = store.declared_dir("harness")
    if content is None:
        return None
    manifest_path = content / "items" / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, dict):
        return None
    return {
        (str(unit.get("student")), str(unit.get("problem_id")))
        for unit in manifest.get("units") or []
        if isinstance(unit, dict)
    }


def close(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def check(content: Path, tolerance: float, expected: set | None) -> tuple[bool, str]:
    """`(verdict, why)`. The reason is printed, so a failure names itself."""
    document = content / "items" / "text.json"
    if not document.is_file():
        return False, "items/text.json does not exist"
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"items/text.json is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return False, "items/text.json is not an object"

    weights = data.get("weights")
    per_case = data.get("per_case")
    per_pair = data.get("per_student_problem")
    per_student = data.get("per_student")
    if not isinstance(weights, dict):
        return False, "'weights' must be an object"
    for name in (
        ("per_case", per_case),
        ("per_student_problem", per_pair),
        ("per_student", per_student),
    ):
        if not isinstance(name[1], list):
            return False, f"{name[0]!r} must be a list"

    # The weights are read from the document rather than hard-coded here. There
    # is one writer of the number — `score.py` — and this recomputes against
    # what was actually published, so a change to the weighting is not also a
    # change to this validator.
    try:
        weight = {name: float(weights[name]) for name in COMPONENTS}
    except (KeyError, TypeError, ValueError):
        return False, f"'weights' must carry numeric {list(COMPONENTS)}; got {weights!r}"
    if not close(sum(weight.values()), 1.0, tolerance):
        return False, f"the weights sum to {sum(weight.values())}, not 1.0"

    # Fold `per_case` first, so that every figure above it is checked against
    # the rows rather than against another summary.
    counted: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(per_case):
        if not isinstance(row, dict):
            return False, f"per_case[{index}] is not an object"
        for field in ("ok", "within_time", "timed_out"):
            if not isinstance(row.get(field), bool):
                return False, f"per_case[{index}].{field} is {row.get(field)!r}, not a boolean"
        pair = (str(row.get("student")), str(row.get("problem_id")))
        tally = counted.setdefault(pair, [0, 0, 0])
        tally[0] += 1
        tally[1] += 1 if row["ok"] else 0
        tally[2] += 1 if row["within_time"] else 0

    seen: set[tuple[str, str]] = set()
    by_student: dict[str, list[float]] = {}
    for index, row in enumerate(per_pair):
        if not isinstance(row, dict):
            return False, f"per_student_problem[{index}] is not an object"
        pair = (str(row.get("student")), str(row.get("problem_id")))
        if pair in seen:
            return False, f"pair {pair} appears twice in per_student_problem"
        seen.add(pair)

        try:
            cases = int(row["cases"])
            passed = int(row["passed"])
            in_time = int(row["in_time"])
            parts = {name: float(row[name]) for name in COMPONENTS}
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"per_student_problem[{index}] is missing or mistyped a field: {exc}"

        tally = counted.get(pair, [0, 0, 0])
        if [cases, passed, in_time] != tally:
            return (
                False,
                f"pair {pair} claims cases/passed/in_time {[cases, passed, in_time]} "
                f"but per_case holds {tally}",
            )
        if cases:
            if not close(parts["correctness"], passed / cases, tolerance):
                return False, f"pair {pair}: correctness {parts['correctness']} != {passed}/{cases}"
            if not close(parts["within_time"], in_time / cases, tolerance):
                return (
                    False,
                    f"pair {pair}: within_time {parts['within_time']} != {in_time}/{cases}",
                )
        if parts["review"] not in (0.0, 1.0):
            return False, f"pair {pair}: review is {parts['review']}, not 0 or 1"

        recomputed = 100.0 * sum(weight[name] * parts[name] for name in COMPONENTS)
        if not close(score, recomputed, tolerance):
            return False, f"pair {pair}: score {score} != {recomputed} under the declared weights"
        if not 0.0 <= score <= 100.0:
            return False, f"pair {pair}: score {score} is outside [0, 100]"
        by_student.setdefault(pair[0], []).append(score)

    unscored = sorted(set(counted) - seen)
    if unscored:
        return False, f"per_case has rows for {unscored}, which per_student_problem does not score"
    if expected is not None and seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        return False, f"coverage: the harness declared pairs missing {missing}, unexpected {extra}"

    finals = set()
    for index, row in enumerate(per_student):
        if not isinstance(row, dict):
            return False, f"per_student[{index}] is not an object"
        student = str(row.get("student"))
        if student in finals:
            return False, f"student {student!r} appears twice in per_student"
        finals.add(student)
        mine = by_student.get(student)
        if not mine:
            return False, f"per_student has {student!r}, which has no per_student_problem rows"
        try:
            final = float(row["final"])
            problems = int(row["problems"])
        except (KeyError, TypeError, ValueError) as exc:
            return False, f"per_student[{index}] is missing or mistyped a field: {exc}"
        if problems != len(mine):
            return False, f"student {student!r} claims {problems} problems, scored on {len(mine)}"
        if not close(final, sum(mine) / len(mine), tolerance):
            return False, f"student {student!r}: final {final} != {sum(mine) / len(mine)}"
        if not 0.0 <= final <= 100.0:
            return False, f"student {student!r}: final {final} is outside [0, 100]"

    unreported = sorted(set(by_student) - finals)
    if unreported:
        return False, f"students {unreported} are scored per problem but have no final"

    note = "" if expected is not None else "; harness coverage NOT checked (see readme)"
    return (
        True,
        f"{len(seen)} pairs over {len(finals)} students, {len(per_case)} case rows, every "
        f"figure recomputed to within {tolerance}{note}",
    )


def main() -> int:
    tolerance = float(store.args().get("tolerance") or 1e-6)
    expected = declared_pairs()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid], why = False, "no staged content"
        else:
            results[hid], why = check(content, tolerance, expected)
        print(f"check_scores: {hid}: {results[hid]} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
