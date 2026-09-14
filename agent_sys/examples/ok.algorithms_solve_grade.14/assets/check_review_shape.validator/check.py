#!/usr/bin/env python3
"""`check_review_shape` — completeness, strong.

Total over each review document: every row carries every key `args.json` names,
`student` and `verdict` are drawn from the closed sets it names, the two
judgement fields are real JSON booleans, and no (student, problem) pair appears
twice.

**Coverage — one row per pair the students actually submitted — is checked when
this phase can reach the submissions, and reported as unchecked when it
cannot.** `readme.md` beside this file says which phase is which and why.

Writes one boolean per handoff id in `inputs.json`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

DEFAULT_ROW_KEYS = (
    "student",
    "problem_id",
    "implements_claimed_algorithm",
    "complexity_credible",
    "verdict",
    "comment",
)


def submitted_pairs(students: list) -> set[tuple[str, str]] | None:
    """Every `(student, problem_id)` the students actually submitted.

    Reached through `store.declared_dir("solutions_<s>")` — the **declared**
    route, `AGENT_SYS_INPUT_SOLUTIONS_A` and its two siblings, which are
    exported for the producing task's own input slots. So this resolves in the
    phase whose configuration is `review_x`'s or `review_y`'s, and not in
    `reconcile`'s input phase, which takes the GLOBAL row
    (`validator/phase.py:297-358`).

    `None` means *could not be determined*, which the caller must not fold into
    a pass. Absent evidence and satisfied evidence are the same shape and only
    one of them has been demonstrated.

    The problem id is the first path component below `items/codes/`, which is
    the only thing `scratch/demo2-2026-08/CONTRACT.md` §1 fixes about a
    solutions artefact.
    """
    pairs: set[tuple[str, str]] = set()
    for student in students:
        content = store.declared_dir(f"solutions_{student}")
        if content is None:
            return None
        codes = content / "items" / "codes"
        if not codes.is_dir():
            return None
        for entry in sorted(codes.iterdir()):
            if entry.is_dir():
                pairs.add((str(student), entry.name))
    return pairs


def rows_of(content: Path) -> list | None:
    document = content / "items" / "text.json"
    if not document.is_file():
        return None
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    reviews = data.get("reviews") if isinstance(data, dict) else None
    return reviews if isinstance(reviews, list) else None


def check(content: Path, args: dict, expected: set[tuple[str, str]] | None) -> tuple[bool, str]:
    """`(verdict, why)`. The reason is printed, so a failure names itself."""
    required = list(args.get("required_row_keys") or DEFAULT_ROW_KEYS)
    students = {str(s) for s in args.get("students") or ("a", "b", "c")}
    verdicts = {str(v) for v in args.get("verdicts") or ("accept", "revise")}

    rows = rows_of(content)
    if rows is None:
        return False, "items/text.json is missing, unreadable, or has no 'reviews' list"

    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return False, f"row {index} is not an object"
        missing = [key for key in required if key not in row]
        if missing:
            return False, f"row {index} is missing {missing}"
        student = str(row.get("student"))
        if student not in students:
            return (
                False,
                f"row {index} has student {row.get('student')!r}, not one of {sorted(students)}",
            )
        if str(row.get("verdict")) not in verdicts:
            return (
                False,
                f"row {index} has verdict {row.get('verdict')!r}, not one of {sorted(verdicts)}",
            )
        # `isinstance(x, bool)` and not truthiness: the string "false" is truthy
        # and would score as an accept, which is the wrong direction to be
        # lenient in.
        for field in ("implements_claimed_algorithm", "complexity_credible"):
            if not isinstance(row.get(field), bool):
                return False, f"row {index} field {field} is {row.get(field)!r}, not a JSON boolean"
        if not isinstance(row.get("comment"), str) or not row["comment"].strip():
            return False, f"row {index} has an empty comment"
        pair = (student, str(row.get("problem_id")))
        if pair in seen:
            return False, f"pair {pair} appears more than once"
        seen.add(pair)

    if expected is None:
        # Not a pass for the coverage question and not a failure either: it was
        # not asked. Reported so that a reader of the transcript can tell this
        # run apart from one where coverage was checked and held.
        return True, f"{len(rows)} rows, all well formed; coverage NOT checked (see readme)"
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        return False, f"coverage: missing {missing}, unexpected {extra}"
    return (
        True,
        f"{len(rows)} rows, all well formed, covering exactly the {len(expected)} submitted pairs",
    )


def main() -> int:
    args = store.args()
    students = list(args.get("students") or ("a", "b", "c"))
    expected = submitted_pairs(students)

    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            # A handoff with no published content is not a pass. `dict.get`
            # folding `None` as falsy is what DeepEval's unreached DAG node did,
            # and it reports identically to a real zero.
            results[hid], why = False, "no staged content"
        else:
            results[hid], why = check(content, args, expected)
        print(f"check_review_shape: {hid}: {results[hid]} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
