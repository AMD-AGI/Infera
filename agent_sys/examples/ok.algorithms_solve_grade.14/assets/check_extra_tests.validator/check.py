#!/usr/bin/env python3
"""`check_extra_tests` — completeness, strong.

Enough new cases per problem, **none of them a copy of a worked example the
problems artefact already ships**, and no empty field anywhere.

The duplication half is the one this validator exists for: an examiner that
retypes the shipped examples has added nothing, and the artefact on its own
cannot show that. It is reached through `store.declared_dir("problems")` — the
declared route to the very problems this examiner was given.

Writes one boolean per handoff id in `inputs.json`.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable


def normalise(text: str) -> str:
    """The comparison used for *is this the same input*.

    Stripped whole, then stripped per line — the same rule `cpp.check_cases`
    uses for output comparison, and for the same reason: re-indenting a shipped
    example does not make it a new case, and a judge that compared bytes would
    call it one.
    """
    return "\n".join(line.strip() for line in str(text).strip().splitlines())


def expected_count(args: dict) -> tuple[int | None, str]:
    """How many cases per problem are required, and where the number came from.

    **`$DEMO2_N_EXTRA` first.** It is what the `examiner` agent's `env` block
    resolved for this run, so `--var n_extra=2` shrinks the requirement in step
    with the instruction the examiner was actually given.

    It is present in the phase whose configuration is the producing task's, and
    absent in the one that takes the GLOBAL row (`validator/phase.py:297-358`).
    `args.default_n_extra` is the fallback, and it matches the agent's own
    default — so a full run agrees with itself and a shrunk one would not.

    `None` means *do not enforce an absolute count*, and the caller then falls
    back to requiring the same number for every problem, which is a real
    completeness property and does not depend on the environment. `readme.md`
    beside this file says why that is the honest fallback rather than reusing
    the default and failing a shrunk run for the wrong reason.
    """
    raw = os.environ.get("DEMO2_N_EXTRA")
    if raw:
        try:
            return int(raw), "$DEMO2_N_EXTRA"
        except ValueError:
            return None, f"$DEMO2_N_EXTRA is {raw!r}, which is not an integer"
    if store.declared_dir("problems") is not None:
        # The producer row: the environment is this examiner's, so the absence
        # of the variable is a real absence and the declared default applies.
        fallback = args.get("default_n_extra")
        if isinstance(fallback, int):
            return fallback, "args.default_n_extra"
    return None, "no count in the environment; uniformity checked instead"


def shipped_examples() -> dict[str, set[str]] | None:
    """The worked examples of each problem, normalised, or `None` if out of reach.

    `None` is not a pass. Absent evidence and satisfied evidence have the same
    shape, and folding the first into the second is how a check stops checking.
    """
    content = store.declared_dir("problems")
    if content is None:
        return None
    document = content / "items" / "text.json"
    if not document.is_file():
        return None
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    entries = data.get("problems") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return None
    out: dict[str, set[str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # `assets/problems.task/readme.md`'s shape — `{"problems": [{"id": ...,
        # "examples": [{"input", "output", "note"}]}]}` — read with the same
        # tolerance `harness.task/build.py` uses. `CONTRACT.md` §1 fixes only
        # that `problems` carries a `text.json`, so the shape is the setter's
        # readme and this crosses a boundary two authors own.
        problem_id = entry.get("id") or entry.get("problem_id") or entry.get("slug")
        raw = entry.get("examples") or entry.get("worked_examples") or entry.get("cases") or []
        out[str(problem_id)] = {
            normalise(ex.get("input")) for ex in raw if isinstance(ex, dict) and ex.get("input")
        }
    return out


def check(content: Path, args: dict) -> tuple[bool, str]:
    """`(verdict, why)`. The reason is printed, so a failure names itself."""
    document = content / "items" / "text.json"
    if not document.is_file():
        return False, "items/text.json does not exist"
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"items/text.json is not valid JSON: {exc}"
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        return False, "expected an object with a non-empty 'cases' list"

    by_problem: dict[str, list[str]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            return False, f"case {index} is not an object"
        # **`expected` is checked for presence, not for content, and the
        # distinction is the package's own.** An empty output is a legal answer
        # here: `problems.task/readme.md` requires each `output_format` to say
        # "what to print when the answer is empty or absent", and the problems
        # artefact ships worked examples that do exactly that.
        #
        # Measured 2026-09-02. A `merging-k-sorted-runs` problem specified *"If
        # the total number of values is zero, print an empty line: a single
        # newline character and nothing before it"* and shipped `output='\n'` as
        # a worked example. The examiner then wrote an extra case with
        # `expected='\n'` and a `why` explaining the k=0 edge — the correct
        # answer, in the format the problem demanded — and `not value.strip()`
        # rejected it as an unfilled field. `check_extra_tests` FAILED a correct
        # artefact and the graph stalled with four tasks unfinished.
        #
        # **The split is metadata against data, and it is the package's own.**
        # `problem_id` and `why` are what the examiner writes *about* a case and
        # can never legitimately be blank. `input` and `expected` are the case
        # itself, and whether an empty one is legal is the *problem's* to say —
        # not this validator's.
        #
        # A second instance, same run: an `edit-distance` problem whose
        # `input_format` reads *"Line 1: the string s. Line 2: the string t.
        # **Either line may be empty.**"* got an extra case with `input='\n\n'`
        # and a `why` of "both strings empty". `.strip()` made it `''` and the
        # case was rejected as unfilled. Two problems, two documented empty
        # edges, both refused.
        #
        # **`isinstance` still carries what this check is for**: an unfilled
        # field is absent or `None` and is caught; a field deliberately holding
        # an empty string is a string and is not.
        for field in ("problem_id", "why"):
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                return False, f"case {index} has an empty or non-string {field}"
        for field in ("input", "expected"):
            if not isinstance(case.get(field), str):
                return False, f"case {index} has a missing or non-string {field}"
        by_problem.setdefault(case["problem_id"], []).append(normalise(case["input"]))

    # Two cases with the same input for one problem is one case written twice,
    # and it is checkable without reaching outside this artefact.
    for problem_id, inputs in sorted(by_problem.items()):
        if len(set(inputs)) != len(inputs):
            return False, f"problem {problem_id!r} has two extra cases with the same input"

    required, source = expected_count(args)
    if required is None:
        counts = {len(v) for v in by_problem.values()}
        if len(counts) != 1:
            per = {k: len(v) for k, v in sorted(by_problem.items())}
            return False, f"uneven case counts across problems ({source}): {per}"
    else:
        short = {
            problem_id: len(inputs)
            for problem_id, inputs in sorted(by_problem.items())
            if len(inputs) < required
        }
        if short:
            return False, f"fewer than {required} cases ({source}) for {short}"

    shipped = shipped_examples()
    if shipped is None:
        return (
            True,
            f"{len(cases)} cases over {len(by_problem)} problems, all fields present, "
            f"count from {source}; duplication against the shipped examples NOT checked "
            f"(see readme)",
        )
    for problem_id, inputs in sorted(by_problem.items()):
        clash = sorted(set(inputs) & shipped.get(problem_id, set()))
        if clash:
            return (
                False,
                f"problem {problem_id!r} has {len(clash)} case(s) duplicating a worked "
                f"example already shipped with the problem",
            )
    unknown = sorted(set(by_problem) - set(shipped))
    if unknown:
        return False, f"cases keyed on problem ids the problems artefact does not have: {unknown}"

    return (
        True,
        f"{len(cases)} cases over {len(by_problem)} problems, count from {source}, "
        f"none duplicating a shipped example",
    )


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid], why = False, "no staged content"
        else:
            results[hid], why = check(content, args)
        print(f"check_extra_tests: {hid}: {results[hid]} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
