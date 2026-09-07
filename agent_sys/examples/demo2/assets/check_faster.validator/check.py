#!/usr/bin/env python3
"""`check_faster` — usability, weak.

Per problem directory in an `optimised` handoff: the source compiles, it answers
every worked example the `problems` artefact carries, and the improvement claim
is structurally supported — fewer non-blank non-comment lines than the baseline,
or a `new_algorithm` that differs from `baseline_algorithm`.

**`new_lines` is recounted here rather than read from `report.json`.** A field
the producer fills is a field the producer can fill wrongly, and this
validator's whole subject is a claim the producer made about its own work.
`baseline_lines` is still the producer's number — the baseline source is in a
different handoff and reaching for it would mean guessing at another kind's
item layout — and `readme.md` records that as one of the reasons this is `weak`.
"""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import cpp  # noqa: E402 — the path insert above is what makes it importable
import store  # noqa: E402

#: Every key `assets/optimise.task/readme.md` asks for. Listed rather than
#: inferred, because a report missing a field is the failure mode a spot check
#: on two of them would not see.
REPORT_KEYS = (
    "problem_id",
    "baseline_student",
    "baseline_lines",
    "new_lines",
    "baseline_algorithm",
    "new_algorithm",
    "what_changed",
    "why_faster",
)

#: `/* ... */`, possibly spanning lines. Removed before `//`, because `// */`
#: inside a block comment is not the end of anything.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//.*$", re.MULTILINE)


def source_lines(text: str) -> int:
    """Lines that are neither blank nor a comment.

    **Crude on purpose, and crude in a stated direction.** Comment markers
    inside a string literal are stripped as if they were comments, so a line
    holding only `puts("// hi");` is counted as blank and the count comes out
    *low* — which flatters the producer. A C++ parser would be exact, and would
    be a second compiler in a validator whose own headline claim is weak.
    Getting this to the nearest line is enough to tell 41 from 22 and is not
    enough to tell 41 from 40; the readme says so.
    """
    stripped = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
    return sum(1 for line in stripped.splitlines() if line.strip())


def worked_examples() -> dict[str, list[dict]] | None:
    """The worked examples from the `problems` artefact, keyed by problem id.

    Reached with `store.declared_dir("problems")` — the content directory of
    the handoff *this producer actually consumed*, named by the environment
    variable `env_mgr` exported for the producing task's own input slot. That
    is narrower than scanning the store for the newest `problems`, and it is
    the only one of the two that works under confinement.

    `None`, distinctly from `{}`, when the artefact could not be read at all:
    no examples to run is a check that did not run, and a check that did not
    run has not found nothing.
    """
    root = store.declared_dir("problems")
    if root is None:
        return None
    document = root / "items" / "text.json"
    if not document.is_file():
        return None
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("problems"), list):
        return None

    out: dict[str, list[dict]] = {}
    for problem in data["problems"]:
        if not isinstance(problem, dict):
            continue
        pid = str(problem.get("id") or "")
        cases = problem.get("examples")
        if not pid or not isinstance(cases, list):
            continue
        # `cpp.check_cases` wants `expected`; `problems` calls it `output`,
        # because a problem statement has an output and a test case has an
        # expectation. Renamed here rather than in either of them.
        out[pid] = [
            {"input": str(c.get("input", "")), "expected": str(c.get("output", ""))}
            for c in cases
            if isinstance(c, dict)
        ]
    return out


def check_problem(where: Path, examples: dict[str, list[dict]], build: Path) -> bool:
    """One problem directory: report, compile, run, and the improvement claim."""
    pid = where.name
    source = where / "solution.cpp"
    report_path = where / "report.json"
    if not source.is_file() or not report_path.is_file():
        print(f"check_faster: {pid}: needs both solution.cpp and report.json")
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"check_faster: {pid}: report.json is not valid JSON: {exc}")
        return False
    if not isinstance(report, dict):
        print(f"check_faster: {pid}: report.json must be a JSON object")
        return False
    empty = [k for k in REPORT_KEYS if k not in report or str(report[k]).strip() == ""]
    if empty:
        print(f"check_faster: {pid}: report.json is missing or empty at {empty}")
        return False

    cases = examples.get(pid)
    if not cases:
        # Either the producer invented a problem id, or it named a real problem
        # that carries no examples. Both leave this with nothing to run, and a
        # program nobody ran is not a program that works.
        print(f"check_faster: {pid}: no worked examples in `problems` for this id")
        return False

    binary = build / pid / "solution"
    ok, diagnostics = cpp.compile(source, binary)
    if not ok:
        print(f"check_faster: {pid}: did not compile:\n{diagnostics}")
        return False

    # 30 s is `cpp.run`'s default per case, and the readme promises it.
    for index, result in enumerate(cpp.check_cases(binary, cases)):
        if not result["ok"]:
            why = "timed out" if result["timed_out"] else f"rc={result['returncode']}"
            print(
                f"check_faster: {pid}: worked example {index} failed ({why})\n"
                f"  expected: {result['expected']!r}\n"
                f"  got:      {result['stdout']!r}"
            )
            return False

    # **The improvement claim.** Either arm is enough; both is what the task
    # asked for. `new_lines` is recounted, `baseline_lines` is the producer's.
    counted = source_lines(source.read_text(encoding="utf-8", errors="replace"))
    try:
        baseline_lines = int(report["baseline_lines"])
    except (TypeError, ValueError):
        print(f"check_faster: {pid}: baseline_lines is not a number")
        return False
    shorter = counted < baseline_lines
    rewritten = (
        str(report["new_algorithm"]).strip().casefold()
        != str(report["baseline_algorithm"]).strip().casefold()
    )
    if not (shorter or rewritten):
        print(
            f"check_faster: {pid}: neither shorter nor a different algorithm — "
            f"{counted} lines against a baseline of {baseline_lines}, both "
            f"{report['new_algorithm']!r}"
        )
        return False
    try:
        claimed_lines = int(report["new_lines"])
    except (TypeError, ValueError):
        claimed_lines = None
    if claimed_lines != counted:
        # Not a failure by itself: the claim is judged on the recount, so a
        # wrong `new_lines` cannot buy a pass. Printed because a producer whose
        # arithmetic is off is worth knowing about even when its work is good.
        print(
            f"check_faster: {pid}: report says new_lines={report['new_lines']}, "
            f"counted {counted}; judged on the count"
        )
    return True


def check(content: Path, examples: dict[str, list[dict]], build: Path) -> bool:
    codes = content / "items" / "codes"
    if not codes.is_dir():
        print("check_faster: no items/codes/ — the `code` content type requires it")
        return False
    problems = sorted(p for p in codes.iterdir() if p.is_dir() and not p.is_symlink())
    if not problems:
        # **`all(())` is `True`**, and an empty artefact passing every check in
        # this file would be the loudest possible instance of an instrument
        # pointed at the safe case. Nothing optimised is not everything
        # optimised well.
        print("check_faster: items/codes/ holds no problem directories")
        return False
    # Not short-circuited: every problem is checked so that one run's output
    # names every failure, rather than the first one and then silence.
    return all(check_problem(p, examples, build) for p in problems)


def main() -> int:
    examples = worked_examples()
    results = {}
    with tempfile.TemporaryDirectory(prefix="check_faster-") as tmp:
        for hid in store.inputs():
            # `materials.json` first: it is the declared route, and a body
            # reading it needs to know neither that a store exists nor where it
            # is. `content_dir` is the fallback for a run with nothing staged.
            content = store.staged_content(hid) or store.content_dir(hid)
            if content is None:
                # No content published. Not a pass: a check that could not run
                # has not found nothing.
                print(f"check_faster: {hid}: no published content")
                results[hid] = False
            elif examples is None:
                print(f"check_faster: {hid}: could not read `problems` — nothing to run")
                results[hid] = False
            else:
                results[hid] = check(content, examples, Path(tmp) / hid)
    store.write_verdict(results)
    print(f"check_faster: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
