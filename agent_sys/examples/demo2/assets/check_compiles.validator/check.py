#!/usr/bin/env python3
"""`check_compiles` — completeness, strong.

Every `solution.cpp` in the handoff compiles with `g++ -O2 -std=c++17`, and
reproduces every example the `problems` artefact ships for the problem it is
filed under, inside the per-case wall-clock cap. One boolean per handoff id in
`inputs.json`; the per-problem detail goes to stdout so a reviewer can see
which one broke rather than only that something did.

**Why it uses `cpp.compile` and `cpp.run` and not `cpp.check_cases`.**
`check_cases` returns `list[dict]`, and the keys of those dicts are not part of
the frozen API — a body that guessed at `ok` versus `passed` would fold a
missing key as falsy and fail every handoff, or worse, read a truthy dict and
pass one. `compile` and `run` have fully specified return tuples, so the
comparison and the timing are done here where their rules are visible.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import cpp  # noqa: E402 — the path insert above is what makes it importable
import store  # noqa: E402

#: The default cap, overridden by the validator spec's `args`. It matches the
#: 30 s the students' readmes promise; the two are kept in one place precisely
#: so the prompt and the check cannot drift apart.
DEFAULT_TIMEOUT = 30.0


def normalise(text: str) -> list[str]:
    """Trailing whitespace per line dropped, then trailing blank lines dropped.

    Deliberately this much and no more. Comparing raw bytes would fail a
    correct program over a final newline; comparing tokens would pass a program
    whose output is the right numbers in the wrong shape, and the IO format is
    part of what the students were asked for.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def load_problems() -> dict[str, list[dict]]:
    """`problem id -> examples`, read from the `problems` artefact.

    Reached by `store.declared_dir("problems")`, which resolves
    `AGENT_SYS_INPUT_PROBLEMS` — the artefact the *producing task* consumed,
    not the newest `problems` anywhere in the store. A validator runs with the
    producer's resolved configuration, so this is the declared route and not a
    scan, and it works under confinement where a store scan does not.
    """
    root = store.declared_dir("problems")
    if root is None:
        return {}
    document = root / "items" / "text.json"
    if not document.is_file():
        return {}
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    rows = data.get("problems") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}
    out: dict[str, list[dict]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or f"p{position:02d}")
        examples = row.get("examples")
        out[pid] = (
            [e for e in examples if isinstance(e, dict)] if isinstance(examples, list) else []
        )
    return out


def check_one(directory: Path, examples: list[dict], timeout: float) -> tuple[bool, str]:
    """One problem directory: compile it, then run every example."""
    source = directory / "solution.cpp"
    if not source.is_file():
        return False, "no solution.cpp"
    if not examples:
        # Either the id names no problem in the artefact, or the problem ships
        # no example. Both mean nothing here was verified, and an unverified
        # submission is not a pass — the whole point of this validator is that
        # the claim was executed.
        return False, "no examples in the problems artefact for this id"

    with tempfile.TemporaryDirectory() as work:
        binary = Path(work) / "solution"
        ok, stderr = cpp.compile(source, binary)
        if not ok:
            first = (stderr.strip().splitlines() or [""])[0]
            return False, f"compile failed: {first}"

        for number, example in enumerate(examples, start=1):
            stdin_text = str(example.get("input") or "")
            # **`output`, and this said `expected` until `author-front` supplied
            # the frozen `problems` schema.** `cpp.check_cases` takes
            # `{"input", "expected"}` as *its own parameter* shape
            # (`cpp.py:124-153`), which is a different thing from what the
            # artefact calls the field; conflating the two would have read
            # `None` for every example and failed every handoff.
            expected = str(example.get("output") or "")
            try:
                rc, stdout, stderr, wall = cpp.run(binary, stdin_text, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - a timeout kill is a fail, not a crash
                return False, f"example {number}: run raised {type(exc).__name__}: {exc}"
            if rc == cpp.TIMEOUT_RETURNCODE or wall > timeout:
                # `cpp.run` does not raise on a timeout: it kills the child and
                # reports `TIMEOUT_RETURNCODE` with the elapsed time. Reported
                # as a cap breach rather than as "exit 124", which reads like
                # the program's own status and is not.
                return False, f"example {number}: took {wall:.1f}s, cap is {timeout:.0f}s"
            if rc != 0:
                first = (stderr.strip().splitlines() or [""])[0]
                return False, f"example {number}: exit {rc}: {first}"
            if normalise(stdout) != normalise(expected):
                return False, f"example {number}: output differs"
    return True, f"{len(examples)} example(s) passed"


def check(content: Path, problems: dict[str, list[dict]], timeout: float) -> tuple[bool, dict]:
    """The whole handoff. `all` over its problem directories, and empty is False.

    `all([])` is True, so a handoff carrying no solution at all would otherwise
    be indistinguishable from one where everything passed. That is the same
    shape as `dict.get` folding `None` as falsy, and it is refused here for the
    same reason.
    """
    codes = content / "items" / "codes"
    if not codes.is_dir():
        return False, {"<content>": "no items/codes directory"}
    directories = sorted(e for e in codes.iterdir() if e.is_dir())
    if not directories:
        return False, {"<content>": "items/codes is empty"}
    detail = {}
    for directory in directories:
        passed, why = check_one(directory, problems.get(directory.name, []), timeout)
        detail[directory.name] = ("ok: " if passed else "FAIL: ") + why
    return all(line.startswith("ok: ") for line in detail.values()), detail


def main() -> int:
    timeout = float(store.args().get("per_case_timeout_seconds") or DEFAULT_TIMEOUT)
    problems = load_problems()
    if not problems:
        print("check_compiles: the problems artefact yielded no problems")
    results = {}
    for hid in store.inputs():
        # `materials.json` first: it is the declared route and it carries the
        # handoff id, so nothing has to infer which staged copy belongs to
        # which input. `content_dir` is the fallback for a run with no
        # `env_mgr` wired, where nothing was staged.
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_compiles: {hid}: no published content")
            continue
        results[hid], detail = check(content, problems, timeout)
        for name, line in sorted(detail.items()):
            print(f"check_compiles: {hid}: {name}: {line}")
    store.write_verdict(results)
    print(f"check_compiles: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
