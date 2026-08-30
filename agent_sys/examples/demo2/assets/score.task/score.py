#!/usr/bin/env python3
"""What `score` runs: every case through the one binary, folded into a score.

**This file imports nothing from `agent_sys`.** It is package data, run as a
subprocess by `agent.backends.program.ProgramExecutor`. `assets/lib/cpp.py` is
package data too.

| | |
|---|---|
| `AGENT_SYS_INPUT_HARNESS` | the one binary and its `items/manifest.json` |
| `AGENT_SYS_INPUT_EXTRA_TESTS` | the examiner's new cases |
| `AGENT_SYS_OUTPUT_SCORES` | where to write `README.md` and `items/` |

`readme.md` beside this file has the weights, the budgets and the reasoning.
Two things a reader of the code needs up front:

**Nothing is rounded.** `check_scores` recomputes every published figure from
the parts published beside it to within 1e-6, and a rounded number disagrees
with its own recomputation at exactly that scale.

**The binary is copied before it is run.** Whether the executable bit survives
`env_mgr`'s staging is not something a task body can assume, so it is copied to
a temporary directory and `chmod`ed there rather than in the input.
"""

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import cpp  # noqa: E402 — the path insert above is what makes it importable

#: The three components and their weights. A mapping rather than three floats
#: because it is published verbatim into the artefact: `check_scores` recomputes
#: against the weights the document declares, not against a copy of this table,
#: so there is one writer of the number and the validator reads it.
WEIGHTS = {"correctness": 0.7, "within_time": 0.2, "review": 0.1}

#: After this a case is killed. **This is `cpp.run`'s own default and not a
#: choice made here**: `check_cases(binary, cases)` takes no timeout argument,
#: so a caller cannot override it, and stating a different number in the
#: published document would be a figure nothing enforces. `cpp.run` reports the
#: kill as `TIMEOUT_RETURNCODE` with the elapsed budget, so a timed-out case is
#: a verdict rather than an exception — which is what lets one runaway
#: submission cost thirty seconds instead of the run.
RUN_TIMEOUT = 30.0

#: A case that finished but took longer than this is correct and not *within
#: time*. Two numbers rather than one because "wrong" and "slow" are different
#: things to tell a student, and folding them would make a slow correct solution
#: indistinguishable from a fast wrong one.
TIME_BUDGET = 2.0

#: How much of a mismatch is kept per failing case. Enough to see what happened;
#: not enough for one chatty program to dominate the artefact.
SNIPPET = 200

README = """# scores

## Purpose

Every submitted solution, run against every test case for its problem, folded
into one score per (student, problem) pair and one final per student.

{n_students} students, {n_problems} problems, {n_cases} case runs.

## Schema

`items/text.json` is a JSON object with four keys.

`weights` declares the arithmetic, and it is declared rather than assumed
because `check_scores` recomputes against **this** document's weights:

```
score = 100 * ({correctness} * correctness + {within_time} * within_time + {review} * review)
final = the mean of that student's per_student_problem.score
```

| component | meaning |
|---|---|
| `correctness` | the fraction of the pair's cases whose output matched |
| `within_time` | the fraction that finished within {budget}s |
| `review` | 1 if the reconciled review said `accept`, 0 otherwise |

A case that exceeded {timeout}s was killed; it counts as neither correct nor
within time.

`per_case` — one row per run: `case_id`, `student`, `problem_id`, `origin`
(`example` or `extra`), `index`, `ok`, `within_time`, `timed_out`,
`returncode`, `seconds`, and on a failure a short `got` / `expected` snippet.

`per_student_problem` — one row per pair: `cases`, `passed`, `in_time`, the
three fractions, and `score`.

`per_student` — one row per student: `problems` and `final`.

**Nothing is rounded.** Every figure is recomputable from the ones below it, to
within 1e-6, which is what `check_scores` tests.
"""


def _required(name: str) -> str:
    """A named refusal instead of a bare `KeyError`."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. A task body reads its inputs from "
            f"AGENT_SYS_INPUT_<KIND> and writes into AGENT_SYS_OUTPUT_<KIND>, "
            f"exported per slot by env_mgr.grants at dispatch."
        )
    return value


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise SystemExit(f"{path} does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid JSON: {exc}") from exc


def runnable_copy(binary: Path, workdir: Path) -> Path:
    """The binary, somewhere it is certainly executable.

    `env_mgr` stages a handoff's content into the consuming task's zone and this
    body has no guarantee about the mode bits that survive, nor any business
    writing into its own input. So: copy out, `chmod`, run the copy.
    """
    if not binary.is_file():
        raise SystemExit(
            f"{binary} does not exist. The harness manifest names it as the one "
            f"executable, so either the build did not ship it or the manifest is "
            f"out of step with the content."
        )
    target = workdir / binary.name
    shutil.copy2(binary, target)
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def extra_by_problem(content: Path) -> dict[str, list[dict]]:
    """The examiner's cases, indexed by problem id."""
    data = _load_json(content / "items" / "text.json")
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list):
        raise SystemExit(
            f"{content / 'items' / 'text.json'}: expected an object with a 'cases' list"
        )
    out: dict[str, list[dict]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        out.setdefault(str(case.get("problem_id")), []).append(
            {"input": str(case.get("input") or ""), "expected": str(case.get("expected") or "")}
        )
    return out


def examples_by_problem(manifest: dict) -> dict[str, list[dict]]:
    """The worked examples, indexed by problem id, out of the harness manifest.

    They reach this task through `harness` because `score` does not consume
    `problems` — see `assets/harness.task/readme.md` for why that route was
    taken and what it costs.
    """
    out: dict[str, list[dict]] = {}
    for entry in manifest.get("problems") or []:
        if isinstance(entry, dict):
            out[str(entry.get("problem_id"))] = [
                {"input": str(ex.get("input") or ""), "expected": str(ex.get("expected") or "")}
                for ex in entry.get("examples") or []
                if isinstance(ex, dict)
            ]
    return out


def verdict_by_pair(manifest: dict) -> dict[tuple[str, str], object]:
    """The reconciled verdict per `(student, problem_id)`. `None` where the two
    reviewers disagreed, which scores as 0 rather than being dropped."""
    return {
        (str(row.get("student")), str(row.get("problem_id"))): row.get("verdict")
        for row in manifest.get("reviews") or []
        if isinstance(row, dict)
    }


def snippet(text: str) -> str:
    text = text.strip()
    return text if len(text) <= SNIPPET else text[:SNIPPET] + "…"


def run_pair(binary: Path, unit: dict, cases: list[tuple[str, int, dict]]) -> list[dict]:
    """Every case for one (student, problem) pair, through the one binary.

    **The case id goes on `argv[1]`, and it has to.** It used to go on the first
    line of stdin, because `cpp.run` took no argv; the harness read it there with
    `fgets`, and every solution begins `std::ios::sync_with_stdio(false)`, which
    discards the position an earlier stdio read left. So the harness pulled the
    whole input into the C buffer and the solution's first `std::cin >> n`
    failed: empty stdout, return code 0, **every case wrong**. The first full
    demo2 run scored every student exactly 30.0 — `0.2*within_time +
    0.1*review`, with correctness zero — and every validator passed, because
    `check_one_binary` asks whether one executable dispatches and `check_scores`
    asks whether the arithmetic is reproducible, and both were true of a
    completely wrong result.

    `scratch/demo2-2026-08/probe_fgets_eats_stdin.py` measures seven programs:
    `fgets`, `getchar` and `getline` all starve a desynced solution, and all
    three are fine with a synced one. Reading stdin at all before the solution
    runs is the fault; the function is not the variable. There is no correct
    stdin fallback, so the id travels on argv and stdin carries only the case.
    """
    prepared = [{"input": case["input"], "expected": case["expected"]} for _, _, case in cases]
    results = cpp.check_cases(binary, prepared, argv=[str(unit["case_id"])])

    rows: list[dict] = []
    for (origin, index, _case), result in zip(cases, results):
        row = {
            "case_id": unit["case_id"],
            "student": unit["student"],
            "problem_id": unit["problem_id"],
            "origin": origin,
            "index": index,
            "ok": bool(result["ok"]),
            "timed_out": bool(result["timed_out"]),
            "within_time": not result["timed_out"] and result["seconds"] <= TIME_BUDGET,
            "returncode": int(result["returncode"]),
            "seconds": float(result["seconds"]),
        }
        if not row["ok"]:
            row["got"] = snippet(str(result["stdout"]))
            row["expected"] = snippet(str(result["expected"]))
        rows.append(row)
    return rows


def fold(rows: list[dict], review: object) -> dict:
    """The three fractions and the score, for one pair.

    A pair with no cases scores zero for correctness and for within-time and
    says `"cases": 0`. It is not omitted: an absent pair and a pair that scored
    nothing look identical downstream, and `check_scores` requires every pair
    the harness declared to be present.
    """
    total = len(rows)
    passed = sum(1 for row in rows if row["ok"])
    in_time = sum(1 for row in rows if row["within_time"])
    correctness = passed / total if total else 0.0
    within_time = in_time / total if total else 0.0
    review_score = 1.0 if review == "accept" else 0.0
    return {
        "cases": total,
        "passed": passed,
        "in_time": in_time,
        "correctness": correctness,
        "within_time": within_time,
        "review": review_score,
        "score": 100.0
        * (
            WEIGHTS["correctness"] * correctness
            + WEIGHTS["within_time"] * within_time
            + WEIGHTS["review"] * review_score
        ),
    }


def main() -> int:
    harness_content = Path(_required("AGENT_SYS_INPUT_HARNESS"))
    extra_content = Path(_required("AGENT_SYS_INPUT_EXTRA_TESTS"))
    dst = Path(_required("AGENT_SYS_OUTPUT_SCORES"))

    manifest = _load_json(harness_content / "items" / "manifest.json")
    if not isinstance(manifest, dict):
        raise SystemExit(f"{harness_content / 'items' / 'manifest.json'}: expected an object")
    units = [unit for unit in manifest.get("units") or [] if isinstance(unit, dict)]
    if not units:
        raise SystemExit("the harness manifest declares no units; there is nothing to score")

    examples = examples_by_problem(manifest)
    extras = extra_by_problem(extra_content)
    verdicts = verdict_by_pair(manifest)

    workdir = Path(tempfile.mkdtemp(prefix="score-"))
    try:
        # **`manifest["binary"]` is relative to `items/`, not to the content
        # root.** Joining it onto the content root instead gave
        # `<content>/codes/bin/harness`, which does not exist, and the run died
        # in `runnable_copy` three lines later with a message about the manifest
        # being out of step — blaming the producer for the consumer's join.
        # `check_one_binary` compares against `content/"items"` too; that is the
        # side that was right. Found by `scratch/demo2-2026-08/probe_harness/
        # probe_chain.py`, which is why the probe is kept.
        binary = runnable_copy(
            harness_content / "items" / str(manifest.get("binary") or "codes/bin/harness"),
            workdir,
        )

        per_case: list[dict] = []
        per_student_problem: list[dict] = []
        for unit in sorted(units, key=lambda u: (str(u.get("student")), str(u.get("problem_id")))):
            student = str(unit.get("student"))
            problem_id = str(unit.get("problem_id"))
            unit = {
                "student": student,
                "problem_id": problem_id,
                "case_id": str(unit.get("case_id") or f"{student}/{problem_id}"),
            }
            cases = [
                ("example", index, case) for index, case in enumerate(examples.get(problem_id, []))
            ] + [("extra", index, case) for index, case in enumerate(extras.get(problem_id, []))]
            rows = run_pair(binary, unit, cases)
            per_case += rows
            per_student_problem.append(
                {
                    "student": student,
                    "problem_id": problem_id,
                    **fold(rows, verdicts.get((student, problem_id))),
                }
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    per_student = []
    for student in sorted({row["student"] for row in per_student_problem}):
        mine = [row for row in per_student_problem if row["student"] == student]
        per_student.append(
            {
                "student": student,
                "problems": len(mine),
                "final": sum(row["score"] for row in mine) / len(mine),
            }
        )

    document = {
        "weights": {
            **WEIGHTS,
            "per_problem": ("100 * (0.7*correctness + 0.2*within_time + 0.1*review)"),
            "final": "mean of per_student_problem.score",
            "run_timeout_seconds": RUN_TIMEOUT,
            "time_budget_seconds": TIME_BUDGET,
        },
        "per_case": per_case,
        "per_student_problem": per_student_problem,
        "per_student": per_student,
    }

    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "items" / "text.json").write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )
    (dst / "README.md").write_text(
        README.format(
            n_students=len(per_student),
            n_problems=len({row["problem_id"] for row in per_student_problem}),
            n_cases=len(per_case),
            correctness=WEIGHTS["correctness"],
            within_time=WEIGHTS["within_time"],
            review=WEIGHTS["review"],
            budget=TIME_BUDGET,
            timeout=RUN_TIMEOUT,
        ),
        encoding="utf-8",
    )
    print(f"score: {len(per_case)} case runs over {len(per_student_problem)} pairs -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
