# score — run every case, fold the weights, publish the arithmetic

The last step of `grade`, and the only one whose output leaves the subgraph.
It runs every test case through the one harness binary, one (student, problem)
pair at a time, and turns the results into a number per pair and a number per
student.

## The weights

| component | weight | what it is |
|---|---|---|
| correctness | **70%** | the fraction of this pair's cases whose output matched |
| within time | **20%** | the fraction that finished inside the time budget |
| review | **10%** | 1 if the reconciled review said `accept`, 0 otherwise |

```
per_student_problem.score = 100 * (0.70 * correctness
                                 + 0.20 * within_time
                                 + 0.10 * review)

per_student.final         = the mean of that student's per_student_problem.score
```

Both formulas are written into `items/text.json` under `weights`, and
`check_scores` **recomputes every published number from the parts published
beside it**, to within 1e-6. That is the property the whole document is shaped
around: a score sheet nobody can recompute is a claim, not a result. It is also
why nothing is rounded on the way out — a rounded figure and its own recomputation
disagree at exactly the tolerance the check uses.

The weights themselves are not defended here. Whether correctness should be 70%
rather than 60% is a course-design question, and no validator in this package
has an opinion about it. What is checked is that the number published is the
number these weights produce.

### The two budgets

| | |
|---|---|
| `run_timeout` | 30 s — after this a case is killed and counts as **timed out**, which is both wrong and out of time |
| `time_budget` | 2 s — a case that finished in longer than this is correct but not *within time* |

`run_timeout` is **`assets/lib/cpp.py`'s own default, not a choice made here**:
`check_cases(binary, cases)` takes no timeout argument, so a caller cannot
override it, and publishing a different figure would be a number nothing
enforces. `time_budget` is this body's, and is the one that separates *wrong*
from *slow*. Both are copied into the output, so they can be read without the
source.

## What it reads

| variable | what is in it |
|---|---|
| `$AGENT_SYS_INPUT_HARNESS` | `items/codes/bin/harness`, the one binary, and `items/manifest.json` |
| `$AGENT_SYS_INPUT_EXTRA_TESTS` | `items/text.json`, the examiner's new cases |

The harness manifest is where the worked examples and the review verdicts come
from. This task never consumes `problems` or `review` — `harness` does, and it
copies both through. `assets/harness.task/readme.md` records that decision and
what it costs.

### The binary is copied before it is run

`env_mgr` stages a handoff's content into the consuming task's zone, and
whether the executable bit survives a stage is not something this body can
assume. So the binary is copied to a temporary directory, `chmod +x`-ed there,
and run from there. The copy is thrown away afterwards; nothing is written back
into the input.

## Which cases run

For each (student, problem) pair in the manifest:

- every **worked example** the problems artefact shipped for that problem, and
- every **extra case** the examiner wrote for it.

Both are fed to the one binary as `<case input>` on stdin, with the case id
`<student>/<problem_id>` on the first line — the stdin form of the harness's
command line, which is what lets `assets/lib/cpp.py`'s `run()` drive it
unchanged. Output comparison is `cpp.check_cases`': stripped whole, then
stripped per line, which is what a competitive-programming judge does.

A pair with no cases at all scores 0 for correctness and for within-time, and
says so with `"cases": 0`. It is not skipped: a silently absent pair is
indistinguishable from a pair that scored nothing, and `check_scores` requires
every pair the manifest declares to be present.

## What it writes

Into `$AGENT_SYS_OUTPUT_SCORES`:

```
README.md         `## Purpose` and `## Schema`
items/text.json
```

```json
{
  "weights": {"correctness": 0.7, "within_time": 0.2, "review": 0.1,
              "per_problem": "100 * (0.7*correctness + 0.2*within_time + 0.1*review)",
              "final": "mean of per_student_problem.score",
              "run_timeout_seconds": 30.0, "time_budget_seconds": 2.0},
  "per_case": [{"case_id": "a/p1", "student": "a", "problem_id": "p1",
                "origin": "example", "index": 0,
                "ok": true, "within_time": true, "timed_out": false,
                "returncode": 0, "seconds": 0.004}],
  "per_student_problem": [{"student": "a", "problem_id": "p1", "cases": 12,
                           "passed": 12, "in_time": 12,
                           "correctness": 1.0, "within_time": 1.0, "review": 1.0,
                           "score": 100.0}],
  "per_student": [{"student": "a", "problems": 12, "final": 87.5}]
}
```

`per_case` carries no stdout. A failing case keeps a short `got` and `expected`
snippet so a reader can see *what* went wrong without the artefact growing by
the size of every program's output.

## Why this is a program and not an agent

Running a binary and folding three fractions is arithmetic. There is nothing a
model would do better and a great deal it could do inconsistently — and
inconsistency is precisely what `check_scores` exists to catch, so paying for it
here would be arranging the failure the validator is meant to find honestly.

## Layout

`entry.sh` is the command; `score.py` is the implementation; `assets/lib/cpp.py`
is the run-and-time helper it shares with `harness` and three validators.
