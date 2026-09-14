# check_scores — completeness, strong

Every (student, problem) pair is scored, and **every number in the document is
recomputed from the numbers published beside it**, to within `args.tolerance`.

## The one idea

A score sheet nobody can recompute is a claim. So nothing here is read and
believed: each figure is derived again from the level below it, and the check
fails on a disagreement larger than 1e-6.

```
per_student_problem.cases / .passed / .in_time   <- counted from per_case
per_student_problem.correctness                  <- passed / cases
per_student_problem.within_time                  <- in_time / cases
per_student_problem.score                        <- 100 * Σ weightᵢ · componentᵢ
per_student.final                                <- mean of that student's scores
```

**The weights come from the document, not from this file.** `score.py` writes
them into `items/text.json`, and this validator recomputes against what was
published. There is therefore one writer of the weighting, and changing 70/20/10
to something else is not also a change to this validator — which is what stops
the two from drifting apart silently.

This check has **no opinion about the weighting itself**. Whether correctness
should be worth 70% is a course-design question and nothing in this package
answers it.

## What else it checks

| | |
|---|---|
| the weights | numeric, all three present, and summing to 1 |
| `per_case` | `ok`, `within_time` and `timed_out` are real JSON booleans |
| duplicates | no pair twice in `per_student_problem`, no student twice in `per_student` |
| both directions | a pair with `per_case` rows and no score fails; a student scored per problem with no final fails |
| `review` | exactly 0 or 1 — it is a verdict, not a fraction |
| range | no `score` and no `final` outside `[0, 100]` |
| coverage | when reachable, exactly the pairs the harness manifest declared |

The both-directions test is the one that catches the mistake that actually
happens: a scorer that silently omits a pair produces a document that is
internally consistent and is missing a student's worst problem.

## The coverage limit

*Every pair the harness declared* needs the harness manifest, and this
validator's input is the score sheet. It is reached through
`store.declared_dir("harness")` — `AGENT_SYS_INPUT_HARNESS`, exported because
the producing task, `score`, consumes `harness`.

That resolves on §8.2's PRODUCER row and not on the GLOBAL one:
`validator/phase.py:297-358` supplies `bound`, `producer` and `global_` and no
`consumer`, so an *input* phase falls through to `cli/main.py:601`'s four
variables, which carry no `AGENT_SYS_INPUT_*`. Concretely: the coverage check
runs in `score`'s output phase, and not in `grade`'s.

When it cannot run, the transcript says `harness coverage NOT checked` and the
weaker question is still answered in full: `per_case` and `per_student_problem`
must cover exactly the same pairs, in both directions. It is **not** folded into
a pass — absent evidence and satisfied evidence have the same shape.

## What it does not check

Whether a solution deserved its score. That would mean re-running every case,
which is `score`'s job and would make two things owners of one fact. This
checks the bookkeeping, which is the part that can be checked exactly.

It also does not check that `per_case` reflects what the binary actually did.
`score.py` writes both the case rows and the summary, so a scorer that lied
consistently would pass here. The validator that stands between the binary and
the case rows is `check_one_binary`, which establishes that the binary runs and
discriminates; nothing re-runs the cases.

`strong` covers what the pass claims: the arithmetic is total over the document
and there is nothing approximate in it.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json`, `inputs.json` and `materials.json` beside it,
and it writes `verdict.json` — one boolean per handoff id in `inputs.json`.
