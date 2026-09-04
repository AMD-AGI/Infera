# `INVALID` means two things, and the run's exit code reads it as one

**Read 2026-09-04 in `agent_sys/`, from a question m2 raised and declined to
guess at.** Code read, not a run; the two decisive sites are quoted below.

`cli/main.py:1278` states the premise the CLI's exit code rests on:

> **`INVALID` and not "anything but `VALID`", and the limit is stated rather
> than hidden.** `INVALID` is a *sealed* negative verdict and is unambiguous.

**It is not unambiguous. Two different paths seal it, and they mean opposite
things about whether the run finished.**

## The two writers

**1. A validator refused.** `task_graph/runner.py:97`:

```python
verdict = HandoffStatus.VALID if valid else HandoffStatus.INVALID
```

The phase ran, every validator ran, one said no. **The run finished and produced
a negative result** — which is a complete outcome and often the correct one.

**2. The attempt died with the version still open.** `agent/runner.py:981`,
`_close_model_slot`, called from `run`'s `finally`:

> """Seal `INVALID` if the attempt ended with a version still open.
>
> The attempt is over and **nothing validated the output**, so the honest
> record is a hole."""

Nothing was graded. **The run did not finish.**

Both are honest records locally. The docstring for (2) is explicit that a hole
is the point. The defect is one level up, where a reader treats the two as one.

## What reads it, and what it then says

`cli/main.py:_completion_gaps` — *every task `SUCCEEDED`, no handoff `INVALID`*
— decides the run's exit code for a package with no declared expectations, and
`_strict` renders it as:

```
"this package promises no failure, and the run did NOT finish: handoff <kind>: invalid"
```

**For case (2) that sentence is true. For case (1) it is false.** A run that
walked every stage, sealed every handoff and had one validator correctly refuse
is reported as a run that *did not finish*.

**This is not hypothetical — it is this package's normal state.** Rung 0's mock
chain completes all five stages and `check_no_regression` refuses the corpus's
own `integration_report`, which is the corpus working as designed
(`cheat_for_mock/README.md` warned about it). The run exits **5** with *"the run
did NOT finish"*, and it did.

`assets/lib/accept_mock.py` and RUN-PLAN's *"Mock e2e green is a file and a
condition"* exist partly because of this: the exit code cannot express *finished,
and something refused*, so the acceptance claim had to move to the artefacts.
That was the right move regardless — a claim should name a file — but the reason
it was **necessary** is this ambiguity.

## Why it is the day's recurring class

The verdict is right and the explanation is false: something *is* wrong, exit 5
*is* the correct code, and the sentence naming the cause is not true. `todo.md`
T49 collects these; this one is distinctive because the false explanation is in
**the most visible line the CLI prints**, and because it survives checking —
*"did the run finish?"* has one answer and *"finished, or refused?"* has two,
and only the second question separates them.

## What would fix it, and none of it is ours

Ordered cheapest first; all are `agent_sys` changes, outside this package's
activity scope, and recorded rather than attempted (principle 6).

1. **Say less, accurately.** `_completion_gaps` names the gap already —
   `handoff <kind>: invalid`. The framing sentence is what overclaims. *"the run
   did not complete cleanly"* covers both cases and asserts neither.
2. **Distinguish at the source.** A version sealed by `_close_model_slot` could
   carry a different status, or a flag — *sealed without validation* is a
   different fact from *validated and refused*, and only the writer knows which.
   This is `todo.md` T29's missing third state, one level up: `verdict.json` is
   `dict[str, bool]` and cannot say *did not run*; `HandoffStatus` cannot say
   *sealed without being graded*.
3. **Read the verdicts.** A handoff with recorded verdicts was graded; one
   without was not. Available today without a framework change — `validation.yaml`
   carries `verdicts` — but it belongs in `_completion_gaps` rather than in five
   packages.

**A caveat against the obvious per-handoff shortcut**, which m2 falsified before
I could propose it: *"status ∈ {valid, invalid} ⇒ the phase completed"* is wrong.
Run `20260904T075753-e4f7ba` holds an `operator_workset` that is `invalid` with
**one validator of three** recorded — killed mid-validation, frozen that way on
disk. Status alone cannot carry the distinction in either direction.

## Provenance

m5's `replay_root` survey flagged `operator_workset` as having an unstable
validator set; m2 chased it, found the one killed run, and asked whether the
shutdown path marks handoffs `invalid` — noting that if it does, *"`invalid`
cannot carry the meaning `_completion_gaps` relies on."* It does. They stopped
rather than guess and routed it; this is the answer to their question.
