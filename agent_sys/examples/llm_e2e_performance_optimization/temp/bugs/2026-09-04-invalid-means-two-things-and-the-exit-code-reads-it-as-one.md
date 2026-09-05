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

**The sealing in (2) is correct and should not be "fixed" — leaving the slot
`GENERATING` would be worse.** `agent/runner.py:952`:

> `INVALID` rather than leaving the slot `GENERATING`: a hole is the honest
> record, and **`open_next` refuses a slot someone else has open, so a
> re-dispatch would raise instead of appending `v+1`**.

A `GENERATING` slot left by a dead attempt would deadlock the retry. Stated
because the natural reading of this bug is *"stop sealing INVALID on the way
down"*, and that is the wrong end: **the defect is not that a hole is sealed, it
is that two meanings then share one value.** (The leader's point, filed
independently and folded in here.)

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

## A second, independent defect in the same function

`_completion_gaps` is **every task `SUCCEEDED`, no handoff `INVALID`**. The
first half of this file is about the second clause. **The first clause is
unsound too, for an unrelated reason.**

**Task status is never finalised.** Found by m5 building a gate m2 proposed,
verified by m2, verified by the leader, and reproduced here independently over
every run in the runroot:

```
runs with task records:            36
runs where EVERY task succeeded:    0
runs where `main` is not succeeded: 36

most common non-succeeded residue:
   13x  {'output_validating': 1, 'running': 2, 'waiting_handoff': 1}
    8x  {'running': 3, 'waiting_handoff': 4}
    8x  {'output_validating': 1, 'running': 2, 'waiting_handoff': 2}
```

Thirteen runs sharing a byte-identical residue makes it **systematic rather than
truncation**, and `main` — the run's own root task — is non-terminal in all 36.
The status is a live field that nothing rewrites on exit.

**What that does to the sentence.** The first half of this file shows *"the run
did NOT finish"* is false for a correctly-refused handoff. This shows the
predicate returns gaps **for a clean run too** — so the CLI has never once
reported a run as finished, and the line could not have been true of anything.
Everyone on this team has been reading it all day as though it discriminated.

**Kept in one file rather than two**, on m2's reasoning: separating them would
let a reader believe one was fixed. Two independent defects, one function, and
it is the CLI's exit code.

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
3. **Read the verdicts, which is available today.** A handoff with recorded
   verdicts was graded; one without was not. **Count the validators attached in
   the handoff's own `validation.yaml`** — CONTRACT §4.5, the only place a
   validator's name survives. `075753`'s `operator_workset` shows one of three,
   the signature of a phase that died; a refused-but-complete handoff shows all
   of them. It belongs in `_completion_gaps` rather than in five packages, but a
   survey can use it now.

   **This makes m5's *"the validator set is part of the verdict"* the
   discriminator for a framework defect**, hours after they wrote it as a
   discipline about rulers. A verdict without its validator set cannot say
   whether it is the whole answer.

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

---

## The contrast pair, and it is what makes the ambiguity concrete — m5

`075753` above shows one meaning. **This run shows the other, with the same
status**, and the two side by side are the whole defect:

| run | handoff | status | validators recorded | what it means |
|---|---|---|---|---|
| `20260904T075753-e4f7ba` | `operator_workset` | `invalid` | **1 of 3** | killed mid-validation; nothing graded it |
| `20260904T125637-e1ddf6` | `deploy_kit` | `invalid` | **3 of 3**, `check_deploy_serves: false` | every validator ran and one refused |

The second is the correctly-refused `max-bs 8` deployment — **a complete record
and the most informative row in a survey.** Any rule that excludes `invalid` to
skip the first also discards the second.

## Why no per-handoff repair is available either, measured

The obvious fallback is to stop asking about handoffs and ask about the **run**:
*was it still going?* That does not work, for a reason unrelated to `INVALID`:

**The task store's `status` is never finalised.** Measured across **36 runs**,
every one of them — including runs that finished cleanly — is left showing
`running: 2` and `output_validating: 1`, because task state is a live field
nobody rewrites on exit:

```
20260904T143952-bec7da  {succeeded 13, output_validating 1, running 2, waiting_handoff 1}   clean
20260904T075753-e4f7ba  {succeeded  8, output_validating 1, running 2, waiting_handoff 2}   killed
```

**Indistinguishable.** A filter built on it excluded **36 of 36 runs**, which is
how it was found. So *"did this run finish"* has one answer on disk —
`_completion_gaps` — and it is at run granularity, which is one level coarser
than the question a per-handoff survey asks.

## What `replay_root` does instead, and the general form

It reports the **distribution** of validator sets rather than resolving it:

```
unstable operator_workset   13x [check_environment,check_workset_runs,
                                 check_workset_shape] | 1x [check_environment]
```

A 20-to-1 split is self-evidently an outlier; a 10-to-11 split is a real
divergence. Nothing is inferred from a proxy and nothing informative is
discarded.

**A tool that cannot get a fact should show the shape and say so, rather than
infer the fact from a signal that does not carry it.** That is this record's
practical consequence for anyone else who reaches for `INVALID`, and it is the
reason the entry is worth more than the fix would be.
