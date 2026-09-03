# A terminal task gets 20 seconds for prepare, body and seal together

> ## SUPERSEDED IN ITS DIAGNOSIS, 2026-09-02 — read this box first
>
> **The 20-second window was the symptom. The cause was the package's own
> `items_schema`, and the framework's contribution is a refusal that nothing
> reads.**
>
> `analyze_packup`'s kind declared six items with `additionalProperties: false`;
> `packup.py` writes **ten**. `FilesystemStore.seal` runs
> `content.check_items` as an admission check and refused every version:
>
> ```
> items $: Additional properties are not allowed
> ('REPRODUCE.md', 'environment.md', 'notes.md', 'results' were unexpected)
> ```
>
> The four undeclared items are exactly the ones `check_analyze_packup_shape`
> **requires**. Producer, validator and kind had drifted apart, and the kind was
> the one nobody exercised — five of six handoffs in that package seal fine, so
> only the terminal one bit.
>
> **The chain, and note that no link in it names the schema:**
>
> | step | what happens |
> |---|---|
> | `seal` | refuses, and returns the reason **as a string** |
> | `_seal_outputs` | files it under `seal_refused` |
> | `seal_refused` | **has no reader anywhere outside tests** — written at `agent/runner.py:717`, lifted in `_LIFTED` at `:97`, read by nobody (verified) |
> | `agent/gate.py` | sees no published version → reports `output_absent` |
> | `monitor/pusher.py` | escalates `nothing to push` — the program body has exited, so there is no handle |
> | the CLI's sink | records the escalation and does not answer |
> | `_settle` | ends the run on its stall branch: `pack_analyze: running` |
>
> So the one message that names the cause is captured at step 1 and discarded,
> and the operator is shown a clock.
>
> **The framework defect worth fixing is therefore not the timeout — it is
> `seal_refused` having no reader.** A diagnostic that is produced and never
> surfaced is worse than one that is never produced, because its absence looks
> like the absence of a reason.
>
> **The check that gives the real answer in one command**, instead of
> suspecting the clock or the executable bit:
>
> ```python
> handoff.content.check_items(
>     content.load(D), content.content_type("<ctype>"), <your items_schema>)
> ```
>
> **Any handoff whose producer writes more items than its kind declares is
> exposed**, and `additionalProperties: false` is the default idiom across
> these packages.
>
> Fixed in that package by `acb8bfe`. Everything below remains true as a
> description of how the failure *presents*, and of why `--resume` cannot
> repair it.

**Found:** 2026-09-02 by `analyze-demo` on `crsuse2-m2m-019`, run
`20260902T091144-096985`. Verified first-hand by the leader against
`cli/main.py`.
**Severity:** **a complete, valid terminal handoff is lost** and the run reports
its last task as `running`. The output looks finished on disk; only the seal is
missing. `--resume` cannot repair it.
**Status:** reported, not fixed — `cli/main.py` is outside this round's agreed
scope and four runs were in flight.

## What happened

Five of six handoffs sealed with every validator passing:

```
check_kernel_table       PASS  usability       / strong
check_worklist_shape     PASS  completeness    / strong
check_identity_resolved  PASS  trustworthiness / strong
check_workset_shape      PASS  completeness    / strong
check_workset_runs       PASS  trustworthiness / strong
analyze_packup slot v0:  generating        <- never sealed
exit_code: 5
```

`build_workset` had succeeded after 1419 s and 67 turns. `verify_workset`
measured both operators cleanly. Then the terminal `pack_analyze` wrote its
whole output and the run ended.

## What it was not

Each ruled out by measurement, not by reasoning:

- **Not the executable-bit gate** (`agent/gate.py`, the documented cause in
  `003`): `items/command` is `-rwxr-xr-x`, mode 0755.
- **Not a slow body**: run against the very same sealed inputs it takes
  **0.444 s**, and its locality check 0.151 s.
- **Not an incomplete output**: 36 files written, and
  `check_analyze_packup_shape.check()` applied to the directory with the real
  args returns `PASS — 4 mandated file(s) present with substance`.
- **Not the settle ceiling**: the run carried `--timeout 7200` and lasted about
  35 minutes.

## What it was

`cli/main.py:911` — `_settle(..., stall_after: float = 20.0)`. A keyword
default. `_settle` is called at `cli/main.py:405` with `timeout=` only, so
`stall_after` is always 20.0, and **there is no CLI flag for it** (`--timeout`
exists; nothing exposes `stall_after`).

The timer runs from the last state change *anywhere in the run*, not from the
terminal task's own start. Zone mtimes:

| time | event |
|---|---|
| 09:38:41 | `verify_workset: succeeded`, `pack_analyze -> running`. **Last state change.** |
| 09:38:41 → 09:38:49 | zone prepare — **8 s**, staging four input handoffs and cutting the workspace |
| ~09:38:50 | body runs, writes 36 files |
| 09:39:00 | run ends. 09:38:41 + 20 = 09:39:01 |

So the real budget for a terminal step is **twenty seconds covering prepare,
body and seal together** — not the four-hour `--timeout`.

## Why every package's last step is exposed

Prepare scales with **how many input handoffs a task declares**, and a terminal
packup task declares the most by construction — it exists to gather everything.
`pack_analyze` declares four and spent 8 of the 20 seconds before its body
started.

The docstring says the branch fires when no status or attempt count has moved
*and no attempt holds a thread*, and `cli/main.py:891` notes that a model call
is deliberately counted as progress. So **an AI-bodied task is protected and a
program-bodied one is not**: a program body that does its work silently emits
nothing the stall detector can see.

Every one of these five packages ends in a program-bodied packup step.

## The second half of `temp/bugs/003`, now confirmed

`--resume` cannot repair it:

```
pack_analyze: failed — HandoffStateError: 818786a1 v0 is already open by task 73c1d79b
```

An interrupted task leaves its output slot open and the store refuses to reopen
it. A fresh run is the only route.

## How to recognise it in one command

If a packup step dies reporting `running` over an output directory that looks
complete:

```sh
ls -la <handoff>/content/items/command    # rules out the documented cause
```

Mode 0755 means it is this bug, not the gate.

## What would fix it upstream

Expose `stall_after` alongside `--timeout`, and start its clock at the task's
own transition rather than at the last global state change. The comment above
`_SETTLE_TIMEOUT` already argues that *"how long a task legitimately takes is a
property of the package, not of this file"* — that argument applies to
`stall_after` at least as strongly, and today the package has no way to say so.

## Workarounds available to a package

- Reduce what the terminal task must scan (`analyze-demo`: `top_n=1` halves
  both `build_workset` and the operator directories the seal walks).
- Declare fewer input handoffs on the terminal task, where the packup can read
  from one gathered handoff rather than four.
- Failing both, run the packup body out-of-band and validate it directly, and
  say so in `PROVENANCE.md`.
