# The merge did not fire — and the reason is a refusal message that names the wrong knob

**Written 2026-09-06T13:53:39Z by m35**, watching `20260906T130845-298750` live, read-only.

## What happened

```
13:46:13Z  run_profiling_mode_off: SUCCEEDED     <- first _off seal on this cluster
13:51:52Z  run_profiling_mode_on:  output_validating,  3 zones, 11 verdicts
13:52:05Z  ONE FALSE verdict, handoff 2d120113 = profiling_mode_on.profile_result
           merge_profiling_evidence: still waiting_handoff
```

**Materials check first: 39 files in that zone, none empty.** The refusal is
attributable.

## The refusal is IDENTICAL to `fdb0bd`'s — on a run that applied the fix

```
# check_trace_coverage
## 2d120113…: REFUSED
  note:    re-parsed …TP-2.trace.json.gz: 1221065 events, 69613 GPU kernels,
           10.939s — manifest agrees
  note:    4 rank(s), 278572 GPU kernel events
  PROBLEM: items/result/stacks_manifest.json is missing — the round was asked for
           a stack window and this handoff carries none … **Set --var
           stack_window_s=0 to say that is intended**
```

**This run DID pass `--var stack_window_s=0`.** Read from the live process,
`/proc/3308290/cmdline`: `stack_window_s=0`, `kernel_table_min_launchers=0`,
`expect_ranks=4`.

## The mechanism: the var reached the producer and not the validator

| | got `stack_window_s=0`? | evidence |
|---|---|---|
| **producer** (`replay.sh:123`) | **YES** | **no `capture_stacks.log` exists** — the capture was correctly not attempted |
| **validator** (`check_trace_coverage`) | **NO** | it still demands the manifest |

Because the validator does not read `stack_window_s` **at all**:

```
steps/m2_profiling.yaml:133     expect_stack_ranks: '${stack_ranks:-2}'
check_trace_coverage:223-225    want_ranks = int(args.get("expect_stack_ranks", 0) or 0)
                                if want_ranks <= 0: return True
```

> **The gate is `stack_ranks`, defaulting to 2. `stack_window_s=0` does not touch
> it.** So the producer skips the capture, the validator still expects the
> manifest, and the refusal is guaranteed.

## Why this is worse than the earlier instance of the class

`bug.record` §7c records a refusal naming `min_launchers_in_top_n` — the
validator's **args field** — when the `--var` is `kernel_table_min_launchers`.
That one names something that is not a `--var` at all, so a careful reader checks.

**This one names a real `--var` that really works — on a different component.**
`stack_window_s=0` is correct advice for the *producer* and useless for the
*validator emitting the advice*. **The message is self-defeating and reads as
authoritative**, and it cost this run the same refusal twice in one day.

The same yaml comment repeats it (`m2_profiling.yaml:131`): *"Set to 0 when
`--var stack_window_s=0` says the round is deliberately taken without it"* —
sitting **directly above** the line whose knob is `stack_ranks`.

## The fix for the next launch

```
--var stack_window_s=0        # producer: do not attempt the capture
--var stack_ranks=0           # validator: do not expect the manifest   <-- MISSING ALL DAY
--var kernel_table_min_launchers=0   # already being passed
```

**All three are needed. Two were passed. The third has never been passed on this
cluster.**

## Consequences

- **`merge_profiling_evidence` did not fire and will not on this run.** It stays
  `waiting_handoff`; `profiling_evidence` still has never existed.
- **The stack-window mechanism IS implicated**, but not as `fdb0bd`'s timing
  defect — that one was real and is separately fixed by `trace_end_ms`. **This is
  a second, independent defect on the same feature**: an incompletely wired
  opt-out.
- **My `ON-ARM-REFUSAL.md` §4 table is now incomplete.** It says
  `stack_window_s=0` makes both validators pass. **It does not** — it makes
  `check_kernel_table` pass (via `kernel_table_min_launchers=0`) and leaves
  `check_trace_coverage` refusing. Corrected here rather than there, because
  this file is the observation and that one is the analysis.

## What was NOT observed

**The merge itself.** It has still never executed anywhere. Everything above is
about why it did not get the chance.
