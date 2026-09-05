# Output validation is handed an empty version directory, and the validator refuses a correct artefact

**The refusal is accurate about what the validator was given and false about the
artefact.** `check_profiling_evidence` reported eight `PROBLEM:` lines of the
form *"items/result/… is missing"* for a handoff whose sealed content contains
every one of those files.

Recorded 2026-09-05 by m1. **Not a producer defect and not a launch-line error** —
both of which this looks exactly like, and both of which cost me runs earlier the
same day.

## The measurement, which is a controlled pair

Two runs, **same package, same corpus, same validator, same kind**, differing
only in `adhoc_cases` and `bench_rounds` — neither of which m2's merge reads:

| run | verdict | what the validator was handed | files |
|---|---|---|---|
| `20260905T174633` (`w17m1e`) | **PASS** | `materials/dc334dbd…/v1` | **45** |
| `20260905T194045` (`w17m1g`) | **REFUSE** | `materials/0704b267…/v0` | **0** |

```
$ find <w17m1g zone>/materials -maxdepth 2
    /0704b267-9883-43f3-b1d0-0c46ab3a6c14
    /0704b267-9883-43f3-b1d0-0c46ab3a6c14/v0
$ find <w17m1g zone>/materials -type f | wc -l
    0
```

And the handoff itself, in the same run:

```
v0   0 files
v1  47 files      items/env/parts.json, items/result/{bench_profiling_mode_off,
                  bench_profiling_mode_on,trace,kernel_table}/ ...
```

**The content the validator said was missing is present in `v1`, in the same
run, at the moment of the refusal.** The per-directory file counts are identical
to the run that passed — 11 / 11 / 11 / 7, 45 files under `content` in both.

## Why this is worse than an ordinary false refusal

**It is indistinguishable from a producer defect by reading the report.** The
report is well formed, names eight specific paths, and every statement in it is
true of the directory it was pointed at. Nothing in the verdict, the report or
the escalation says *"v0"*. The only way to tell is to list the validation
zone's `materials/`, which nobody does when a report already reads like an
explanation.

**It also falsifies the obvious next move.** My first hypothesis was that
`bench_rounds=1` had changed what m2's merge produced. That is a plausible,
testable, and completely wrong story that would have cost another 45-minute run
to disprove — and *disproving it by re-running would not have worked*, because
the fault is not deterministic in the launch line.

## What is not established

**Why one run staged `v1` and the other `v0`.** Both are `mock_stages=m2,m3,m4,m5`
on the same node against the same corpus. A race between the producer's seal and
the dispatch of output validation is the obvious candidate, and the timestamps
are *consistent* with it — `v1`'s newest file and `validation_failed` share the
millisecond `20:09:04.105` — but **mtimes in these trees are not trustworthy for
this question**: the corpus is copied with `cp -a`, so `v1`'s oldest file carries
a 10:31 mtime from nine hours earlier. **The timing evidence is suggestive and I
am not resting the diagnosis on it.** The `materials/` listing is what settles
the fact; the mechanism is open.

**Whether it can strike other kinds.** Only `profiling_evidence` was observed.
The staging is generic, so there is no reason to think it is specific to that
kind, and no evidence that it is not.

## What it cost

The chain stopped at `m2_profiling`'s output validation; `rank`'s input
validation failed with it, and stages 3–5 were never reached. **Fourteen of the
seventeen restored validators went uninvoked**, on a run whose whole purpose was
to invoke them.

## Corrected 2026-09-05, same day: `v0` is NOT the defect

**The first version of this record was titled "handed the empty `v0`" and that
framing is wrong.** Measured afterwards, in the run that *passed*
(`20260905T174633`), `check_acceptance`'s zone:

```
materials/238016f5…/v0      \
materials/4715253a…/v0       >  97 files
materials/a01a6c99…/v0      /
```

**`v0` is a normal staged version and can be fully populated.** In the same run
`profiling_evidence` was staged from `v1` and `stock/patched.measurement` from
`v0`, both correctly.

**So the defect is not the version number — it is being handed a directory with
ZERO files while the same handoff's other version holds 47.** A reader who
searched for "materials contains v0" would find it constantly and conclude
nothing. **The check is the file count, not the version.** Left in rather than
edited away, because the wrong version of this record circulated first and
someone may have read it.

## How to recognise it

**Before believing any refusal that says a file is missing, list the validation
zone's materials:**

```sh
z=$(find <run> -type d -name 'validation.*output_validation*' | head -1)
find "$z/materials" -maxdepth 2
find "$z/materials" -type f | wc -l
```

**A file count of zero means the validator was shown an empty version and its
refusal says nothing about the artefact. `v0` on its own means nothing** — see
the correction above; it is the normal staged version for several kinds.

## Not fixed here

`agent_sys/` outside `examples/llm_e2e_performance_optimization/` is out of this
effort's scope. Recorded with the controlled pair, both zone paths, both file
counts, and the run ids so whoever owns handoff staging has the whole case.

Pairs with `2026-09-04-a-retry-deadlocks-on-its-own-half-open-handoff-version.md`
— same `v0`/`v1` machinery, different symptom.
