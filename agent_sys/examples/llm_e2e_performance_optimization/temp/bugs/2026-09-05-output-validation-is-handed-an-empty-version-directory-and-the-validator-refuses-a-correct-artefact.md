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

## Second instance, and it is worse: the same fault produced a PASS

**Found 2026-09-05 by running the recognition check over the run that "passed".**
`20260905T174633` (`w17m1e`) — the run used as the control above — has an empty
zone of its own:

```
zone      validation.74b535db….output_validation.773bf177
materials 0 files
args      []          (no parameters)
report    none written
verdict   {"dc334dbd-…": true}      <- PASSED, on nothing
```

**The handoff is `profiling_evidence`, and in the SAME RUN a different validator
was staged 45 files for it and passed on those.** So one zone got the content and
another got nothing, for the same handoff, in the same run.

**The validator is not special.** A control settles it: in `20260905T121310`
(`p6m1`), where every kind carried the injected `check_nothing`, a zone with the
**identical signature** — `args []`, no report — was staged **33 files**, and
**none of that run's 11 zones was empty.**

| run | validator signature | materials | verdict |
|---|---|---|---|
| `p6m1` | `args []`, no report | **33 files** | true |
| `w17m1e` | `args []`, no report | **0 files** | true |
| `w17m1g` | `check_profiling_evidence` | **0 files** | **refused** |

**So the fault is nondeterministic and validator-agnostic, and its visibility
depends entirely on which validator it lands on:**

- on a **strict** validator it produces a **false refusal** that reads exactly
  like a producer defect (`w17m1g`);
- on a **permissive** one it produces a **pass on nothing**, and is completely
  invisible (`w17m1e`).

**The second is the dangerous half.** A refusal at least stops the chain and gets
investigated. **A green from a validator that was handed an empty directory looks
identical to a green from one that read the artefact**, and nothing in the
verdict, the report, or the run log distinguishes them.

**Consequence for every result this effort has recorded: a PASS establishes that
the validator was invoked, not that it saw anything.** The only way to tell is
the file count under the zone's `materials/`, after the fact.

## Third instance, and it killed a healthy run in progress

**`r5m1b`, run `20260905T221356-9219d7`, 2026-09-05.** Found by team-lead, verified
here independently with the tool below.

```
EMPTY  validation.e2d93386-….output_validation.fdf55b49
         versions: v0   files: 0   validators: check_profiling_evidence
1 of 13 zone(s) were handed ZERO files.
```

**`check_profiling_evidence` refused mocked m2 from a zero-file zone.** The
refusal escalated, the escalation reached a sink that records and does not
answer, and 900 s later the run was declared over:

```
done  main is waiting on a decision no one will make … Nothing has changed for
      900 s; still in a phase: integrate_and_verify:running,
      m2_profiling:output_validating, m5_integration:running
```

**The two lines above that `done` are the point:**

```
handoff  integration_report slot v0: generating
```

**Stage 5's real, two-arm measurement was fifteen minutes in and actively
writing its report when a spurious refusal on a *mocked upstream stage* ended the
run.** The measurement was healthy. Nothing was wrong with any artefact.

**So the fault has now produced three distinct consequences, in increasing
severity:**

| where it lands | result | visible? |
|---|---|---|
| a strict validator | false refusal reading like a producer defect | yes, and misleading |
| a permissive one | pass on nothing | **no** |
| a strict one **upstream of live work** | **kills a healthy run in progress** | only via the `done` line |

**And the terminal line hides it.** The final `done` reads like an ordinary stop;
`tail -1` shows nothing about a validator, and the cause is in the lines above —
the same shape as the four "cause unknown" runs earlier the same day.

**Cost of this instance:** the fourth consecutive attempt to establish whether a
real `integration_report` carrying a real environment record passes
`check_environment`, and the node's remaining hold.

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

## Tooling

`assets/lib/refusal_saw_something.sh <run dir>` reports the file count for every
validation zone and exits non-zero if any was handed none. **Verified against
all three runs above**: it flags `w17m1g`'s refusing zone by name, flags
`w17m1e`'s passing one — which is how the second instance was found — and clears
`p6m1` at 11 zones, none empty.

## Not fixed here

`agent_sys/` outside `examples/llm_e2e_performance_optimization/` is out of this
effort's scope. Recorded with the controlled pair, both zone paths, both file
counts, and the run ids so whoever owns handoff staging has the whole case.

Pairs with `2026-09-04-a-retry-deadlocks-on-its-own-half-open-handoff-version.md`
— same `v0`/`v1` machinery, different symptom.
