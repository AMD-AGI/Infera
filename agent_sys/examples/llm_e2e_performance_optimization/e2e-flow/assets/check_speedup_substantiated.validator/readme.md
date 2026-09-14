# `check_speedup_substantiated`

A **program** validator (G4.1). The expensive half of m4's output gate —
`tags.cost: gpu_hours`, so a phase runs it after `check_optimization_shape`.

**This body is a reversal of the one it replaces.** `../../../kernel-opt-demo`'s
version re-measured the baseline here and disbelieved the workset's; its loudest
rule was *"最要命的那条：不要拿 workset 里印的数字当分母"*. Mission M4.3.5
overrules it:

> 这一点不成立，优化任务的 ground truth 本身就应该严格的从 workset 中来，如果
> 最基础的硬件、优化前提不一样，直接报错 abort。软件环境不太一样可以报 warning。

So the question changes from *"does your number reproduce against a baseline I
measured"* to *"did you take the workset's baseline, and did the premise hold"*.

## Criteria

Ordered as they run, and the free ones run first: a run whose premise did not
hold must not reach a timing loop, for the same reason a kernel that is wrong
must not.

| # | criterion | cost | verdict on failure |
|---|---|---|---|
| 1 | **the premise holds on every abort-level field** — each field in `abort_on_mismatch` is equal between `premise.workset_environment` and `premise.run_environment` | free | **ABORT**. Named field, both values |
| 2 | **the operator is one the workset defines** | free | ABORT |
| 3 | **the shapes measured are the shapes the workset declares**, and there are at least `min_shapes_measured` of them | free | ABORT |
| 4 | **every software difference is recorded** — a field in `warn_on_mismatch` may differ, and must then appear in `run_environment.warnings[]` or `premise.verdict.warnings[]` | free | FAIL. Differing is tolerated; hiding it is not |
| 5 | **the producer's own verdict agrees with the computed one** | free | FAIL — m5 reads `premise.verdict.held`, not this transcript |
| 6 | **correctness passed, on every declared case** | free | FAIL. Boolean, never a ratio |
| 7 | **the denominator is the workset's own**, exactly, per case, against the carried copy of the workset's performance report | free | FAIL — *this is where "prove you took the workset's own baseline" is proven* |
| 8 | **the workset's baseline reproduces here** within `baseline_agreement_tolerance` | GPU | **ABORT** — see below |
| 9 | **the optimised side re-measures** within `tolerance` of `measured.per_case_ms` | GPU | FAIL |
| 10 | **the claim clears the noise floor and is not over-claimed** — one-sided | GPU | FAIL |

## Three things this validator is easy to misread

**Criterion 8 is not the deleted rule coming back.** It re-runs the seed and
compares against the workset's figure, and when they disagree it **refuses**. It
does not substitute a denominator. That substitution is precisely what M4.3.5
struck out: it makes the report internally consistent and still answers a
different question. A disagreement here is the premise failing empirically, and
the answer to a premise failure is abort.

**The numerator is still measured, and that is why the cost tag is
`gpu_hours`.** The reversal is about the denominator. A producer's own claim
about the kernel it just wrote is exactly the claim worth re-running.

**Why taking the workset's number on trust is safe, and what would make it
unsafe.** `check_workset_runs` (m3, `gpu_hours`) executes the workset's own
entrypoints **on this hardware** before m4 starts, and again in m4's input
phase. The workset's printed baseline is therefore a number reproduced here, not
one carried in from another machine. **If that validator is ever weakened to a
shape check, this body has to go back to re-measuring the baseline itself.**
Nothing in the code can detect that happening; it is a standing dependency
between two validators, written here because there is nowhere else to write it.

## Why the abort, in one measured example

The 2026-09-02 run timed `B8_V151936` at **50.18 µs on gfx950** against the
workset's **55.40 µs on gfx942** — 9.6% apart. Divide one by the other and 9.6%
of newer silicon appears as a speedup, in a comparison that reads as entirely
legitimate and that nobody downstream can detect. That is far more likely than
an outright fabrication and it is worse, because it survives review.

## Three environment traps the body handles, all measured

1. **The interpreter.** A validator body is started by `/bin/sh` with a closed
   environment and no `PATH`, and on the output phase the PRODUCER row shadows
   the GLOBAL row carrying `AGENT_SYS_DEMO_PYTHON`
   (`kernel-opt-demo/bugs/002-…`). `"${AGENT_SYS_DEMO_PYTHON:-python3}"` then
   resolves to `/usr/bin/python3`, which has no `torch`, and the measurement
   dies on the import in 0.1 s — reported as a measurement disagreement. The
   body probes candidates and verifies each really has `torch`.
2. **`PATH` for the compiler.** A Triton kernel compiles `hip_utils.c` through
   `/bin/gcc`, which needs `as`, `ld` and `collect2`. The baseline side survives
   a missing `PATH` because `torch.softmax` compiles nothing, so the symptom is
   *only the optimised side fails* — which reads exactly like a broken kernel.
3. **`TMPDIR` pointing at a directory that does not exist** makes every HIP
   kernel launch segfault with no output while `torch.cuda.is_available()` still
   returns `True`. It cost the 2026-09-02 run 25 minutes.

`HIP_VISIBLE_DEVICES` is deliberately **not** defaulted: it arrives from the
agent spec's `env:` block, and inventing a default would move the measurement
onto card 0, which on a shared host is somebody else's.

## The driver has to have measured the file we handed it

m3's harness records `impl_read = {path, sha256, bytes, loaded_by}` **at the
moment it reads** (`_common.py:333`, `782bb08`); this body compares that digest
against the file it named on `--impl`, before any number in the report is read.

**`impl_path` cannot do this.** It is `args.impl` copied at parse time — an echo
of the request, identical whether the file was exec'd, imported, shadowed by
another copy of the same module name, or never opened. The stub's negative
control makes the difference visible: the recorded `path` is *the same path*
and only the digest differs.

Why it matters here specifically: `30_run_forge.sh` hands forge a copy of the
engine sources that is on **no interpreter's import path**. If the loader ever
resolved a module instead of exec'ing the file, forge would keep editing the
copy while the driver measured the container's untouched tree — and **every
ratio would come back ~1.0 with no error anywhere**, which is byte-identical to
an honest `improved: false`. A wrong answer that looks exactly like the right
one is the only kind this check exists for.

An absent or `null` `impl_read` is **not** a failure: that is what a baseline
run writes and what a pre-`782bb08` workset writes, and refusing those would
fail correct artefacts for being older than the check.

**Seeing it refuse:**

```sh
KFO_STUBKIT_WRONG_IMPL=1 python3 stubkit/run.py
```

The stub then hashes something other than the file it was given. Clean, all 8
cases pass; with the flag, every candidate case refuses and names both digests.
Five of the eight then report `verdict right but never said '<their own
phrase>'` — the refusal is correct in all of them and arrives before the one
each case was written for, which is what "refuses first" looks like from a kit
that grades wording.

## One assumption that is an `args` and needs m3 to confirm it

`workset.schema.json`'s `$defs/entrypoint` fixes `cmd` and `report` and says
nothing about **how a candidate implementation is selected**, while its
`$defs/performance_report` carries `impl` and `impl_path` — so a selector exists
and its spelling is m3's to declare. Until it is in the schema, the body drives
`<cmd> --report <path> --impl candidate --impl-path <file>` and every one of
those three flags is an `arg` (`report_flag`, `impl_flag`, `impl_path_flag`), so
a workset that spells it differently is re-pointed rather than silently
mis-driven. **This is the one place the body could be driving the workset
wrongly and not know it.**
