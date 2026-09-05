# A failing program task is recorded `succeeded` when its outputs exist — and unescalatable when they do not

**Found** 2026-09-05 by readme-cn, diagnosing *"`merge_profiling_evidence` never
starts its body"*. It starts. Four stages had already failed silently upstream.

**This builds on m4's
`2026-09-05-a-failed-body-has-its-debris-sealed-and-validated-and-says-nothing.md`
and does not restate it.** Their §1 establishes the mechanism — `_seal_outputs()`
runs unconditionally before `_gate(result)` (`agent/runner.py`, the block whose
comment reads *"Before the gate, and that ordering is the whole ruling"*), so a
failed producer's partial output becomes a handoff. **That is theirs.**

What is here is what happens **after** that seal, in the two directions the
outcome can go, and why the pair loses the diagnosis either way.

---

## The asymmetry

> **A program body's non-zero exit is *discarded* when its declared outputs are
> present, and *unescalatable* when they are not.**

| | outputs present | outputs absent |
|---|---|---|
| what is recorded | `outcome: succeeded` | `output_absent`, task stays `running` |
| is the reason kept? | no — stderr goes nowhere | **yes**, in `attributes.detail` |
| who finds out | nobody | nobody: the escalation has no recipient |

**m4's case is the left column with validators ON**, and there the debris was
caught — three `strong` refusals on the sealed v1. **With validators off it is
not caught at all**, and that is the new fact: the task is marked `succeeded` and
the run walks on.

---

## Measured, run `20260905T110109-b2e7af` (`--package e2e-flow-noval`)

`deploy_and_prove`, `agent_spec: runner`, mock path:

```
store/task   outcome: succeeded    11.0 s    deploy_kit v1 delivered
store/event  phase_done RUNNING 11:01:20.677 — no failure event of any kind
find $RUN/handoffs -name environment.yaml  ->  0
```

The body's own log, at `<zone>/tmp/m1_mock_adapt.359567.log`, 417 bytes:

```
mock_adapt: could not read a digest for
  'infera/engine-sglang:test-local-mooncake_hip_dmabuf1' on the node.
  ... Pass --var image=<an image on the node>, or set MOCK_IMAGE_ID explicitly
```

`mock_adapt.sh:70-72` reads the digest **from the node**; on a login node that
returns empty and `:74-80` refuses. **Reproduced standalone, login node, seconds:**

```
cp -a <corpus>/stage1-deploy/deploy_kit/content/. $T/
E2E_IMAGE='infera/engine-sglang:test-local-mooncake_hip_dmabuf1' \
  bash assets/deploy_and_prove.task/mock_adapt.sh $T
  -> MOCK_ADAPT_RC=3, byte-identical message, no environment.yaml
```

So the refusal is correct, loud, names its own fix — and the task that contains
it is `succeeded`.

**One step observed only in code, marked as such:** `entry.sh` does
`… || arc=$?` then `if [ "$arc" != 0 ]; then … exit "$arc"; fi`. I measured
`mock_adapt.sh` exiting **3** and the task recorded **succeeded**; I did not
separately instrument `entry.sh`'s own exit. The four lines are unambiguous, but
that is a read, not a measurement.

## It compounded across four tasks in one run

`mock.sh` copies the sealed corpus into `$AGENT_SYS_OUTPUT_*` **first**, so every
one of these had its declared output present before it failed:

```
deploy_and_prove        mock_adapt refused (rc 3)                    -> succeeded
run_profiling_mode_off  env_render --inherit <missing>, set -eu      -> succeeded
run_profiling_mode_on   same                                          -> succeeded
merge_profiling_evidence  nothing copied; it BUILDS its output       -> output_absent
```

**Four bodies failed. Three were recorded as successes.** The only one that
surfaced anything is the one with nothing pre-copied to seal — and its reason
survives *only* in `store/event`'s `attributes.detail`:

```
exit 1: merge: …/v0 carries no environment record at items/env/environment.yaml
escalated  why = "nothing to push: the executor is a program body:
                  there is no agent to instruct"
```

That escalation reaches the top with `target: user` and nobody receives it
(`2026-09-04-an-escalation-with-no-recipient.md`), so the task sits `running`
until `--timeout`.

**The diagnosis was inverted by this.** The visible symptom was one task
apparently never starting; the actual event was four stages of correctly-detected
failure being thrown away, and the one task that *could* not be thrown away
looking like the culprit.

## Why it matters more now than yesterday

`make_debug_package.py` exists so a chain can walk before verdicts are argued
about, and it is the right tool. **But with every validator replaced by
`check_nothing`, the left column of the table above has no backstop at all.** m4's
three refusals are what caught the same mechanism a run earlier. A `-noval` run
therefore cannot distinguish *walked* from *walked over four silent failures*, and
any claim from one must say so.

## What would make it visible, in this package's own terms

**Tier 2, and one instance already exists.** `deploy_and_prove/entry.sh` tees
`mock_adapt.sh` to `<zone>/tmp/m1_mock_adapt.$$.log` *specifically because* "a
task body's stderr goes nowhere a person can read" — and that file is the only
reason this was diagnosable. **It is the fix, present in exactly one body and
absent from every other.** A body that writes its own log is checkable; a rule
that says "read the artefact" is not.

Framework-side this is a design consequence rather than an oversight (m4's §1
quotes the ordering's justification), so the honest ask is not to reorder the
seal but to make a body's non-zero exit **reachable** — `program.py:172-181`'s
`_detail(code)` already captures what the body said, and on this path it reaches
nothing a reader sees.

## Workaround, and it produced the first full five-stage walk

Set **`MOCK_IMAGE_ID=sha256:<a real digest>`**, or pass `--var image=` naming an
image whose digest is readable where the body runs. Applied by the leader with
the digest from the 217 chain's sealed kit: run `20260905T111302-99008d` walked
**all five stages in under four minutes on the login node**, 15 sealed handoffs,
18 `environment.yaml` where there had been 0. `m5_integration` had never reached
`succeeded` in 45 runs. **Validation was disabled for that run**, so it is reach,
not correctness.

## Two negatives, recorded so nobody re-runs them

- **No live `claude` process has a cwd under `…-cc5813`.** The
  "agent still writing when the seal happened" hypothesis is unsupported there.
- **`env_render.py` is healthy.** `--inherit <missing>` exits 1 with a traceback;
  `--new` with an incomplete record exits 1 saying *"nothing was written: an
  environment record that does not validate is worse than none"*; `--new` with
  valid `E2E_*` exits 0 and writes the file. It never returns 0 having written
  nothing — a hypothesis that survived two rounds before being tested directly.

*A first attempt at the isolation above was denied by the permission system for
containing `rm -rf`. It was re-run against a fresh timestamped directory rather
than routed around; a denial is a decision, not an obstacle.*
