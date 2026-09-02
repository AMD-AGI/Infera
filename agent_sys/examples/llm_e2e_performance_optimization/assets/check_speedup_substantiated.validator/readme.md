# check_speedup_substantiated

**Claim:** the speedup this handoff claims re-measures on this machine, over
enough rounds to survive the measurement spread this machine actually shows.

`trustworthiness` / **`weak`** / `cost: minutes` / `logic_source: external_dynamic`.

## What it actually does

1. Finds the packup and its `scripts/kernel/` — the measurement apparatus the
   producer was required to ship.
2. Makes **two** fresh copies of it: one left as the seed, one with
   `results/optimized_kernel.py` dropped in over the seed.
3. Runs `measure_baseline.py` — 5 rounds × 30 iterations, each round a fresh
   process — against both.
4. Computes the mean per-case speedup from the medians and compares it to
   `verification.json`'s `mean_case_speedup`.

It uses the **producer's** driver, not one of its own. A validator with its own
measurement harness would be comparing against something the producer never
measured, and any disagreement would be unattributable.

## Why it declares ONE input, after declaring two

The first version was `inputs: [kernel_optimization, workset]`, because the
driver lived in the workset and an output phase stages only outputs. That is the
documented route to a handoff a phase would not otherwise stage. **It was wrong,
and the failure is worth knowing about before you copy the pattern.**

A phase's validator set is `closures.validators_for(<closure>)` — the union of
the closure's own `validators` list **and every validator joined to one of its
handoff kinds**. A validator is joined to a kind by *naming it in `inputs`*. So
naming `workset` did not merely grant access to it; it **bound this validator to
the `workset` kind's phases**. Measured 2026-09-01: it ran in `publish_workset`'s
output phase, against a workset with no optimization staged beside it, and
recorded `trustworthiness / weak / FAIL` against a task that had done nothing
wrong. The publisher cannot substantiate a speedup; it never claimed one.

**`inputs` is not an access list. It is a binding.** That is the lesson, and it
is not obvious from the schema.

The fix chosen was *not* to make the body tolerant of missing materials — a
validator that passes when its inputs are absent is worse than one that is
scoped correctly. It was to make the handoff **self-contained**: the producer
ships `scripts/kernel/`, `check_optimization_shape` requires it, and this body
reads from there. The handoff is strictly better for it, because a reproduction
kit that does not carry what measures it could not be checked by anyone lacking
the workset.

## The numbers, and why they are what they are

| arg | default | why |
|---|---|---|
| `rounds` | 5 | the workset's baseline protocol is five. A comparison across differently-sized samples is not a comparison |
| `iters` | 30 | ≥10 is the protocol floor; 30 costs nothing extra at this kernel size |
| `tolerance` | 0.15 | the optimized side has shown ~8% round-to-round spread here against the baseline's ~2%. A tight bound would fail honest handoffs more often than dishonest ones — the wrong direction for a trustworthiness check to fail in |
| `noise_floor` | 1.05 | below this, a "speedup" is not distinguishable from spread on this machine, so reporting one is a false claim rather than a small win |
| `timeout_seconds` | 1800 | slack, not a budget. Pass something small when testing the wiring |

**The tolerance check is one-sided.** A handoff that under-claims passes. Only
over-claiming fails. Under-claiming is honesty and this validator has no
business punishing it.

## Mock handoffs

If `verification.json` says `"mock": true`, this body **passes and says why**: a
mock claims no speedup, so there is nothing to substantiate. Inventing a failure
there would be wrong. The risk that a mock is mistaken for a result is handled
one validator earlier — `check_optimization_shape` refuses a mock that claims a
speedup or that does not say `MOCK` in its README.

## Why `weak`, argued rather than asserted

It establishes exactly one thing: **the claimed number reproduces here, today.**
It does **not** establish:

- that the optimized kernel is *correct*. It re-measures timing, not accuracy.
  Correctness rests on forge's own SNR gate and the producer's re-run, both of
  which are claims in the handoff rather than things this body checks.
- that the kernel is correct or fast at any shape the workset does not cover.
- that the speedup survives integration. A kernel that is 14.5% of decode GPU
  time bounds its own end-to-end benefit by Amdahl, and nothing here measures
  the service.
- that the result is stable across days. Two measurements a day apart on this
  host agreed on medians to 2% and disagreed on *spread* by 5×, and that is not
  explained.

A `strong` label on any of those would be a claim the body cannot support, and
`strength` qualifies a PASS — so overstating it would silently upgrade every
future pass.

## What it cannot catch

- A handoff whose `optimized_kernel.py` differs from what forge actually
  produced. It measures the file it is given.
- A kernel that is fast because it is wrong. There is no correctness check here.
- Interference from other tenants on this shared host. Nothing reserves a GPU;
  a noisy neighbour during these 5 rounds is indistinguishable from a slow
  kernel.
