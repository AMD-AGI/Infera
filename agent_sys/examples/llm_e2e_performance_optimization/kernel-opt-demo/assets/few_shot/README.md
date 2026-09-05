# Few-shot — one handoff that passes, and two that do not

This is the mission's *"few shot in the assert"*: `assets/` **is** the assert
directory. The loader supplies its name as `${TASK_PACKAGE_ASSERT_DIR}`
(`spec_loader/variables.py`), whose value is the literal string `assets` — the
`ASSERT` spelling is the schema's, not a typo here.

Everything below was **measured**. `01_pass/` is a real campaign's output, and
the two failure messages are the ones the validator actually printed when run
against fixtures built from it on 2026-09-01. Nothing here is illustrative
prose.

---

## `01_pass/` — the shape to copy

A complete `kernel_optimization` handoff from the sampler-softmax campaign.
Passes `check_optimization_shape` **and** `check_speedup_substantiated`.

```
README.md                                   ## Purpose / ## Interface / ## Boundary
items/codes/sampler_vocab_softmax.packup_20260831/
    README.md          with a ## Result section carrying the numbers
    REPRODUCE.md       ordered commands + an "Expected output" section
    environment.md     host, GPU, image digest, versions, what was installed on top
    notes.md           the traps, and the two things left unexplained
    scripts/
        run_forge.sh   the invocation, verbatim
        kernel/        driver.py · graph_harness.py · measure_baseline.py · seed kernel
    results/
        forge_result.json         forge's own verdict
        optimization_report.md    forge's report
        optimized_kernel.py       the artefact
        verification.json         the producer's OWN re-measurement
        best_result.json · candidates_index.jsonl
```

**The four things that make it pass, and they are not the file list.**

1. **`verification.json` is the producer's own measurement, not a copy of
   forge's.** It records 2.6123× where forge reported 2.8328×, and it says so:
   `forge_reported_mean_case_speedup` is in the file next to it. A handoff whose
   own number quietly equals forge's has not re-measured.
2. **The disagreement is explained rather than hidden.** `noise_note` records
   19–21% round-to-round spread on the optimized side against 1.7–2.2% on the
   baseline, medians of five fresh processes each — which is why a 7.8% gap
   between the two numbers is not evidence that either is wrong.
3. **`## Boundary` says what is *not* known.** No end-to-end claim; one dtype and
   one vocabulary; the spread unexplained; no GPU reserved and no clock locked.
4. **`scripts/kernel/` ships the apparatus**, so the kit can be checked by
   someone who does not have the workset — and so the expensive validator has
   something to run.

When the same kit is re-measured, `check_speedup_substantiated` has returned
2.723× and 2.912× on two separate occasions against a claim of 2.6123×. Both
pass: the tolerance is one-sided and **under-claiming is honesty**.

---

## `02` — a mock that claims a speedup

Built by taking `01_pass` and setting `"mock": true`, changing nothing else.
What the validator said, verbatim:

```
verification.json says mock=true but README.md never says MOCK —
  a mock that is not visibly a mock reads as a success
verification.json is mock=true and still claims a speedup of 2.6123
check_optimization_shape: 0/1 passed
```

**Why this is the most dangerous failure in the package, and why it gets two
rules instead of one.** Every file is present, every document has substance,
every number parses. It is indistinguishable from a success to a human skimming
it, and three weeks later nobody remembers which runs were wiring tests. So the
gate is not "did you do the work" — it is *a run that did not optimize must be
impossible to read as one that did*.

The same rule covers **smoke** runs (`"degraded": true` requires `SMOKE` in the
README), for the same reason: a one-hour degraded campaign produces a real
`forge_result.json` and reads exactly like a three-hour one.

What a correct mock looks like: `mean_case_speedup` at or below 1.0, `MOCK` in
the first line of `## Result`, and — best practice, from a real mock run on
2026-09-01 — the seed measured against **itself** as a null control. That run
reported 0.9949 and 0.9992 on two independent protocol passes, which is a more
useful artefact than a mock that measures nothing: it puts a number on this
machine's noise floor and confirms the workset's ~1.05× guidance empirically.

---

## `03` — a real run with no re-measurement of its own

Built by copying `01_pass` **without** `results/verification.json`.

```
missing evidence results/verification.json
check_optimization_shape: 0/1 passed
```

Short message, large point. A handoff carrying `forge_result.json` and nothing
else is asking the reader to take the optimizer's word for the optimizer's
work. `verification.json` is the one file in the kit that was produced by
something other than the thing being judged, and the expensive validator has
nothing to compare against without it.

---

## The failure this list does not contain

There is **no example of a kernel that is fast because it is wrong**, because
nothing in this package would catch one. `check_speedup_substantiated`
re-measures timing and not accuracy; correctness rests on forge's SNR gate and
on the producer's own re-run, both of which are *claims inside the handoff*.

If you are the producer: that gap is why `driver.py` and `graph_harness.py` are
protected and must be copied unmodified. The oracle is the only thing standing
between a wrong kernel and a passing handoff, and you are the one shipping it.
