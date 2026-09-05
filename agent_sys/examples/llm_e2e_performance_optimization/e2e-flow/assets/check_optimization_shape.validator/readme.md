# `check_optimization_shape`

A **program** validator (G4.1: 能程序化的，尽量不用 ai). Nothing here is judged;
every rule is decided by a file that is there or is not, a document that
validates or does not, or two values that are equal or are not. That is why the
spec calls it `strong` without qualification — it cannot be *approximately*
right about whether `apply.integration_point.source_file` is the file the
workset declared.

It is the **cheap** half of m4's output gate. `tags.cost: seconds` against
`check_speedup_substantiated`'s `gpu_hours`, and a phase runs cheapest-first, so
a handoff whose document does not parse fails in a second rather than after a
measurement with nothing to measure.

## Criteria

| # | criterion | fails when |
|---|---|---|
| 1 | **the document exists and validates** | `results/kernel_optimization.json` is missing, does not parse, or does not validate against `assets/schemas/kernel_optimization.schema.json` — the same file the producer was handed (G2) |
| 2 | **the workset it claims to come from travels with it** | `results/workset.snapshot.yaml` is missing or is not a valid `workset` |
| 3 | **the premise is the workset's** | `premise.abort_on_mismatch`, `warn_on_mismatch` or `workset_environment` differ from the snapshot's `ground_truth` — m4 does not get to decide which of its own differences were the harmless kind |
| 4 | **the integration point is the workset's** (M5.1.1) | `apply.integration_point.{source_file,entry_function}` differ from the workset operator's `edit_target`. This is what lets m5's `apply_patch` be a program |
| 5 | **the protocol and the entrypoints are the workset's** | `evidence.performance.protocol` differs from the snapshot's `protocol`, or either `entrypoint` is not the workset's `cmd` verbatim. A producer that re-implements the correctness suite has measured a different thing and the difference is invisible in a report |
| 6 | **every declared performance shape was measured** | the workset declares a `performance` shape absent from `measured.per_case_ms` |
| 7 | **the ratios follow from the tables** | any `claim.speedup_per_case[c]` is not `baseline[c]/measured[c]` within 0.5%, or `mean_case_speedup` is not their mean. Recomputed rather than trusted: a figure that does not follow from the raw numbers can only come from the record having been edited after it was measured |
| 8 | **a claim clears the workset's own noise floor** | `mean_case_speedup < claim.noise_floor`. Reporting 1.02× as an improvement is a false claim on a protocol whose null control measures ±0.2% |
| 9 | **the packup is a packup** | a missing `README.md` / `REPRODUCE.md` / `environment.md` / `notes.md`, one below its content-line floor, an empty `scripts/` or `results/`, a `## Result`-less README, an `environment.md` with no numbers in it, or an explicit `TODO`/`TBD`/`FIXME`/`XXX` placeholder |
| 10 | **a mock is visibly a mock** | the document says `forge.mock` and `README.md` never says MOCK, or says `forge.degraded` and it never says SMOKE. The schema already forbids either from carrying a claim; this is the half a schema cannot reach, because it is about what the prose says |
| 11 | **the evidence and the overlay files are present and non-empty** | anything in `args.required_evidence` is missing or zero-length, or `apply.files[].source` names a file the packup does not carry |

## The one opportunistic criterion, and why it is honest to have one

Criterion 3 holds the document to the **snapshot**. Whether the snapshot is a
faithful copy of the real workset is a different question, and on m4's output
phase it cannot be asked: a validator is handed only the handoffs it declared in
`inputs`, staged over `layout.stage(task.outputs, …)`. Declaring
`operator_workset` in `inputs` instead is not a route to it — it binds this body
to the *workset kind's* phases and records a FAIL on a producer that did nothing
wrong, which is measured and written up in
`../../../kernel-opt-demo/assets/check_speedup_substantiated.validator/check.py`.

So `_cross_check` looks through `materials.json` for a staged handoff that
*is* a workset and compares. In m4's output phase there is none and it **says
so in a note**. In m5's input phase there is one, because that phase stages
every one of m5's inputs, and the snapshot stops being taken on trust there.

A check that silently passes when it did not run is worse than one that is
honestly absent, so this one is never silent.

## What it cannot catch

It runs nothing. No number here is checked for truth — a document claiming
`mean_case_speedup: 99.0` passes criterion 7 as long as 99.0 is what the two
tables divide to. Substantiating a number is `check_speedup_substantiated`'s
job, and the two are deliberately separate so the cheap one can fail first.

There is deliberately **no `<...>` template-slot rule**, and its absence was
measured rather than assumed — see the comment at the head of `check.py`.
