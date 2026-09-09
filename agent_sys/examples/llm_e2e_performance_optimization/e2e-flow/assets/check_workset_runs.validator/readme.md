# `check_workset_runs` — trustworthiness, strong

The workset's own one-click correctness and performance entrypoints run, on this
hardware, and agree with the numbers it prints.

## This validator is the package's trust chain

CONTRACT.md §4.0. m4 is told to take its ground truth **strictly** from the
workset and to abort rather than re-measure when the premise does not hold —
M4.3.5, reversed from the previous round's *"do not trust the workset's printed
number"*. That instruction is only safe because something has already run the
workset's own tests on this hardware.

`build_workset` builds **and** measures — there is no `verify_workset` task,
because splitting build from measure across two agents is what M2.5 forbids. So
the evidence in the workset is the producer's own claim about itself, and grading
the *shape* of that claim would make the chain a claim about a claim.

**So this validator re-measures.** `build_workset` asserts a baseline; this
confirms the assertion; m4 may then divide by it.

## What it checks

1. The recorded performance report has `impl: baseline`, and the recorded
   correctness report passed. A timing of a wrong kernel is not evidence.
2. Every timed operator has a correctness row, and that row passed.
3. Per shape: at least `min_groups` groups (5) totalling at least
   `min_groups × min_iters_per_group` iterations, and `per_group_ms` with one
   entry per group.
4. `weighted_mean_ms` and `rsd` are **recomputed from `per_group_ms`** and must
   agree. The producer and this validator import the same `workset_io`
   functions, so a stored figure that disagrees cannot come from a different
   formula — only from the record having been edited after it was measured.
5. `rsd` is at or below `max_rsd` (0.10).
6. **`reverify_shapes` shapes are re-measured here**, through the workset's own
   `./run_performance.sh --operator <id> --shape <primary> --json <tmp>`, and the
   fresh number must be within 25% of the recorded one. `reverify_shapes: 0` is
   **refused**, not honoured: it would turn this into a reader of the producer's
   own claim.
7. The share of operators that ran clears `min_pass_ratio`.

## The thresholds, and why they differ

`max_rsd` is 0.10 and the re-verify tolerance is 0.25. They measure different
things. `max_rsd` is spread *within* one sitting on a quiet node, where the
measured round-to-round figure is ~2%. The re-verify comparison is across two
sittings minutes and one co-tenant apart — the effect `todo.md` T7 measured at
12% between two arms that were 1.1% apart under matched load. A tolerance as
tight as `max_rsd` would fail honest evidence on a shared box.

**Do not widen either to make a run pass.** A previous round widened a comparable
pair from 5%/10% to 35%/30% in response to a cross-instance artefact, and that
was the wrong response: the missing control is a comparability gate, not a looser
bar.

## Two failures that are not the workset's fault, and are still failures

A node too busy to give a stable measurement fails rule 5. A host on which the
entrypoints refuse to run — `ground_truth.abort_on_mismatch` — fails rule 6.
Neither is a defect in the artefact and both are correct verdicts: the artefact's
claim is *"these numbers hold on this hardware"*, and neither case establishes it.

`todo.md` T10 records that rule 5 and `min_pass_ratio` express opposite
philosophies about forgiveness in one validator. That is unresolved, not
overlooked.

## Cost

`gpu_hours`, and honestly so: one shape, five groups. The `--shape` selector
exists on the entrypoints for this — a full re-run would double the workset's GPU
bill at every seal, and the cheap version of this check would not exist without
it.
