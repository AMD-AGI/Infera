# check_kernel_table

Usability, `strong`. Five rules over the ranking.

## Why usability and not completeness

A CSV can arrive with every column and a hundred rows and still be unusable. If
the top twenty-five kernels account for a tenth of the CUDA time, there is no
operator worth picking and the stage that consumes this would be choosing noise.
Completeness cannot express that; usability can.

`strong` is still honest: every rule is a count or a sum over the table, and none
of them is approximately right.

## The rules

1. **Magpie's columns, exactly.** A rename upstream stops the pipeline here
   rather than at whatever parses the CSV next.
2. **Enough rows to rank.**
3. **The shares add up to a whole run.** Too low and the trace covered a fraction
   of the work; above 100 and something is being double-counted across ranks.
4. **The head accounts for enough to act on.** This is the usability rule.
5. **`Input Shapes` is populated.**

## Rule 5 is the one worth explaining

That column exists only when the capture asked the profiler for `record_shapes`.
A trace taken without it produces the same CSV, with the same rows and the same
percentages, and that one column empty. Everything downstream that needs an
operator's shapes — a naive reimplementation, a roofline, a correctness test —
becomes impossible, and nothing else in the pipeline would notice until somebody
tried.

The capture sets `record_shapes: true` explicitly for exactly this reason. This
rule is what makes that setting's absence loud instead of silent.

## The head's share is recomputed, not read

`top_kernels.json` carries the share the producer computed. This validator sorts
the CSV and computes it again. Same reason `check_facts` recomputes its totals in
the reference demo: a number a producer wrote about itself is not evidence about
the producer.
