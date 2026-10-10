# `check_kernel_table`

Usability, `strong`, **program**. Eight rules over a ranking.

**One definition, used by two modules.** Mission M3.5: m2 produces the table and
m3 reads it. The two demo packages each carried a copy with different `args` and
different `min_rows`, over two artefacts that turned out to be different things
— a real 124-kernel capture and a 34-row synthetic seed. The spec lives once, in
`steps/common.yaml`; this directory is its one body. **Do not define a second.**

`usability` and not `completeness`: a table that arrived with every column and a
hundred rows can still be unusable. If the top twenty-five kernels account for a
tenth of the time there is no operator worth picking and m3 would be ranking
noise.

## The rules

1. **The record validates against the package's schema**, resolved by name from
   the same loader the producer used (mission G2).
2. **The schema copy the handoff carries is byte-identical to the package's.**
   CONTRACT §3.4: a `structured_text` handoff is self-describing, so it carries
   the schema it was written against — and identity with the package's is what
   stops the copy being a private fork that quietly says something else.
3. **Magpie's columns, exactly**, in `items/table.csv`. A rename upstream stops
   here rather than at whatever parses the CSV next.
4. **Enough rows to rank.**
5. **The shares add up to a whole run.** Too low and the trace covered a
   fraction of the work; above 100 and something is double-counted across ranks.
6. **The head accounts for enough of the run to be worth acting on** —
   recomputed from the CSV rather than read from the producer's own summary, for
   the reason every recomputation here exists: a number a producer wrote about
   itself is not evidence.
7. **`Input Shapes` is populated.** That column exists only when the capture
   asked the profiler for `record_shapes`, and without it every roofline
   downstream is impossible — but the table looks entirely normal, so nothing
   else would notice. **Empty on a row is legal; empty on every row is not.** 25
   of the 124 rows on the reference table are legitimately empty.
8. **The record carries every row of the CSV**, under the field names m3 reads.
   Rules 3–7 judge the CSV; this is the one that checks the thing m3 is actually
   handed. **The schema cannot express it** — a 25-row document is a perfectly
   valid `kernel_table` — which is exactly why the rule is here and not there,
   and it is the case the schema's own gating exercise found passing.

Optionally, rule 9: enough of the head carries a launcher frame, when the round
was asked for a stack window. Off unless `min_launchers_in_top_n` is set,
because a round taken with `--var stack_window_s=0` resolves none by design. On
the reference table 24 of the top 25 carry one, so a floor of 10 is comfortable.
A frame naming no `container_root` is counted as absent: the consumer has no
repository to resolve it against, so it is present without being usable, which
is worse than missing because it counts towards coverage.

## The layout is `structured_text`'s, not the demo's

The record is `items/text.json`, Magpie's export is `items/table.csv`, the
schema copy is `items/schema`. The sealed sample was produced as a
`reproducible` handoff with everything under `items/result/`; `../lib/m2_reshape.py`
corrects it and this body reads the kind's layout rather than the sample's.

`top_kernels.json` and `launchers.json` from the sealed sample are deliberately
not carried: both are derivable from `text.json`, which holds every row and each
kernel's `launcher` block, and rule 6 recomputes the head's share from the CSV
rather than reading anybody's summary of it.

## Proven both ways

The real 124-kernel table PASSES — 124 kernels, top 25 covering 92.6%, shares
summing to 99.96, 24 of 25 with launcher frames. Six hand-built failures are
refused naming the fault: a forked schema copy, a top-25 head shipped as a whole
table, a capture with no `record_shapes`, a missing schema, a record not bound
to its CSV, and a producer whose row count disagrees with its own rows.
