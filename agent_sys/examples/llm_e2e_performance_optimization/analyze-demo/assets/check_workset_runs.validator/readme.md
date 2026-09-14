# check_workset_runs — trustworthiness, strong

Every driver ran on the target GPU, met the correctness gate, and produced a
measurement with the shape mission 3.2.7 asks for.

## What it checks

1. The evidence says which node it was measured on.
2. Per operator: it ran, it was correct, and it produced a measurement.
3. At least `min_groups` groups totalling at least `min_groups x min_iters_per_group`
   iterations.
4. The stored weighted mean agrees with the per-group figures to within 1%.
5. Run-to-run spread is at or below `max_rsd`.
6. The share measuring cleanly clears `min_pass_ratio`.

## Why rule 4 recomputes instead of trusting

The producer and this validator both import `assets/lib/bench_stats.py`. A
stored figure that disagrees with the per-group numbers therefore cannot come
from a different formula — it can only come from the record having been edited
after it was measured. Putting the arithmetic in one shared module is what turns
that into a checkable property.

## Why rule 5 matters more than it looks

A noisy baseline is worse than no baseline. forge-loop optimizes against the
number this step produces; if the spread is wide, the first candidate that
happens to land on a fast sample looks like a win, and the campaign chases
noise for hours.

## Why rule 6 is a ratio

A workset whose entry point is still unknown legitimately cannot run yet — see
`identify`'s `agent_recovered` case. Failing the whole step for it would block
the operators that are ready.
