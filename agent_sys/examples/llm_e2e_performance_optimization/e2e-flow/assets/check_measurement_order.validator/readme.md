# check_measurement_order

Trustworthiness, strong. Both arms ran the same measurements in the same order,
they did not overlap in time, and the patched arm started only after the stock
arm had finished.

## Why it exists: an ordering guarantee that stopped being a graph edge

`integration-demo` had eight leaves, and three of its edges carried an
**argument** rather than a datum. The load-bearing one was
`serve_patched ← measure_stock`. It did not exist because the patched bring-up
needed the stock arm's numbers — it existed because **the patched bring-up's
first act is to destroy the stock deployment**, and connecting it to
`serve_stock` instead would have let the scheduler start the teardown while the
stock arm was still being measured.

Mission M5.2 collapsed those five leaves into one task, so that edge is gone. The
order is now a numbered list in one readme: weaker than a scheduler constraint,
stronger than nothing. **This validator is what makes the difference visible.**
The producing task timestamps every step into each arm's `env/steps.json`, and
this reads the two records back.

## The four rules

**Same steps, same order.** "Round 1 is cold against this trace" is only true of
an arm if the same things happened before it. Two arms with different sequences
are two different experiments and every comparison downstream is between them
rather than between stock and patched.

**The arms do not overlap in time.** Overlapping arms shared a node, eight GPUs
and a page cache. Measured here: one stock deployment read 193.59 tok/s beside an
idle neighbour and 47 tok/s beside a neighbour holding all eight cards — a 4×
swing produced by nothing the pipeline changed.

**No overlap inside an arm either.** A saturated trace replay running while
lm_eval scores invalidates both numbers, and neither looks wrong afterwards.

**Every step exited zero.** A step that failed did not measure.

## The weak link, and what corroborates it

`env/steps.json` is written **by the same body whose ordering it attests to**. On
its own it is a file the producer could have assembled at the end from memory,
and "the arms did not overlap" is the one claim nothing else in the graph can
recover afterwards — by validation time both deployments are gone.

So the timestamps are cross-checked against a record **the producer did not
write**: AIPerf's own `start_time` and `end_time` in each round's
`profile_export_aiperf.json`.

**Disjointness, independently.** The stock arm's last AIPerf window must end
before the patched arm's first begins. This is the strong rule and it is
offset-free — both timestamps come from one clock read one way, so it holds in
any timezone and whatever `steps.json` claims.

**Containment, with a skew allowance.** Each `bench_r<N>` AIPerf window must fit
inside the step window claimed for it. AIPerf writes naive local time and the
step record writes UTC with an offset, so a node off UTC shows a constant skew —
a timezone, not a fabrication, and it is printed rather than failed. **What fails
is the two arms disagreeing about that offset:** one node has one clock, so a
per-arm difference means at least one arm's record was not written while it ran.

Proven on the sealed pair: PASS, with AIPerf's own windows 3104 s apart. Proven
against three fabrications — a `steps.json` claiming disjoint arms whose AIPerf
records overlap, a per-arm clock disagreement, and exports with the timestamps
removed — all three refused, each naming what it found.

## What it does not do

It does not bound the *gap* between the arms. A long gap is where node load can
change underneath the comparison, so the gap is printed and, past
`warn_gap_seconds`, noted — and that is all, because the fix for it is a
comparability gate at bring-up rather than another threshold here
(`../../../todo.md` T7).

## It needs both arms, and that is a feature

The two kinds are produced by one task, so the output phase stages both. If only
one arm reaches this body, the arms were produced by two tasks — which is the
split M5.2 forbids — and saying so is more useful than grading half a comparison.

## `bench` matches `bench_r1`

An `expected_steps` entry is satisfied by an exact match or by the same name with
a `_r<digits>` suffix, because the number of replay rounds is a `--var`. Nothing
else is fuzzy: `smoke_v2` does not satisfy `smoke`, because a renamed step is a
changed method.
