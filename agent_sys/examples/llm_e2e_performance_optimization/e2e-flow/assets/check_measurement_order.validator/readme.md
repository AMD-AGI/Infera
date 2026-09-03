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
