# compare

Put the two arms side by side and state whether the patch is accepted.

Pure computation over four handoffs. It touches no cluster, which makes it the
one leaf here that can be re-run for free against a finished run's store — useful,
because the judgement is the part most likely to need revising after a review.

## What counts as a regression

- **llm-eval**: the Newcombe interval on the difference excludes zero *and* the
  difference is negative. Not "the score went down": at 200 questions two runs of
  the same deployment differ by several points routinely, and treating that as
  evidence would make this a coin-flip detector.
- **needle**: a depth the stock arm retrieved and the patched arm did not. Both
  failing is not a regression — it is a property of the model at that length,
  which is why the frontier run exists and why it is not gated.
- **smoke**: any check that passed on stock and failed on patched.
- **performance**: a relative change past the bar, compared round for round only.
  Cold and warm differ by an order of magnitude on this trace.

## What it deliberately does not do

Judge an arm on its own. Every line is a comparison, because that is the only
form of evidence this stage can produce honestly.

## Why the statistics live in lib/

`check_no_regression` recomputes every line of this report, and the two agreeing
is the whole point of that validator. They cannot agree if each carries its own
copy of the arithmetic.
