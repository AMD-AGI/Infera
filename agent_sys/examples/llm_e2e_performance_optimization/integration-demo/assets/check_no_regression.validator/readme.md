# check_no_regression

Usability, strong. Recompute the entire accept-or-reject argument from the raw
numbers, and fail if the answer disagrees with the one the report states.

**It does not read the `verdict` field.** That is the whole design. `compare` is
a program in this package, and a bug in it would otherwise produce a report that
passes its own validation — with the decision this stage exists to make resting
on nothing.

The statistics come from `assets/lib/eval_stats.py`, which `compare` also uses.
Sharing the arithmetic is deliberate: what is being checked is the report's
*assembly* — which rows were included, which were judged, whether a regression
was noticed — not whether two copies of a Wilson interval agree.

A disagreement fails the handoff **even when the recomputation says accepted**.
Two answers that differ mean one is wrong, and until that is resolved neither can
be relied on. Agreeing on the answer is also not enough: the reasons are compared
too, because a verdict that is right by accident is not a verdict.

`usability` rather than `completeness`, because what it judges is whether the
report can be used to make the decision it was built for.

The performance bars it recomputes against have no measured basis yet. They want
the natural run-to-run spread of a single arm, which the first full run will
produce; until then 5% and 10% are placeholders wide enough not to fire on noise.
That is recorded as an open question in `DESIGN.md` section 11.
