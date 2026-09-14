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

---

## What changed on the way into `e2e-flow`

**`trustworthiness`, not `usability`.** What it judges is not whether the report
is legible but whether its conclusion is the one its own numbers support.

**Two rules the mission added.**

`stock_vs_m2` (M5.1.3.1) is a **blocker**. The stock arm and m2's
`profiling_mode_off` bench are the same measurement of the same deployment one
stage apart. If they disagree beyond `stock_vs_m2_tolerance`, the two stages
measured different machines — or one machine in two states — and this report
compares numbers that were never comparable. Both signs count: a stock arm
*faster* than m2's is as much evidence of a different machine as a slower one.
The comparison is carried into the report by `compare.py`, because this
validator's only input is the report.

`kernel_reconciliation` (M5.1.3.2) is a **warning**, exactly as the mission asks:
作为 report/warning 报告，不作为 blocker. Amdahl over one kernel, with the
kernel's share of the profile measured under the profiler with CUDA graph off and
the end-to-end number measured under neither. The two sides legitimately
disagree, so gating on it would refuse correct work. **The hard floor stays where
it already was** — the performance bars — and a literal "strictly not slower"
floor would be the wrong reading of it, because the within-arm round-to-round
spread on a steady node is ~2% and such a floor would refuse noise about half the
time.

**A producer may not widen its own bar.** The report records the thresholds it
was decided against; if any is looser than this validator's `args`, that is a
failure on its own. The sealed 2026-09-02 report is exactly that case — it
declares 0.35/0.30, widened in response to two arms measured fifteen minutes and
one co-tenant apart. The right response to that was a comparability gate at
bring-up (`../../../todo.md` T7), and **the bars are not to be widened**: the
5% / 10% pair was measured to be right.

**The reason-agreement rule was broken and is fixed.** It compared
`recomputed_reason[:40]` against each stated reason, which only works when both
sides phrase a finding identically — and they do not, because `compare` writes
for a human and the recomputation writes a metric id. Measured against the real
refused report: all seven of its reasons were flagged as "reasons the report does
not list" while the report listed all seven. It had never fired before because
the only report it had ever graded was an accepted one, where both sides carry
zero reasons and the comparison is vacuous. Matching is now structural — round,
metric (by id or by label) and column.
