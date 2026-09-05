# check_acceptance

Completeness, strong. All three kinds of correctness evidence arrived, and each
is readable on its own terms.

**It does not compare the arms.** An eval score is admitted whatever it is,
because a single score has no baseline; `check_no_regression` is where two of
them become a judgement. What is checked here is that the measurement happened
and can be read.

Two rules exist because of failures that produce a *number* rather than an error,
which is the only kind worth a strong validator:

**The needle actually sent a long prompt.** A run that silently sent 3000 tokens
is not a long-context test, and it passes every other check. `needle.py` reads
`usage.prompt_tokens` back from the server for exactly this reason, and the ratio
against the declared target is checked here.

**The eval scored what it says it scored.** The scored count comes from counting
`Correct Answer` in the html report, because the dataset size is the wrong
answer — GSM8K ships 1319 rows, `gsm8k` scores 1314 and `mixed_prefix_gsm8k`
1299, since the evaluator slices the few-shot examples out of the evaluation set.
Every interval computed downstream needs that number, so a mismatch between the
index and the html is a failure rather than a discrepancy.

The frontier needle length is reported and never gated. A stock deployment was
measured to fail its head depth at 127k tokens, and a gate the baseline cannot
pass is a gate that gets switched off.

---

## What changed on the way into `e2e-flow`

`require_frozen_checks` is the old `require_smoke_checks`, renamed to say what
those checks are: **frozen**, shipped in `assets/accept/`, part of the package.

**And that is why M5.4 exists.** A frozen suite is in the repository, so an
optimisation — or an agent driving one — can be made to satisfy exactly those and
nothing else, and the suite then measures compliance with itself. So
`min_adhoc_cases` cases are invented per run: 同时还要临时 ai 生成几个。免得作弊.

Four rules beyond the count, each closing a way the requirement could be met
without being met:

- **the generator prompt is recorded.** What was *asked* is half the evidence; a
  generated case whose text was thrown away cannot be audited, and "three ad-hoc
  cases passed" then means nothing.
- **no ad-hoc case repeats a frozen one.** Regenerating the shipped suite
  satisfies the count and adds no coverage.
- **no ad-hoc case repeats another.** Three copies of one case is one case.
- **both arms run the same set**, checked across the two handoffs. Different
  cases per arm is not a comparison, and it is the shape a regression could hide
  behind.

A failing ad-hoc case on one arm is printed and does not refuse: a single arm's
result is a fact about that deployment, and stock-passed-patched-failed is a
regression that `check_no_regression` decides.

**The mock cannot exercise this rule.** No sealed artefact carries an
`adhoc.json`, and inventing one would be exactly the synthesis `MOCK-MAP.md`
forbids, so a mock run passes `--var adhoc_cases=0` and this rule is untested
until the first real run.
