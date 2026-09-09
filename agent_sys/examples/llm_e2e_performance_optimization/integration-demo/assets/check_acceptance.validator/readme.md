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
