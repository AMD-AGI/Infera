# production_grade

Judge whether the deployment configuration under check is state-of-the-art and
production-grade, against public knowledge, prior experience, and — for
individual items — an open-source publication or a customer guarantee.

Write `verdict.json`: one boolean per id in `inputs.json`.

**This validator is `weak`, and the label is the honest one.** It cannot state,
in advance, the number or the comparison that decides it; the answer is "the
agent assesses whether it looks reasonable", which is spec §5.6's test for `weak`
whatever the intent. A weak check labelled strong is worse than no check, because
it stops anyone looking further.

Two things follow, and they are not the same thing:

- **A failure here binds.** The label qualifies a pass, never a failure. A check
  that found something wrong found something wrong, whatever its rigour.
- **A pass here is not evidence.** It is reported as a low-confidence pass and
  must not be read as the quality being established.

The cost of that asymmetry is named rather than discovered: this validator's
false positive halts a branch, and it may be wrong.
