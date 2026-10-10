# check_reviews_agree — trustworthiness, strong

Two reviewers judged the same submissions without seeing each other's answers.
`reconcile` merged the two documents. This checks the merge: **the disagreement
list is empty, and every count in it is the count the lists actually hold.**

This is the *"两份审阅要一致"* requirement, and it is the only validator in
`grade` that can fail for a reason nobody wrote a bug to cause.

## What it checks

| | |
|---|---|
| `disagreed` | **empty**. A non-empty list fails, and the message names the first five pairs |
| `totals.agreed` | recomputed from the `agreed` list, never read |
| `totals.disagreed` | recomputed from the `disagreed` list |
| `totals.pairs` | equals the two lengths added |
| the pairs | no pair in both lists, and no pair listed twice in either |

Every number is recomputed rather than trusted. A merger that lost a pair and a
merger that miscounted produce the same `totals` if the `totals` are believed,
which is what makes reading them instead of deriving them a check that cannot
fail.

## The limit, and it is the interesting part

**Two AI reviewers agreeing is evidence of consistency. It is not evidence of
correctness.**

Nothing in this package can establish that a review is right. Both reviewers
are language models given the same submissions; they can reach the same wrong
conclusion, and they are more likely to do so where the submission is
misleading in an ordinary way — a plausible complexity claim beside code that
does not support it is exactly the sort of thing two readers miss together.

What a pass does establish is narrower and still worth having:

- neither reviewer's document was garbled, truncated, or written to a different
  shape — a mismatch shows up here as a disagreement;
- neither reviewer skipped a pair the other reviewed;
- the two reached the same verdict by two different reading methods
  (`assets/review_x.task/readme.md` works student by student and reads the
  claim after the code; `assets/review_y.task/readme.md` sweeps problem by
  problem and writes its own description before looking at the claim). Two
  routes to one answer is a weaker guarantee than a proof and a much stronger
  one than a single pass.

`strong` is the right label for that, because `strength` qualifies a **pass**
(`validator` spec §5.4) and the pass being qualified is *"the two reviewers
agreed everywhere and the merge is arithmetically sound"*. The check is total
over the document and there is nothing approximate in it. What is limited is
what agreement means, and that is a property of the subject rather than of the
check — which is exactly the distinction `strength` is for.

## Why the merge and the judgement are two steps

`reconcile` records disagreement; this decides what to do about it. A single
body doing both would be tempted to resolve a conflict — take the stricter
verdict, or two of the three fields — and the artefact would then contain a
consensus that never happened. Splitting them keeps `review` honest and leaves
this check with something real to test.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json`, `inputs.json` and `materials.json` beside it,
and it writes `verdict.json` — one boolean per handoff id in `inputs.json`.
