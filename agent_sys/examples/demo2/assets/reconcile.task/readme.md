# reconcile — merge the two reviews, and hide nothing

`reviewer_x` and `reviewer_y` judged the same submissions independently. This
step puts the two documents side by side and produces one: the pairs they agree
on, the pairs they do not, and a count of each.

**It merges; it does not judge.** Whether a disagreement is acceptable is
`check_reviews_agree`'s question, and keeping the two apart is what lets this
artefact stay honest. A body that resolved a conflict — took reviewer X's
answer, or the stricter of the two, or the majority of the three fields —
would produce a document in which the disagreement had never happened, and the
validator downstream would then be checking a fact this program had
manufactured.

So a pair the two reviewers answered differently is copied into `disagreed`
with **both** answers intact, and the run fails on it.

## What it reads

| variable | what is in it |
|---|---|
| `$AGENT_SYS_INPUT_REVIEW_X` | reviewer X's document — `items/text.json` |
| `$AGENT_SYS_INPUT_REVIEW_Y` | reviewer Y's document, the same shape |

Both are `{"reviews": [ {student, problem_id, implements_claimed_algorithm,
complexity_credible, verdict, comment}, ... ]}`.

## What it compares

A pair is `(student, problem_id)`. Two rows for the same pair agree when all
three **judgement** fields are equal:

- `implements_claimed_algorithm`
- `complexity_credible`
- `verdict`

`comment` is prose and is not compared. Two reviewers reaching the same verdict
by different reasoning is agreement; requiring them to phrase it identically
would make agreement a test of language rather than of judgement.

A pair present in one document and absent from the other is a **disagreement**,
not an omission to be tidied away: one reviewer saw a submission the other did
not, and that is exactly the sort of thing this step exists to surface. The
missing side is recorded as `null`.

## What it writes

Into `$AGENT_SYS_OUTPUT_REVIEW`:

```
README.md          `## Purpose` and `## Schema`
items/text.json
```

```json
{
  "agreed": [
    {"student": "a", "problem_id": "p1",
     "implements_claimed_algorithm": true, "complexity_credible": true,
     "verdict": "accept",
     "comment_x": "...", "comment_y": "..."}
  ],
  "disagreed": [
    {"student": "b", "problem_id": "p1",
     "x": {"implements_claimed_algorithm": true,  "complexity_credible": true,
           "verdict": "accept", "comment": "..."},
     "y": {"implements_claimed_algorithm": false, "complexity_credible": true,
           "verdict": "revise", "comment": "..."}}
  ],
  "totals": {"pairs": 36, "agreed": 36, "disagreed": 0}
}
```

`agreed` keeps both comments, because the two reviewers' reasoning is the part
of the review a student can act on and there is no reason to discard half of it
just because the verdicts matched.

`totals.pairs` is the size of the **union** of the two documents' pairs, so
`agreed + disagreed == pairs` always holds and `check_reviews_agree` recomputes
it rather than trusting it.

## Why this is a program and not an agent

The comparison is `==` over three fields. There is no judgement in it, nothing
is learned by paying for a model call, and the one deterministic step between
two AI steps is the one place a run's flakiness can be driven to zero for free.

Ordering is by `(student, problem_id)` so that two runs over the same inputs
produce byte-identical output, which is what makes the handoff's digest mean
something.

## Layout

`entry.sh` is the command; `reconcile.py` is the whole implementation. Compare
this folder with `assets/review_x.task/`, which holds a `readme.md` and no
`entry.sh` — one file's difference is the whole of what "a program task" versus
"an agent task" means here.
