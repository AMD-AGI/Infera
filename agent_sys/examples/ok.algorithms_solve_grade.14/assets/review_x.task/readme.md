# review_x — does the code do what its author says it does?

You are the first of two reviewers. The other one is working on the same
submissions at the same time and cannot see your answers, and a program called
`reconcile` will compare the two of you field by field afterwards. **Do not try
to guess what the other reviewer will say.** Agreement is only worth something
if it was reached independently.

You are **not** asked whether a solution is fast, elegant, or well named. You
are asked two narrow questions about each submission, and nothing else.

## What you are given

Four directories, each named by an environment variable:

| variable | what is in it |
|---|---|
| `$AGENT_SYS_INPUT_PROBLEMS` | the problem set — `items/text.json`, one entry per problem |
| `$AGENT_SYS_INPUT_SOLUTIONS_A` | student **a**'s submissions — `items/codes/`, one directory per problem |
| `$AGENT_SYS_INPUT_SOLUTIONS_B` | student **b**'s submissions, same shape |
| `$AGENT_SYS_INPUT_SOLUTIONS_C` | student **c**'s submissions, same shape |

Each student's submission for a problem carries the C++ source **and the
student's own claim** about it: which algorithm they used and what its
complexity is. Where that claim is written is the students' business — read the
solution directory's `README.md` and the source's own comments and take the
claim from wherever it actually is. If you cannot find a claim at all, that is
itself an answer: see *"When you cannot tell"* below.

## The two questions

For every (student, problem) pair that was actually submitted:

**1. `implements_claimed_algorithm`** — read the code and decide whether it is
the algorithm the student says it is. A submission that claims a monotonic
stack and implements a nested loop is `false` even if it produces the right
answers. A submission that claims binary search and implements binary search
with an off-by-one is `true` for *this* question — correctness is measured by
running the code, which is somebody else's job, and conflating the two is the
mistake this field exists to avoid.

**2. `complexity_credible`** — decide whether the stated complexity is a
defensible description of the code as written. Count the loops and the
recursion; do not accept a bound because it is the bound the textbook gives for
that algorithm. `O(n log n)` beside a sort is credible; `O(n)` beside a sort is
not.

Then one summary field:

**`verdict`** — `accept` if both answers above are `true`, `revise` otherwise.
It is a function of the other two and nothing else. Do not use it to express an
opinion the two questions did not ask for.

And **`comment`**: one or two sentences saying *why*, naming the specific line,
loop or call that decided it. A comment that restates the verdict without
evidence is not useful to the student and is not what this field is for.

## Working method

Go student by student, problem by problem. For each pair:

1. Read the problem statement first, so you know what the algorithm is supposed
   to be doing.
2. Read the student's claim.
3. Read the code **before** forming a view about the claim. Reading the claim
   first and then looking for it in the code is how a reviewer confirms
   whatever they were told.
4. Write the row.

## When you cannot tell

Say so, and record it as a failure rather than as a pass. A submission whose
claim you cannot find, or whose code you cannot follow well enough to judge,
gets `false` for the question you could not answer, `revise`, and a comment
that says exactly what was missing or unreadable. **Never write `true` because
nothing looked wrong.** An unread submission and a correct submission produce
the same absence of evidence, and only one of them deserves an `accept`.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_REVIEW_X`**. It already
exists and you are granted write on it. Do not create anything beside it —
`claim/` and `manifest.yaml` belong to the system, and the manifest is what
makes the version published.

Two files, both required:

```
items/text.json   the review, exactly the shape below
README.md         with a `## Purpose` and a `## Schema` section
```

`review_x` is `content_type: structured_text`, which requires one of
`text.json` / `text.yaml` / `text.xml` and a README carrying `Purpose` and
`Schema`. A README whose sections are empty, or that says *"to be filled in"*,
is rejected by the content check and the run fails.

`items/text.json`:

```json
{
  "reviews": [
    {
      "student": "a",
      "problem_id": "<the problem's id, exactly as the problems artefact spells it>",
      "implements_claimed_algorithm": true,
      "complexity_credible": true,
      "verdict": "accept",
      "comment": "Monotonic stack in the single pass at line 21; O(n) is right."
    }
  ]
}
```

Hard requirements, each of which is checked:

- `student` is one of `"a"`, `"b"`, `"c"` — the letter, lower case, nothing else.
- `problem_id` is the id **as the problems artefact spells it**. Do not
  reformat it, do not prefix it, do not renumber.
- `implements_claimed_algorithm` and `complexity_credible` are JSON booleans,
  not the strings `"true"` and `"false"`.
- `verdict` is `"accept"` or `"revise"`.
- `comment` is a non-empty string.
- **Exactly one row per (student, problem) pair that was submitted.** Not one
  per problem; not one per student. If student `b` submitted nothing for a
  problem, there is no row for that pair — and if all three submitted, there
  are three rows for it.
- No pair appears twice.

Under `## Schema` in the README, describe those fields in your own words and
say how many rows you wrote and over how many pairs.
