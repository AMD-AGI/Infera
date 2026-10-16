# review_y — describe the code first, then check the claim against it

You are the second of two reviewers. Another reviewer has been given the same
submissions and is working now; you cannot see their answers and they cannot
see yours. Afterwards a program compares the two reviews field by field, and a
validator fails the run if you disagree anywhere.

That comparison is only informative because the two of you worked apart, so
**do not attempt to anticipate the other reviewer.** Answer from the code.

## The method: describe, then compare

The failure mode this review exists to catch is a submission whose *claim* and
whose *code* are different things. The way to miss it is to read the claim
first and then go looking for it — the claim tells you what to see, and code is
almost always compatible with a description somebody has already handed you.

So the order here is inverted, and it is not optional:

1. **Read the code with the claim covered.** Write down, for yourself, what the
   algorithm actually is and what its running time actually is. Count the
   nested loops. Note the recursion and its branching factor. Note the data
   structure the work is done in.
2. **Only then read the student's claim.**
3. **Compare your description with theirs.** The two questions below are that
   comparison, nothing more.

## Sweep by problem, not by student

Take one problem and read all three submissions for it before moving on. Three
solutions to one problem are directly comparable — the same statement, the same
constraints, the same shape of answer — and a claim that looked plausible on
its own often stops looking plausible beside two submissions doing something
else. This is the opposite of going student by student, and it is deliberate.

Comparing them is a *reading aid*, not a rule. Three students may legitimately
choose three different algorithms, and a submission is never wrong for being
the odd one out. Judge each against its own claim.

## What you are given

| variable | what is in it |
|---|---|
| `$AGENT_SYS_INPUT_PROBLEMS` | the problem set — `items/text.json`, one entry per problem |
| `$AGENT_SYS_INPUT_SOLUTIONS_A` | student **a**'s submissions — `items/codes/`, one directory per problem |
| `$AGENT_SYS_INPUT_SOLUTIONS_B` | student **b**'s submissions, same shape |
| `$AGENT_SYS_INPUT_SOLUTIONS_C` | student **c**'s submissions, same shape |

Each submission carries the C++ source and the student's own statement of the
algorithm and its complexity. Where they wrote it is up to them — the
solution directory's `README.md`, or a comment at the head of the source. Look
in both.

## The two questions

**`implements_claimed_algorithm`** — is the description you wrote in step 1 the
same algorithm the student named? Not *does it work*: whether the code produces
correct answers is measured by running it, and running it is not your job. A
correct program that is not the algorithm it claims is `false`. A buggy
implementation of the right algorithm is `true`.

**`complexity_credible`** — is the bound the student stated a defensible reading
of the code you just described? Derive your own bound from the loop nesting and
the recursion, then compare. Reject a bound that is the textbook figure for the
named algorithm but not a description of what is written. Accept a bound that
is loose but true.

**`verdict`** — `accept` when both of the above are `true`; `revise` otherwise.
Nothing else feeds it.

**`comment`** — one or two sentences. Say what you found the code to be doing,
and where. Cite the loop, the call or the line that settled it. Do not simply
repeat the verdict.

## What "I could not tell" means here

It means `false`, `revise`, and a comment naming the obstacle: a claim you
could not find, a source you could not follow, a file that is not there. It
does not mean `accept`. Silence and correctness look identical from the
outside, and only one of them has been demonstrated.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_REVIEW_Y`**. It already
exists and you are granted write on it. Do not create anything beside it —
`claim/` and `manifest.yaml` belong to the system, and the manifest is what
makes the version published.

Two files, both required:

```
items/text.json   the review, exactly the shape below
README.md         with a `## Purpose` and a `## Schema` section
```

`review_y` is `content_type: structured_text`, which requires one of
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
      "comment": "Two nested loops over n at lines 14-22; the claimed O(n) is not that."
    }
  ]
}
```

The shape is fixed and is checked, because the other reviewer's document is
compared with yours key by key. A field you rename, or a type you change, reads
downstream as a disagreement:

- `student` is `"a"`, `"b"` or `"c"` — the bare letter, lower case.
- `problem_id` is spelled **exactly** as the problems artefact spells it.
- `implements_claimed_algorithm` and `complexity_credible` are JSON booleans,
  not `"true"` / `"false"` strings.
- `verdict` is `"accept"` or `"revise"`.
- `comment` is a non-empty string.
- **One row per submitted (student, problem) pair**, no more and no fewer. A
  pair nobody submitted has no row; a pair all three submitted has three.
- No pair twice.

Under `## Schema` in the README, describe those fields in your own words, and
record how many pairs you reviewed and how many you marked `revise`.
