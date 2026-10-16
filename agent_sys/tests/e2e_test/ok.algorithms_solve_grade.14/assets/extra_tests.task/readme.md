# extra_tests — new cases, worked out by hand from the statement

The problems ship with a few worked examples. Those examples are what every
student already tested against, so they measure nothing: a solution that
handles exactly the shipped examples and nothing else passes them all.

Your job is to write the cases that were not shipped.

## How many

**`$DEMO2_N_EXTRA` new cases for every problem in the problem set.** Read the
number out of the environment; do not assume a value. A bring-up run sets it
small on purpose, and a body with a literal in it silently ignores that.

The same number for each problem. Not more for the interesting ones.

## What you are given

| variable | what is in it |
|---|---|
| `$AGENT_SYS_INPUT_PROBLEMS` | the problem set — `items/text.json`, one entry per problem, each with its statement, its IO format and its worked examples |
| `$AGENT_SYS_INPUT_HARNESS` | the built test harness — `items/manifest.json` lists every case id the one binary answers to |

Read the manifest. It tells you which `<student>/<problem_id>` pairs exist and,
crucially, **how `problem_id` is spelled**. Every case you write is keyed on
that spelling.

## The two rules, and they are both checked

### 1. Nothing may duplicate a worked example

A case whose `input` is one of the shipped examples for that problem is
rejected, and the whole artefact fails with it. Comparison ignores leading and
trailing whitespace, so re-indenting an example does not make it new.

This is the rule the task exists for. Before writing a case, read that
problem's worked examples and make sure you are not about to re-type one.

### 2. The expected output is **derived**, never guessed

For each case you invent, work the answer out from the statement yourself — by
hand, on the input you just wrote. Do not run a student's solution and record
what it printed: those solutions are the thing being tested, and a test whose
expected output came from the program under test cannot fail.

If you cannot work out the answer for an input you invented, **throw the input
away and invent an easier one.** A case with a guessed expectation is worse
than no case: it fails a correct solution, and the failure looks exactly like a
bug in the student's code.

## What makes a good set of cases

Spend the budget on the places where implementations actually differ, not on
more of the same:

- the smallest legal input, and the empty or single-element case if the
  statement permits one;
- the boundary the statement names — the maximum value, the maximum length,
  the last index;
- ties, duplicates, and every-element-equal;
- already-sorted and reverse-sorted input, where order is in play;
- negatives and zero, where the statement allows them;
- an input whose answer is the degenerate one — zero, empty, "no solution" —
  when the statement defines that case.

Each case carries a `why`, and the `why` is what stops the set from being ten
variations of one idea. Write it before the case if that helps.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_EXTRA_TESTS`**. It
already exists and you are granted write on it. Do not create anything beside
it — `claim/` and `manifest.yaml` belong to the system, and the manifest is
what makes the version published.

Two files, both required:

```
items/text.json   the cases, exactly the shape below
README.md         with a `## Purpose` and a `## Schema` section
```

`extra_tests` is `content_type: structured_text`, which requires one of
`text.json` / `text.yaml` / `text.xml` and a README carrying `Purpose` and
`Schema`. A README whose sections are empty, or that says *"to be filled in"*,
is rejected by the content check and the run fails.

`items/text.json`:

```json
{
  "cases": [
    {
      "problem_id": "<the id, spelled as the problems artefact spells it>",
      "input": "5\n1 1 1 1 1\n",
      "expected": "5\n",
      "why": "every element equal — the tie-breaking branch nothing else reaches"
    }
  ]
}
```

- `problem_id` is spelled **exactly** as the problems artefact and the harness
  manifest spell it. A case keyed on an id nothing recognises is never run.
- `input` is the complete stdin for one run of the program, as a single string,
  newlines included. Not a list of lines, not a fragment.
- `expected` is the complete stdout the statement says that input produces.
  Trailing whitespace does not matter; the content does.
- Neither `input` nor `expected` may be empty. A problem whose correct answer
  is genuinely nothing is not a case you can express here — pick another.
- `why` is one short line saying what this case exercises that the others do
  not.

Under `## Schema` in the README, describe those fields in your own words, and
record how many cases you wrote per problem and the value of `$DEMO2_N_EXTRA`
you read.
