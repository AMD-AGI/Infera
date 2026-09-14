# optimise — one better implementation per problem

Three students solved the same set of problems and a grader scored them. Your
job is to produce, for each problem, **one** implementation that is better than
all three.

## What you are given

| where | what |
|---|---|
| `$AGENT_SYS_INPUT_PROBLEMS` | the problem set: `items/text.json`, each problem with an id, a statement, an IO format and worked examples |
| `$AGENT_SYS_INPUT_SOLUTIONS_A` | student A's sources, one directory per problem under `items/codes/` |
| `$AGENT_SYS_INPUT_SOLUTIONS_B` | student B's, same shape |
| `$AGENT_SYS_INPUT_SOLUTIONS_C` | student C's, same shape |
| `$AGENT_SYS_INPUT_SCORES` | the grader's verdict: `items/text.json`, a score per student per problem |

Each is the content directory itself — `items/` is directly inside it, with no
`content/` hop. Read all five. The scores are what tell you which of the three
attempts to start from; the problem statement is what tells you whether a
rewrite is still answering the question that was asked.

## What to do, per problem

1. **Pick a baseline.** Read the three sources and the scores. The
   best-scoring implementation is your starting point. Record which student it
   was — that is `baseline_student`, and it is `a`, `b` or `c`.
2. **Read it properly**, and name the algorithm it uses. Not "a loop" — the
   name a textbook would use: `two pointers`, `sieve of Eratosthenes`,
   `Kadane`, `binary search on the answer`, `brute force O(n^2)`.
3. **Write a better one.** Better means **shorter**, or **a better
   algorithm**, and ideally both. It must still be correct.

A rewrite that is neither shorter nor algorithmically different is not an
improvement, however much nicer it reads, and it is rejected. If after reading
the baseline you conclude it genuinely cannot be improved, say so in
`what_changed` and `why_faster` and submit your honest attempt anyway — a
recorded failure is worth more than a cosmetic reshuffle presented as a win.

### What the new source has to be

A **standalone stdin/stdout program**. It is compiled on its own with

```
g++ -O2 -std=c++17 -o solution solution.cpp
```

and then run with each worked example's input on stdin; its stdout must match
that example's expected output, within 30 seconds. No arguments, no input
files, no second translation unit, no headers of your own beside it. If it does
not compile, or it answers even one worked example wrongly, nothing else about
it counts.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_OPTIMISED`**. That is
your output handoff's content directory; it already exists and you are granted
write on it. Do not create anything beside it — `claim/` and `manifest.yaml`
are the system's to write, and the manifest is what makes the version
published.

```
README.md                              with ## Purpose, ## Interface, ## Boundary
items/codes/<problem_id>/solution.cpp
items/codes/<problem_id>/report.json
```

`<problem_id>` is the problem's id from `problems`, spelled exactly as it
appears there. **Everything goes under `items/codes/`** and not directly under
`items/`: the `optimised` kind is `content_type: code`, which defines one item
named `codes` and refuses any item it never defined, so a directory per problem
at the top level fails the content check before anything is run.

The three README sections are required by the same content type and the run
fails without them. A section that is empty, or that says *"to be filled in"*,
is rejected too — the check reads the body, not just the heading.

- **`## Purpose`** — what this artefact is and what it is for.
- **`## Interface`** — how one of these programs is built and invoked: the
  compile line above, stdin in, stdout out.
- **`## Boundary`** — what is *not* here. Say plainly what you did not verify:
  you did not benchmark against the baselines on large inputs, and you did not
  prove correctness beyond the worked examples.

### `report.json`, one per problem

```json
{
  "problem_id": "<the id, matching the directory name>",
  "baseline_student": "a | b | c",
  "baseline_lines": 0,
  "new_lines": 0,
  "baseline_algorithm": "<the name of the algorithm the baseline uses>",
  "new_algorithm": "<the name of the algorithm yours uses>",
  "what_changed": "<one or two sentences: what you actually did>",
  "why_faster": "<which claim you are making, and on what grounds>"
}
```

All eight keys are required and none may be empty.

`baseline_lines` and `new_lines` are counted the same way: **lines that are
neither blank nor a comment.** Count them, do not estimate them.

`why_faster` must say **which** of the two claims you are making — *"fewer
lines: 41 down to 22, same algorithm"*, or *"different algorithm: the baseline
sorts and scans, this one is a single pass with a hash map"*. Both, if both are
true. It is the sentence a reviewer reads first, so it must be checkable
against the two `*_algorithm` fields and the two line counts, and it must not
claim a speed-up you have not measured.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** Compare this
folder with a program task's: one file's difference is the whole of what "an
agent task" versus "a program task" means in this system.
