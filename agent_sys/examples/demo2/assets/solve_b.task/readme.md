# solve_b — student B, solve your share of the problem set

You are one of three students working the same problem set. Each of you takes a
**disjoint third of it**, so between you every problem is solved exactly once.

## Your share

Read the problem set from **`$AGENT_SYS_INPUT_PROBLEMS/items/text.json`**. It
holds `$DEMO2_N_PROBLEMS` problems in a fixed order. Number them from **0** in
the order they appear in that file.

> **You solve every problem whose index `i` satisfies `i % 3 == 1`.**

Student A takes `i % 3 == 0` and student C takes `i % 3 == 2`. Apply the rule to
the list you actually read — do not assume how many problems there are, and do
not solve one that is not yours. Solving another student's problem is not
harmless: it is the same work done twice and it is not what you were asked for.

Each problem carries these keys, and the names are exact:

| key | what it is |
|---|---|
| `id` | the problem's identifier, and the directory name you file it under |
| `title`, `statement`, `constraints` | what the problem is and what it bounds |
| `input_format`, `output_format` | two strings, the IO contract you must match |
| `examples` | one or more worked cases, each an `input` and the `output` it must produce |
| `direction_id`, `leetcode_slug` | where the problem came from; nothing for you to do |

`examples[].output` is the expected stdout, verbatim. An example may also carry
a `note`, which is prose for you and is not checked.

## What each solution has to be

Each solution is a **standalone C++ program**: it reads its input from **stdin**
and writes its answer to **stdout**, in exactly the format the problem's
`input_format` and `output_format` state. No arguments, no input files, no prompts or banners on stdout — the
output is compared against the expected output verbatim, line by line, with
trailing whitespace ignored.

Two hard requirements, both of which are checked by running your code:

- it **compiles** with `g++ -O2 -std=c++17`, with no errors;
- it produces the right answer for **every** example the problem ships, and each
  run finishes in **under 30 seconds**.

A source that does not compile fails the whole handoff, not just its own
problem, so a problem you cannot finish is better left with a correct simple
solution than with a clever one that does not build.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_SOLUTIONS_B`**. That is
your output handoff's content directory; it already exists and you are granted
write on it. Do not create anything beside it — `claim/` and `manifest.yaml` are
the system's to write, and the manifest is what makes the version published.

```
README.md                             with ## Purpose, ## Interface, ## Boundary
items/codes/<problem_id>/solution.cpp
items/codes/<problem_id>/notes.json
```

One directory per problem **you** solved, named by that problem's `id` exactly
as it appears in `text.json`.

**`items/codes/` is not decoration.** This handoff's `content_type` is `code`,
and that type requires an item named `codes` and nothing else at the top level
(`handoff/content.py`). A `solution.cpp` placed directly under `items/` is
rejected before anyone reads it.

### `notes.json`

One per problem, beside its source, with exactly these five keys:

```json
{
  "problem_id": "<the same id as the directory name>",
  "algorithm": "sliding window over the prefix sums",
  "time_complexity": "O(n log n)",
  "space_complexity": "O(n)",
  "rationale": "why this approach, and why the complexity is what you claim"
}
```

`algorithm` must be a real name or a real description, not a restatement of the
problem. `time_complexity` and `space_complexity` must be written in big-O —
`O(1)`, `O(n)`, `O(n log n)`, `O(n^2)`, `O(m + n)`, `O(2^n)` and the like.

State the complexity your program **actually** has. Nothing here verifies that
claim against your code — `check_analysis` is a `weak` validator and says so in
its own readme — so it is on you, and a wrong number that survives the check is
worse than no number at all.

### `README.md`

Three sections, all required, all checked for being non-empty and for not being
a placeholder — "to be filled in" is rejected:

- **`## Purpose`** — what this handoff is: student B's solutions to their share.
- **`## Interface`** — how the programs are invoked: one binary per problem,
  stdin to stdout, the compiler and standard they need.
- **`## Boundary`** — what is *not* here. Name the problems you did not solve
  because they were not your share, and anything you left incomplete.

Do not claim in the README that anything was verified unless you actually ran
it. The compile-and-run check is separate and it will disagree with you.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" versus "a program task" means in
this system.
