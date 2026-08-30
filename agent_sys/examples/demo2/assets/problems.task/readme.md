# problems — write the problem set

You are the teaching assistant. The teacher has set the course's directions; you
turn them into problems a student can actually sit down and solve.

Write exactly **`$DEMO2_N_PROBLEMS`** problems. That is an environment variable
and it is set for you; read it rather than assuming a number.

## What you are given

The directions artefact is at **`$AGENT_SYS_INPUT_DIRECTIONS`** — the content
directory itself, so the document is
`$AGENT_SYS_INPUT_DIRECTIONS/items/text.json`. Each row carries an `id`, a
`title`, a `clrs_chapter`, a `leetcode_tag` and a `why`.

**Spread the problems across the directions.** Every direction the teacher set
must get at least one problem, and no direction may take a run of the set so
long that another is left with none. If the count does not divide evenly, put
the remainder where the direction is richest and say nothing about it — the
spread is a judgement, not an arithmetic rule.

Every problem names the direction it came from, by that direction's `id`. An
`id` that is not in the directions artefact is a failure.

## Every problem must be a real one

`leetcode_slug` must be the slug of a problem that exists. As with the
directions, you are not asked to remember which: the permitted set is written
down and closed.

```
$AGENT_SYS_TASK_PACKAGE/assets/catalog/leetcode_index.json
```

```json
{"problems": [
  {"slug": "merge-k-sorted-lists",
   "title": "Merge k Sorted Lists",
   "topic_id": "divide_and_conquer",
   "difficulty": "hard"}
]}
```

Choose slugs from it and **copy them verbatim**. Each row's `topic_id` is the
`id` of a row in `clrs_topics.json`, which is also where the directions came
from — so **prefer a slug whose `topic_id` equals the `direction_id` you are
writing it under**. That is what makes the problem an example of the direction
rather than an unrelated exercise.

Prefer, not must. The check is on the slug existing, so a defensible near-fit
under a neighbouring topic is better than an invented slug; and if a direction
has fewer index rows than the problems you owe it, spread the extras over the
closest topics rather than inventing.

You are writing the problem in your own words. You are not reproducing
LeetCode's text; the slug says which classic problem this is a version of.

## stdin and stdout, and why it is not negotiable

**Every problem must be answerable by a standalone program that reads standard
input and writes standard output.** Downstream of you, three students each
compile a single self-contained program per problem and run it against your
worked examples. There is no function signature, no test harness with a class to
fill in, and no framework — a program, a stream in, a stream out.

So:

- `input_format` describes what the program will find on stdin: how many lines,
  what is on each, in what order, separated how.
- `output_format` describes exactly what the program must print, including
  whether there is a trailing newline and what to print when the answer is empty
  or absent.
- Every `examples` entry carries the **exact bytes**: `input` is what is fed to
  stdin, `output` is what must come out, character for character. Not a
  description of them, not a rendering with `>>>` prompts in it.

Give **at least two** worked examples per problem. One example illustrates; two
constrain the format. Include a third when the second only repeats the shape of
the first — an edge case, an empty input, a tie.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_PROBLEMS`**. That is your
output handoff's content directory; it already exists and you are granted write
on it. Do not create anything beside it — `claim/` and `manifest.yaml` are the
system's to write.

```
items/text.json   the problem set, and nothing else
README.md         with a `## Purpose` and a `## Schema` section
```

Both are required and the run fails without either: the `problems` kind is
`content_type: structured_text`, which requires one of
`text.json` / `text.yaml` / `text.xml` and a README carrying `Purpose` and
`Schema`. A README whose sections are empty or say *"to be filled in"* is
rejected too.

`items/text.json`:

```json
{
  "problems": [
    {
      "id": "p01",
      "direction_id": "sorting",
      "leetcode_slug": "sort-colors",
      "title": "Three-way partition",
      "statement": "Given n items each coloured 0, 1 or 2, print them sorted ascending. Do it in one pass.",
      "input_format": "Line 1: n. Line 2: n integers in 0..2, space separated.",
      "output_format": "One line: the n integers sorted ascending, space separated, trailing newline.",
      "examples": [
        {"input": "6\n2 0 2 1 1 0\n", "output": "0 0 1 1 2 2\n", "note": "the worked case from the statement"},
        {"input": "1\n0\n", "output": "0\n", "note": "single item, nothing to swap"}
      ],
      "constraints": "1 <= n <= 300; each value is 0, 1 or 2."
    }
  ]
}
```

Every key is required on every problem, and none of them may be empty:

| key | what it is |
|---|---|
| `id` | `p01`, `p02`, … Unique across the set, and **filesystem-safe**: lower-case letters, digits and underscores only. Downstream it becomes a directory name, one per problem, so a space or a slash in it is a broken path rather than an untidy label |
| `direction_id` | the `id` of a direction in the input artefact |
| `leetcode_slug` | a slug from the index, verbatim |
| `title` | the problem's name, as you would print it |
| `statement` | what is being asked, in full, in prose |
| `input_format` | what is on stdin |
| `output_format` | what must be on stdout |
| `examples` | at least two, each `{input, output, note}`, all three non-empty |
| `constraints` | the bounds a solution may rely on |

Under `## Schema` in the README, describe that document in your own words, and
say which directions the set covers.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That is what
makes this an agent task rather than a program one.
