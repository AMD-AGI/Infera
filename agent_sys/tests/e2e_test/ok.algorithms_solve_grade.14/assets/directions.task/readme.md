# directions — set the syllabus

You are the teacher of a university algorithms course. Before anyone writes a
problem, decide **which areas the course will cover**.

Choose exactly **`$DEMO2_N_DIRECTIONS`** directions. That is an environment
variable and it is set for you; read it rather than assuming a number, because a
short bring-up run sets it to two and a real run sets it to five.

## What counts as a direction

A *classic* area of algorithms — the kind a textbook devotes a chapter to and an
interview site devotes a tag to. Every direction you choose must be present in
**both**:

- CLRS (*Introduction to Algorithms*), as a chapter or a run of chapters, and
- LeetCode, as one of its topic tags.

You do not have to work out for yourself which areas satisfy that. The
intersection has already been written down, and it is a **closed list**:

```
$AGENT_SYS_TASK_PACKAGE/assets/catalog/clrs_topics.json
```

Read that file and choose from it. Its shape is:

```json
{"topics": [
  {"id": "divide_and_conquer",
   "clrs_chapter": "Ch. 4 -- Divide-and-Conquer",
   "leetcode_tag": "divide-and-conquer"}
]}
```

**Copy `clrs_chapter` and `leetcode_tag` verbatim** from the row you picked,
including the `--` and the lower-case hyphenated tag. Do not reformat them, do
not expand `Ch.` to `Chapter`, do not tidy the tag into title case.
The check that follows compares the two strings as a pair against the catalogue,
character for character; a paraphrase is a failure, and it is the one failure
mode of this task that has nothing to do with your judgement.

Choose **`$DEMO2_N_DIRECTIONS` different** rows. Two directions with the same
`id` are a failure even if everything else about them differs.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_DIRECTIONS`**. That is
your output handoff's content directory; it already exists and you are granted
write on it. Do not create anything beside it — `claim/` and `manifest.yaml` are
the system's to write, and the manifest is what makes the version published.

Two things go in it:

```
items/text.json   the directions themselves, and nothing else
README.md         with a `## Purpose` and a `## Schema` section
```

Both are required and the run fails without either: the `directions` kind is
`content_type: structured_text`, which requires one of
`text.json` / `text.yaml` / `text.xml` and a README carrying `Purpose` and
`Schema`. A README whose sections are empty or say *"to be filled in"* is
rejected as well.

`items/text.json`:

```json
{
  "directions": [
    {
      "id": "sorting",
      "title": "Comparison sorting and its lower bound",
      "clrs_chapter": "Ch. 2, 6-8 -- Sorting and Order Statistics",
      "leetcode_tag": "sorting",
      "why": "Students meet the n log n lower bound here for the first time, and it is the argument every later hardness claim is modelled on."
    }
  ]
}
```

Every key is required on every row:

| key | what it is |
|---|---|
| `id` | a short slug, unique across the list. Reuse the catalogue's `id` |
| `title` | the direction as you would print it on a syllabus |
| `clrs_chapter` | copied verbatim from the catalogue row |
| `leetcode_tag` | copied verbatim from the catalogue row |
| `why` | a sentence or two of **prose** saying why the course covers this |

`why` is read by a person and it is checked for being real prose rather than a
placeholder — a few words is not enough. Say what the area teaches that the
others do not.

Under `## Schema` in the README, describe the document above in your own words,
so a reader of the artefact alone knows what they are holding.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" versus "a program task" means in
this system.
