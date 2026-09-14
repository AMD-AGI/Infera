# check_directions — completeness, strong

The direction list is the requested length, every entry is drawn verbatim from
the closed CLRS catalogue, no `id` repeats, and every `why` is prose.

## Why `strong` is honest here

The check is **total** over the document — every row, every declared key — and
three of its four rules are exact:

| rule | how it is decided |
|---|---|
| the count | `len(directions) == args.expected_count` |
| the catalogue | the pair `(clrs_chapter, leetcode_tag)` is a row of `assets/catalog/clrs_topics.json`, compared character for character with no normalisation |
| uniqueness | `id` seen twice |
| `why` is prose | **crude, and stated**: at least 24 characters and at least 4 words after stripping |

The fourth is the only approximation and it is written down rather than
implied. `TBD`, `important` and a bare topic name all fail it; any sentence
someone actually wrote passes it with room to spare. A crude check that is
honestly described is a strong validator; a sophisticated one that is silently
approximate is not — which is the distinction
`examples/demo/assets/check_grounded.validator/readme.md` draws about its own
`\d+`.

## What it will not tell you

Whether the five areas are a *good* syllabus. Whether they are the five a
particular course should cover, whether they are in a sensible order, whether
the `why` is true. Those are judgements, this is a completeness check, and the
dimension on the spec is the honest name for what it does.

## The catalogue shape it reads

```json
{"topics": [
  {"id": "divide_and_conquer",
   "clrs_chapter": "Ch. 4 -- Divide-and-Conquer",
   "leetcode_tag": "divide-and-conquer"}
]}
```

A catalogue that is missing, unparseable, or has no non-empty `topics` list
makes **every handoff False** and prints why. That is deliberate and it is the
one place this file departs from the obvious code: returning an empty permitted
set instead of `None` would fail every direction too, and would read exactly
like a teacher who invented all five. *Could not check* and *checked and found
nothing* must not report identically.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json`, `inputs.json` and `materials.json` beside it,
and it writes `verdict.json` — one boolean per handoff id in `inputs.json`.

It runs **twice per graph**: in `directions`' output phase, and again in
`problems`' input phase against the same artefact. A phase is a position, not a
kind of validator.

Both counts reach it through `args.json` rather than the environment, and that
is measured rather than stylistic. `validator.choose_configuration`
(`validator/environment.py:116-142`) uses the producing task's configuration on
the **output** phase only, and there is no source for the consumer row at all,
so an input phase falls through to the global row —
`{PATH, AGENT_SYS_DEMO_PACKAGE, AGENT_SYS_DEMO_STORE, AGENT_SYS_DEMO_PYTHON}`
(`cli/main.py:601-615`). `DEMO2_N_DIRECTIONS` is not in it. `args.json` is
written by the phase runner and does not vary by phase, and the `${...}` in the
spec is expanded at load time over every string in the file
(`spec_loader/package.py:335`), so one `--var n_directions=2` still moves the
prompt and the check together.
