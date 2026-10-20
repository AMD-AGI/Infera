# Content layouts per content type

The README sections and required items are agent_sys's (`handoff/content.py`);
the rest maps experiment-result-packup's reproduction kit onto them.

| content_type | packup tree? | where the seven things go |
|---|---|---|
| `code` | **yes** — `items/codes/<name>.packup_<date>/` is the tree | inside the tree |
| `reproducible` | no — its items *are* the kit (`result`, `env`, `command`, …) | items, plus the five README sections |
| `structured_text`, `text` | no | README sections (+ extra items only if `items_schema` admits them) |

Extra items beyond these are legal unless the kind's `items_schema` sets
`additionalProperties: false`; then use README sections instead.

## `reproducible` — a result someone can re-run

```
README.md            ## Purpose / ## How to run / ## Result / ## Environment / ## Watch out
items/result/        machine-readable numbers (summary.json, *.csv) + plots; what the claim rests on
items/env/           environment.md (astk env), environment.yaml if the package defines one,
                     image digest, git commit; secrets by NAME only
items/command        executable (#!/bin/sh or bash), zero -> result, the exact commands that ran
items/script/        optional: the scripts `command` calls, copied verbatim
items/code/          optional: patches (*.patch + a what/why/how note each)
items/logs/          optional, only if the user said yes (trimmed / gzipped)
items/watchout       optional: traps, wrong turns (notes.md content)
```

`How to run` = the command item, in prose; `Result` = the numbers vs. the
success criteria; `Watch out` = the top traps.

## `code` — a kit, a patch set, a packup tree

```
README.md            ## Purpose / ## Interface / ## Boundary
items/codes/<name>.packup_<YYYYMMDD>/      the reproduction kit:
    README.md        what / why / result at a glance + folder map
    REPRODUCE.md     ordered copy-pasteable commands, prerequisites, expected output
    environment.md   hardware + software, pinned (astk env)
    scripts/         verbatim
    patches/         *.patch + note each
    results/         CSV / JSON / PNG
    notes.md         gotchas as what / why / how / context
    logs/            only on the user's yes
items/logs/          optional
items/watchout       optional
```

`Interface` = how a consumer uses it (entry points, arguments, the files it
reads/writes); `Boundary` = what it does not do, what it assumes.

## `structured_text` — a record another program reads

```
README.md            ## Purpose / ## Schema
items/text.json      (or text.yaml / text.xml) — the record
items/schema         optional: JSON Schema of the record
items/env/           when the package convention asks for it (e2e-flow does)
```

## `text` — prose

```
README.md            ## Purpose
items/content        the text
```

## Package conventions win

A package may add rules of its own: e2e-flow requires `environment.yaml` in
every handoff (`items/env/` or `items/codes/`) validated by
`assets/schemas/environment.schema.json`. Read the validators' readmes; they
are the bar, and `astk check` runs them.

## README rules the store enforces

- Each section is a **document-root heading** with the exact name; any level
  works, `# Title` on top is fine. Headings inside code fences do not count.
- Each section has real text; `[More Information Needed]` (the template body)
  is refused.
