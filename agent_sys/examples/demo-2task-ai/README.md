# demo-2task-ai — two AI tasks and one handoff

This directory is **data**. It holds YAML specs and the programs they name, and
nothing in `agent_sys` imports any of it — there is no `__init__.py` here and
there never will be. It may use nothing a task package outside this repository
could not use: no privileged import, no private loader path, no schema of its
own.

```bash
pip install -e agent_sys
agent-sys run --package agent_sys/examples/demo-2task-ai
```

## What it is for

A **smoke test of `ai` + `task_graph`**. Two leaves, both `kind: ai`, one
handoff between them, three validators. It answers *does an agent run and does
the graph hand its output to the next task* in two model calls, which is what
makes it usable as a first check after a change to either component.

`../demo/` is a chain of three and `../demo2/` is a seven-node graph with
fan-out, fan-in and depth-2 nesting. Neither is cheap. This one deliberately
proves less and costs less.

## The graph

```
main                              non-leaf: readme, no entry.sh, NO agent
│                                 inputs [] · outputs [problems]
│
├── directions                    ai: teacher
│   │                             picks CLRS topics from assets/catalog/
│   │                             out: directions           [structured_text]
│   └── check_directions               completeness, strong
│
└── problems  ← directions        ai: setter          is_end
    │                             out: problems             [structured_text]
    ├── check_problems                 completeness, strong
    └── check_solvable                 trustworthiness, weak
```

Two handoff kinds, two leaves, one non-leaf, three validators.

`check_directions` runs **twice**: `directions`' output phase and `problems`'
input phase, against the same artefact. That is the one thing this package shows
that a single-task package could not — a phase is a position, not a kind of
validator.

## The scale knobs

Every count reaches its body through its agent's `env`, so a readme says
*"write `$TWO_TASK_N_PROBLEMS` problems"* and never a literal count — otherwise
the knob and the instruction disagree and the instruction wins.

| variable | default | what it sizes |
|---|---|---|
| `n_directions` | 2 | topics `directions` picks |
| `n_problems` | 3 | problems in the set |

The defaults are small because this package's purpose is to be run often. A
larger run is `--var n_problems=8`, not an edit.

## Layout

```
main.yaml       the outermost graph. Its name is fixed
steps/          one file per step, holding everything that step introduces
assets/         MANDATORY. every body found by filename convention
  main.task/            readme.md            (non-leaf: no entry.sh)
  directions.task/      readme.md            (ai: no entry.sh)
  problems.task/        readme.md            (ai: no entry.sh)
  <name>.validator/     readme.md, entry.sh, check.py
  lib/store.py                 reading a published handoff without importing `handoff`
  catalog/clrs_topics.json     the closed list check_directions matches
  catalog/leetcode_index.json  the closed list check_problems matches
```

**There is no `shared.yaml`**, because nothing here has two users. Both leaves
are AI and carry their own agent; each validator body is a script that
`validator.ScriptBodyRunner` runs directly, so the `runner` program agent that
`../demo2/shared.yaml` exists for has no user in this package.

**Nothing binds a filename.** There is no `body:` key anywhere. A folder named
`${name}.${type}` under `assets/` scopes the lookup, and inside it `readme.md`
and `entry.sh` are found by their own names. Binding one by hand is legal and
warns at compile time.

## What is copied from `../demo2/`

`assets/lib/store.py` and both catalogues are **byte-identical** to demo2's.
`store.py`'s own docstring sets the rule: it is a verbatim copy of demo-1's, an
agreement test pins demo-1's copy against `handoff`'s real constants, and a copy
that drifted would be a second unpinned reader of a layout `handoff` owns. Fix
the original and copy it across.

The three validators started as copies and are **not** byte-identical. Each one
documented how many phases it runs in and named `solve_a`/`_b`/`_c`; in a
two-task graph `check_problems` and `check_solvable` run once, not four times,
so those passages were corrected rather than carried across false.

## What it needs

Credentials for the Claude backend and a working sandbox. No compiler, no GPU,
no cluster, and **no network during the run**: the two catalogues under
`assets/catalog/` are shipped with the package precisely so that the closed
lists the validators match against do not depend on reaching leetcode.com.
