# The demo task package

This directory is **data**. It holds YAML specs and the programs they name,
and nothing in `agent_sys` imports any of it — there is no `__init__.py` here
and there never will be. It is the only *working* task package that lives in
this repository — `../demo-broken/` is the second directory that is structurally
one, and it exists so that criterion 11 has a closure that fails to load — and
the exception is deliberate: the demo is the system's
falsifiable claim and the first thing a reviewer runs, so putting it elsewhere
would mean *"clone two things to see whether this works"*.

The exception is bounded by one rule: **this package may use nothing a task
package outside the repository could not use.** No privileged import, no private
loader path, no schema of its own. If it needs something the system does not
offer to everyone, that is a missing feature and not a demo detail.

```bash
pip install -e agent_sys
agent-sys show
```

---

## The graph

```
system whole task
└── main                    parent = None. A non-leaf: readme, no entry.sh
    │                       NO agent. A non-leaf declares none; the system
    │                       supplies one, and `compose` is gone with the rule
    │
    ├── produce             kind: program — runs assets/produce.task/collect.py
    │   │                   input validation:  EMPTY. It has no inputs
    │   │                   output: facts                    [structured_text]
    │   └── output validation
    │       └── check_facts          completeness / strong        PASSES
    │
    ├── describe            kind: ai, backend claude-agent-sdk
    │   │                   input validation:  POPULATED — check_facts on `facts`
    │   │                   output: summary                  [text]
    │   └── output validation
    │       └── check_grounded       trustworthiness / strong     FAILS
    │
    └── consume             kind: program — would render the report
                            input: summary. Never runs; its input never becomes valid
```

The run ends **quiescent with `consume` still in `WAITING_HANDOFF`**. That is
the correct outcome, and the demo reports it as the expected one rather than as
a crash.

## Why `check_grounded` fails, and why that is not rigging

A demo whose failure is arranged proves nothing, so this one is **structural**
and comes from the handoff contract rather than from the validator's taste:

- The `summary` kind's contract is one sentence — *every number appearing in the
  summary must also appear in the `facts` artefact it summarises.* That is
  groundedness, it is `trustworthiness / strong`, and it is checkable by
  extracting the numerals from both and testing set inclusion.
- `describe`'s goal asks for a short summary **including how long the collection
  took**.
- `facts` carries no duration. `assets/produce.task/collect.py` does not measure
  one and the kind does not declare one.

So the agent is asked, in good faith, for a figure its input cannot ground. The
failure does not depend on the model behaving badly, on a prompt trick, or on a
validator that is secretly `return False`. It depends on **the task having been
specified with a gap in it** — which is the failure this whole system exists to
catch.

If the model ever answers *"the facts do not record a duration"*, there is no
ungrounded numeral, the verdict passes, and **the demo fails loudly with exit
code 3**. An expected failure that passes is a failure; pytest calls that
`xfail(strict=True)` and here it is not optional.

## The three bodies, and the one file that separates them

`closure` spec §2.6: every task has a `readme.md`; a programmatic one adds an
`entry.sh`.

| Task | Body | What you see |
|---|---|---|
| `produce` | `readme.md` + `entry.sh` | The exact command that ran. Reproducible by hand, outside the system |
| `describe` | `readme.md` **only** | The instruction the agent worked from — the same kind of file, read by a different executor |
| `consume` | `readme.md` + `entry.sh` | Never runs; its readme still says what it would have done |
| `main` | `readme.md`, no `entry.sh` | A non-leaf. Its work is its subgraph, and the two are mutually exclusive |

**That `describe` and `produce` differ by one file is the point.** Comparing the
two folders tells you the whole of what "an agent task" versus "a program task"
means here, without reading a design document.

## Layout

```
main.yaml       the outermost graph. MANDATORY, and its name is fixed
shared.yaml     what more than one step uses — today, the `collect` agent
steps/          one file per step, holding everything that step introduces
  produce.yaml    check_facts, facts, produce
  describe.yaml   describe (agent), check_grounded, summary, describe (task)
  consume.yaml    consume
assets/         MANDATORY. every body found by filename convention
  main.task/            readme.md
  produce.task/         readme.md, entry.sh, collect.py
  describe.task/        readme.md
  consume.task/         readme.md, entry.sh, render.py
  check_facts.validator/     readme.md, entry.sh, check.py
  check_grounded.validator/  readme.md, entry.sh, check.py
  lib/                  store.py — shared by the two validator bodies
```

**Nothing in this package binds a filename.** There is no `body:` key anywhere.
A folder named `${name}.${type}` under `assets/` scopes the lookup, and inside
it `readme.md` and `entry.sh` are found by their own names. Binding one by hand
is legal and **warns at compile time**, so a layout that needs bindings is a
layout that failed; this one needs none.

### Why by step and not by kind

`closures/`, `handoffs/`, `agents/`, `validators/` was the old layout, and the
directory name *was* the object's kind — the loader read it. It does not any
more: an object is a validator because it says `module: validator`. Keeping the
directories would leave one fact with two writers that nothing checks against
each other, and a `module: agent` document sitting in `closures/` would be
perfectly legal and perfectly misleading.

A step file groups what changes together. Adding a step is adding one file;
deleting one is deleting one file. It also puts the reference graph mostly
*inside* a file, which is where the one ordering rule lives — a name referenced
before it is defined in the same file is an error — so that rule becomes a
readable "parts first, the step last" convention instead of a puzzle. Under a
per-kind layout every reference is cross-file, where the rule does nothing.

The division is one sentence: **a step file holds what only that step uses; a
handoff kind belongs to the step that produces it, and a validator to the
handoff it judges; anything a second step needs moves up to `shared.yaml`.**

### No inline definitions, and that is the recommendation

The format lets an object be written where it is referenced. This package uses
it nowhere, deliberately: an inline definition is registered under its own name
and hoisted above its host, so it is exactly a top-level definition written
somewhere less greppable. It earns that only when the object is single-use *and*
short enough not to bury the thing it sits inside. Nothing here is both — the
`describe` agent is single-use but is the longest document in the package, and
`collect` is short but has two users, so inlining it is not even legal.

### The broken closure had to leave the package

`broken/` used to be a subdirectory here and is now `../demo-broken/`, a sibling
package. The loader scans **every** `*.yaml` under a root except `assets/`, so a
broken document anywhere inside this directory would load on every ordinary run.
`--dry-run --with-broken` loads the sibling and the run exits 1, naming the file:

```
$ agent-sys run --dry-run                 → exit 0
$ agent-sys run --dry-run --with-broken   → exit 1
  examples/demo-broken/main.yaml::$.task: closure 'dangling' names handoff
  kind 'nonexistent', which does not resolve.
  known kinds: facts, summary
```

One flag, two runs, no editing — which is how criterion 11's broken closure and
criterion 13's *no hand-editing* both hold at once.

## What the specs need from the runner

**One variable**, expanded as `${outside:-<outside: not filled>}` in the
`describe` agent's `env` block:

| | |
|---|---|
| `outside` | Criterion 8's leak target. Per-run and absolute — `Layout.outside` is `<run root>/outside` and a run root is a fresh directory — so no static string can name it |

`package_root` and `store_root` used to be passed beside it. The first made body
paths absolute (F-D3); the assets convention finds them now and they are
package-relative again, which is what `_common.schema.json` always said they
were. The second was referenced by no spec in the package.

Every spec also loads with **no** variables supplied at all, because `show` and
`--dry-run` supply none — which is what the `:-` default is for, and why it is
visibly unfilled rather than empty. `'' + "/leak.txt"` is `/leak.txt`, a
plausible absolute path that would demonstrate nothing.

## Interrupting it

```bash
agent-sys run              # ^C during `produce`
agent-sys run --resume
```

**Interrupt during `produce`, not during `describe`.** A resume re-runs the
interrupted attempt — correctly — so an interrupt during the SDK task pays for a
second model call on restart. `produce` is a program, free to re-run, and the
demonstration is identical: attempt 0 recorded `SUSPENDED`, attempt 1 open.

## Before it runs, it changes one thing on your machine

`env_mgr` cuts each task's workspace with `git clone --shared` and requires
`extensions.preciousObjects` on the repository it clones from, so an ordinary
`git gc` there cannot delete a pack an agent's clone is reading through
`objects/info/alternates`. `prepare()` enforces it.

So **the first thing the demo does is set one git config key on the checkout you
just cloned.** It is reversible and it is genuinely required, but it happens
before anything has been demonstrated, so it is said here rather than done
silently.

## What it does not need

No GPU. No cluster and no remote host — the local↔remote mapping degenerates to
a strong same-machine mapping, which is a legitimate configuration. No long run:
under a minute of wall clock excluding model latency. No hand-editing to run
twice.

It **does** need credentials and a working sandbox. With credentials absent it
fails in about a second with the backend's own message and exit code 2; with
neither `bwrap` nor Landlock available it refuses to start, also with exit code
2, and that refusal is correct behaviour rather than a demo failure.
