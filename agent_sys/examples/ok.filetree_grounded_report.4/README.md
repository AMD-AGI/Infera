# `ok.filetree_grounded_report.4` — survey a file tree, report it, stay grounded

The runnable proof that the components compose: four tasks, one program and one
AI, a check that passes and a check that fails for a structural reason.

**This directory is data.** YAML specs and the programs they name. Nothing in
`agent_sys` imports any of it — there is no `__init__.py` here and there never
will be. It is the only *working* task package in this repository (its sibling
`../fail.dangling_handoff_kind.1/` is structurally one and exists so criterion 11
has a closure that fails to load), and the exception is deliberate: the demo is
the system's falsifiable claim and the first thing a reviewer runs.

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

The failure is **structural**. The `summary` kind's contract is *every number in
the summary must also appear in the `facts` it summarises*
(`trustworthiness / strong`); `describe`'s goal asks for a summary **including
how long collection took**; and `facts` carries no duration. So the agent is
asked, in good faith, for a figure its input cannot ground.

No bad model, no prompt trick, no validator that secretly returns `False` — only
**a task specified with a gap in it**, which is the failure this system exists
to catch. If the model answers *"the facts do not record a duration"* the verdict
passes and **the demo fails loudly with exit 3**: an expected failure that
passes is a failure.

## The three bodies, and the one file that separates them

Every task has a `readme.md`; a programmatic one adds an `entry.sh`.

| Task | Body | What you see |
|---|---|---|
| `produce` | `readme.md` + `entry.sh` | the exact command that ran, reproducible by hand |
| `describe` | `readme.md` **only** | the instruction the agent worked from |
| `consume` | `readme.md` + `entry.sh` | never runs; its readme still says what it would have done |
| `main` | `readme.md`, no `entry.sh` | a non-leaf. Its work is its subgraph, and the two are mutually exclusive |

**That `describe` and `produce` differ by one file is the point** — comparing the
two folders is the whole of what "agent task" versus "program task" means here.

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

### Three layout choices worth copying

**By step, not by kind** — a step file groups what changes together and keeps
the reference graph inside one file, where the single ordering rule lives.
**No inline definitions** — an inline object is hoisted and registered under its
own name anyway, so it is a top-level definition somewhere less greppable.
**The broken closure is a sibling package** — the loader scans every `*.yaml`
under a root except `assets/`, so a broken document here would load on every
ordinary run.

## What the specs need from the runner

**One variable.** `outside` is criterion 8's leak target — per-run and absolute,
so no static string can name it. Every spec also loads with **no** variables at
all, because `show` and `--dry-run` supply none; that is what the `:-` default is
for, and why it renders visibly unfilled rather than empty. An empty string would
make the leak target `/leak.txt`, a plausible path that would demonstrate
nothing.

## Interrupting it

```bash
agent-sys run              # ^C during `produce`
agent-sys run --resume
```

**Interrupt during `produce`, not `describe`.** A resume re-runs the interrupted
attempt, so interrupting the SDK task pays for a second model call; `produce` is
a program and free to re-run. The demonstration is identical.

## What it changes, needs, and does not need

**It sets one git config key on your checkout before anything is demonstrated.**
`env_mgr` cuts each task's workspace with `git clone --shared` and requires
`extensions.preciousObjects` on the source repository, so an ordinary `git gc`
cannot delete a pack an agent's clone reads through `objects/info/alternates`.
Reversible, genuinely required, and said here rather than done silently.

No GPU, no cluster, no remote host, no hand-editing to run twice, under a minute
of wall clock excluding model latency.

It **does** need credentials and a working sandbox. Without credentials it fails
in about a second with the backend's own message and exit 2; with neither `bwrap`
nor Landlock it refuses to start, also exit 2 — and that refusal is correct
behaviour rather than a demo failure.
