# agent_sys

A system for running **multi-agent work whose quality is checked rather than
trusted**. You declare a workflow as data; the system loads it, decides what runs
when, gives each task an isolated environment, and refuses to let an unchecked
artefact reach the next task.

> **A task is a function.** Its signature is `<handoffs, agent>`. Quality comes
> from standardising the inputs and outputs, not from trusting the executor.

**The workflow is not code.** A *task package* is a directory of YAML plus the
files those documents name; nothing here changes to add, change or retire one.

## How it works

```
 DECLARED   ┌ a task package ─────────────────┐   ┌ this repository ───────────┐
            │  handoff · validator · task ·   │   │  5 JSON Schemas            │
            │  agent · closure — YAML         │   │  general specs             │
            │  documents, one workflow's own  │   │  the loader                │
            └────────────────┬────────────────┘   └─────────────┬──────────────┘
                             │                                  │
 LOADED     ┌────────────────▼──────────────────────────────────▼──────────────┐
            │  scan & discriminate ──► validate (JSON Schema) ──► admit         │
            │  the source is never seen; the package delivers parsed documents  │
            └────────────────────────────────┬─────────────────────────────────┘
                                             ▼
            ┌ four independent registries ───────────────────┐  ┌ closure ─────┐
            │  handoff · validator · task · agent            │◄─┤ the binding  │
            │  name → spec                                   │  │ of the four  │
            └────────────────────────────────┬───────────────┘  └──────────────┘
                                             │
 SCHEDULED  ┌────────────────────────────────▼─────────────────────────────────┐
            │  task_graph — decides WHEN a task runs. Never WHAT it does        │
            └────────────────┬──────────────────────────────┬──────────────────┘
                             │ dispatches ONE task          │ read-only: is this
                             ▼                              │ input's latest
 EXECUTED   ┌ TaskRunner ─────────────────┐                 │ version VALID?
            │  1. input validation  ──────┼─────────────────┘
            │  2. main ──► agent, or a    │   the two validation phases are
            │     subgraph of tasks       │   invisible to the scheduler, and
            │  3. output validation ──────┼──► the verdict is written in 3
            └──────┬───────────────┬──────┘
                   ▼               ▼
            ┌ handoff storage ┐  ┌ env_mgr ────────────────────────────────────┐
            │ versioned slots │  │ workspace · playground · storage · isolation │
            └─────────────────┘  └──────────────────────────────────────────────┘
```

Full version, with the two authority boundaries marked: [`docs/spec.md`](docs/spec.md) §2.1.

## Run one

`examples/ok.agent_capabilities.2` asks whether the environment an agent is
promised actually arrives. `show` loads and type-checks a whole package and
dispatches nothing — the cheapest way to see a graph:

```bash
pip install -e agent_sys        # once, from the repository root

agent-sys show \
  --package agent_sys/examples/ok.agent_capabilities.2 \
  --var nonce=$(python3 -c 'import secrets;print(secrets.token_hex(16))') \
  --var uv_root=/tmp/$USER/agentsys_uv
```

```
   package  loaded 1 task package(s) from …/ok.agent_capabilities.2
   closure  probe_env: agent 'env_probe', 0 in, 1 out
     graph  2 tasks: 1 root and 1 subtasks
      done  2 tasks in the graph; nothing was dispatched
```

`agent-sys run` executes it, and needs credentials and a working sandbox.
`--var` supplies a package variable; a `${name}` with no default and no value is
a **load error naming the file and the variable** — which is the point of `show`.

## What a task package looks like

```
ok.agent_capabilities.2/
├── main.yaml      MANDATORY, and the name is fixed. Its presence is what makes
│                  this directory a package, and its absence makes it a library
├── steps/         one file per step. A file may declare SEVERAL objects —
│   └── check.yaml   1 agent · 1 handoff kind · 1 task · 2 validators
└── assets/        MANDATORY. Every body is found here BY CONVENTION
    ├── main.task/readme.md                    a non-leaf: readme, no entry.sh
    ├── probe_env.task/readme.md               an AI task: readme only
    ├── env_probe.agent/…                      an agent's own .claude/ tree
    ├── check_env_report_shape.validator/      readme.md · entry.sh · check.py
    └── lib/                                   shared by several bodies
```

Four rules, and they are the whole format:

| | |
|---|---|
| **`module:` decides what a document is** | Not the directory, not the filename. One file may hold a task, its handoff kind and its validators |
| **Bodies are found by convention** | `<name>.<type>/` scopes the lookup; `readme.md` and `entry.sh` are found by their own names. **There is no `body:` key** |
| **A leaf has an `entry.sh` or an agent; a non-leaf has neither** | Its work is its subgraph, and the two are mutually exclusive |
| **Only two names are reserved** | `main.yaml` and `assets/` |

## The life of a package

```
 AUTHORED   YAML for its handoff kinds, validators, tasks, agents, closures
                │  scan ─► discriminate ─► validate against the schema
 ADMITTED   four registries + the closure check — the only pass that sees all
            four at once. Any failure here and NOTHING runs: a LOAD error
                │  a graph is assembled; after this no closure is read again
 RUNNING    ┌ run ─┐  out v1     ┌ run ─┐  out v1     ┌ run ─┐
            │ task ├── VALID ───►│ task ├── INVALID ─╫│ task │  never runs: its
            │  A   │             │  B   │            ╫│  C   │  input never
            └──────┘             └──────┘             └──────┘  became VALID
 LEFT       versioned handoffs with their verdicts · one execution record per
  BEHIND    run · agent bindings — what "reproducible" is claimed against
```

**An `INVALID` output is not a task failure.** Task B succeeded; its output
simply never became eligible input for C. [`docs/spec.md`](docs/spec.md) §2.4.

## The nine packages

Each owns a `README.md`; all but `spec_loader` own a `docs/spec.md` (the
properties it guarantees) and a `docs/design.md` (how it is built).

| | |
|---|---|
| [`spec_loader/`](spec_loader/README.md) | The loader, the five JSON Schemas, and the shared vocabulary. The leaf: it imports nothing of ours. Specified by [`docs/spec.md`](docs/spec.md) §4.3–§4.5 and [`docs/design.md`](docs/design.md) §3–§5 |
| [`handoff/`](handoff/README.md) | What a unit of transfer carries: content shape, digest, scope tags, validator binding |
| [`validator/`](validator/README.md) | What makes a handoff checkable, and how far a check can be trusted |
| [`task_graph/`](task_graph/README.md) | Decides **which task runs when**, and nothing else |
| [`agent/`](agent/README.md) | What wraps a task spec for execution, and the backend abstraction |
| [`monitor/`](monitor/README.md) | The task's event loop: everything that happens to a task and is not its own work |
| [`closure/`](closure/README.md) | The predefined binding of the four objects |
| [`env_mgr/`](env_mgr/README.md) | All interaction with the operating system, including isolation |
| [`cli/`](cli/README.md) | The composition root. It may import anything of ours; nothing of ours may import it |

## The documents

[`docs/spec.md`](docs/spec.md) is the one a reader must finish; its §1.4 says
which of the others to open when. [`docs/design.md`](docs/design.md) is how the
system is assembled, [`docs/interfaces.md`](docs/interfaces.md) is normative for
what crosses a module boundary, [`engineer_principle.md`](engineer_principle.md)
is binding for anyone writing here, and planned work is
[`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/TODO.md`](docs/TODO.md).

```bash
pytest agent_sys                   # every package, and the seams between them
```
