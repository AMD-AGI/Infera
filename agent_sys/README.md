# agent_sys

The agent work system. A multi-agent system that records and replays a fixed
task flow, on one claim:

> **A task is a function.** Its signature is `<handoffs, agent>`. Quality is
> guaranteed by standardising the inputs and outputs, not by trusting the
> executor.

**Start with [`docs/spec.md`](docs/spec.md)** — the whole-system specification.
It is the only document a reader must finish; the rest are read on demand, and
its §1.4 says which one to open when. Its §2 is the architecture: the flow on one
page (§2.1), the structure (§2.3), and the life of one task (§2.4).

## Components

| | | |
|---|---|---|
| [`handoff/`](handoff/docs/spec.md) | What a unit of transfer carries: content shape, digest, scope tags, validator binding | Specified |
| [`validator/`](validator/docs/spec.md) | What makes a handoff checkable, and how far a check can be trusted | Specified |
| [`task_graph/`](task_graph/README.md) | Decides **which task runs when**, and nothing else | **Implemented** (rev. 7); subgraphs, validation phases, task-owned transitions, and leaf-only resource acquisition specified at rev. 8–12 |
| [`agent/`](agent/docs/spec.md) | What wraps a task spec for execution, and the backend abstraction | Specified |
| [`closure/`](closure/docs/spec.md) | The predefined binding of the four objects | Specified |
| [`env_mgr/`](env_mgr/README.md) | All interaction with the operating system, including isolation | **Implemented** (environment recipes); widened scope specified |
| [`cli/`](cli/docs/spec.md) | The runnable proof that the above compose. The first **task package**, and the only one in this repository | Specified |

**A concrete workflow's specs do not live here.** This repository holds the JSON
Schemas, the loader, and the workflow-independent general specs; a workflow's own
handoff kinds, validators, tasks, agents, and closures live in a **task package**
outside it. See [`docs/spec.md`](docs/spec.md) §4.3.

The whole-system specification's index into every component's acceptance criteria
is [`docs/spec.md`](docs/spec.md) §9.

Planned work lives in two places: [`docs/ROADMAP.md`](docs/ROADMAP.md) for
long-term subsystems — observability, the monitor agent system, human-in-the-loop
— and [`docs/TODO.md`](docs/TODO.md) for near-term decisions and pieces.

Each component owns its own `docs/spec.md`. `env_mgr` and `task_graph` also have
a `README.md` and, for `task_graph`, a design document.

## Layout

```
agent_sys/
├── pyproject.toml       declares the packages; ruff and pytest settings
├── docs/
│   ├── spec.md          the whole-system specification — start here
│   ├── ROADMAP.md       long-term subsystems
│   └── TODO.md          near-term decisions and work
├── handoff/docs/
├── validator/docs/
├── agent/docs/
├── closure/docs/
├── cli/docs/
├── env_mgr/             README.md, docs/spec.md, *.py
├── task_graph/          README.md, docs/{spec,design}.md, *.py
└── tests/
    ├── env_mgr/
    └── task_graph/
```

Arriving with the loader, at the implementation stage:

```
agent_sys/
├── schemas/             the spec of the spec — one JSON Schema per object
├── general_specs/       workflow-independent specs. Templates with an empty
│                        `config`; loaded by the ordinary path (docs/spec.md §4.5)
└── examples/demo/       the demo task package — jsonnet specs, like any package's

<anywhere else>/
└── <a task package>/    one workflow's specs. Not in this repository (§4.3)
```

## Running what exists

```bash
pip install -e agent_sys              # once, from the repository root
pytest agent_sys                      # both implemented components — 423 tests
pytest agent_sys/tests/task_graph     # one of them
```

The repository's own `pyproject.toml` is untouched: its
`[tool.setuptools.packages.find] include = ["infera*"]` does not cover
`agent_sys`, and does not need to.

## Delivery

Three stages, each reviewed before the next begins: **spec → design → test &
code**. `task_graph` and `env_mgr` have completed all three. Everything else is
at stage one.
