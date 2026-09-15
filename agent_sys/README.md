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

Nine packages. Each owns a `README.md` (what it is for), and all but
`spec_loader` own a `docs/spec.md` (the properties it guarantees) and a
`docs/design.md` (how it is built). **The division is deliberate: a spec is about
properties, a design is about the API.** That is why a spec names far less of a
package's exported surface than its design does, and why reading only the spec
will not tell you what to call.

| | |
|---|---|
| [`spec_loader/`](spec_loader/README.md) | The loader, the five JSON Schemas, and the vocabulary every other package shares. The leaf: it imports nothing of ours |
| [`handoff/`](handoff/README.md) | What a unit of transfer carries: content shape, digest, scope tags, validator binding |
| [`validator/`](validator/README.md) | What makes a handoff checkable, and how far a check can be trusted |
| [`task_graph/`](task_graph/README.md) | Decides **which task runs when**, and nothing else |
| [`agent/`](agent/README.md) | What wraps a task spec for execution, and the backend abstraction |
| [`monitor/`](monitor/README.md) | The task's event loop: everything that happens to a task and is not the task's own work |
| [`closure/`](closure/README.md) | The predefined binding of the four objects |
| [`env_mgr/`](env_mgr/README.md) | All interaction with the operating system, including isolation |
| [`cli/`](cli/README.md) | The runnable proof that the above compose. The composition root: it may import anything of ours, and nothing of ours may import it |

`spec_loader` is the one package with no `docs/` of its own; what specifies it is
[`docs/spec.md`](docs/spec.md) §4.3–§4.5 and [`docs/design.md`](docs/design.md)
§3–§5.

**A concrete workflow's specs do not live here.** This repository holds the JSON
Schemas, the loader, and the workflow-independent general specs; a workflow's own
handoff kinds, validators, tasks, agents, and closures live in a **task package**
outside it. `examples/` is the one exception, and it is a directory rather than a
list — an example there is a task package like any other, uses nothing an outside
package could not, and is never a dependency of this repository's suite. See
[`docs/spec.md`](docs/spec.md) §4.3.

The whole-system specification's index into every component's acceptance criteria
is [`docs/spec.md`](docs/spec.md) §9.

Planned work lives in two places: [`docs/ROADMAP.md`](docs/ROADMAP.md) for
long-term subsystems — observability, the monitor agent system, human-in-the-loop
— and [`docs/TODO.md`](docs/TODO.md) for near-term decisions and pieces.

## Layout

```
agent_sys/
├── pyproject.toml       declares the packages; ruff and pytest settings
├── docs/
│   ├── spec.md          the whole-system specification — start here
│   ├── design.md        how the system is built; §3–§5 specify spec_loader
│   ├── interfaces.md    normative for what crosses a module boundary
│   ├── ROADMAP.md       long-term subsystems
│   └── TODO.md          near-term decisions and work
├── spec_loader/         *.py, and schemas/ — the five JSON Schemas
├── handoff/             README.md, docs/{spec,design}.md, *.py
├── validator/           likewise, plus general_specs/ — the workflow-independent
│                        specs, loaded by the ordinary path (docs/spec.md §4.5)
├── agent/               likewise, plus backends/
├── monitor/             likewise
├── closure/             likewise
├── task_graph/          likewise
├── env_mgr/             likewise, plus installers/ isolation/ fs/ remote/ o11y/
├── cli/                 likewise, plus render/
├── examples/            task packages. YAML and data — not installed, and
│                        imported by nobody (docs/spec.md §4.3)
└── tests/               one directory per package, plus interfaces/

<anywhere else>/
└── <a task package>/    one workflow's specs. Not in this repository (§4.3)
```

The schemas live under `spec_loader/` rather than at the top level, and that is
forced rather than chosen — [`docs/design.md`](docs/design.md) §2.2 has the
measurement.

## Running what exists

```bash
pip install -e agent_sys              # once, from the repository root
pytest agent_sys                      # every package
pytest agent_sys/tests/task_graph     # one of them
pytest agent_sys/tests/interfaces     # the seams between them
```

No count is quoted here on purpose: a number in a README is a second copy of
something the command already reports, and it is the copy that goes stale.

The repository's own `pyproject.toml` is untouched: its
`[tool.setuptools.packages.find] include = ["infera*"]` does not cover
`agent_sys`, and does not need to.
