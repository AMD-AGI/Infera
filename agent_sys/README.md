# agent_sys

The agent work system. A multi-agent system that records and replays a fixed
task flow, on one claim:

> **A task is a function.** Its signature is `<handoffs, agent>`. Quality is
> guaranteed by standardising the inputs and outputs, not by trusting the
> executor.

**Start with [`docs/spec.md`](docs/spec.md)** — the whole-system specification.
It is the only document a reader must finish; the rest are read on demand.

## Components

| | | |
|---|---|---|
| [`handoff/`](handoff/docs/spec.md) | What a unit of transfer carries: schema, digest, scope tags, validator binding | Specified |
| [`validator/`](validator/docs/spec.md) | What makes a handoff checkable, and how far a check can be trusted | Specified |
| [`task_graph/`](task_graph/README.md) | Decides **which task runs when**, and nothing else | **Implemented** (rev. 7); subgraph nesting specified at rev. 8 |
| [`agent/`](agent/docs/spec.md) | What an executor declares, and the backend abstraction | Specified |
| [`closure/`](closure/docs/spec.md) | The predefined binding of the four objects | Specified |
| [`env_mgr/`](env_mgr/README.md) | All interaction with the operating system | **Implemented** (environment recipes); widened scope specified |
| [`demo/`](demo/docs/spec.md) | The runnable proof that the above compose | Specified |

Each component owns its own `docs/spec.md`. `env_mgr` and `task_graph` also have
a `README.md` and, for `task_graph`, a design document.

## Layout

```
agent_sys/
├── pyproject.toml       declares the packages; ruff and pytest settings
├── docs/spec.md         the whole-system specification — start here
├── handoff/docs/
├── validator/docs/
├── agent/docs/
├── closure/docs/
├── demo/docs/
├── env_mgr/             README.md, docs/spec.md, *.py
├── task_graph/          README.md, docs/{spec,design}.md, *.py
└── tests/
    ├── env_mgr/
    └── task_graph/
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
