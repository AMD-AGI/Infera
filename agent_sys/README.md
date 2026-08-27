# agent_sys

The agent work system. Two independent components, each a top-level package
declared by `pyproject.toml` here.

| | |
|---|---|
| [`env_mgr/`](env_mgr/README.md) | Layered environment manager. One YAML recipe drives check / dry-run / install / bootstrap |
| [`task_graph/`](task_graph/README.md) | Task-management substrate: decides **which task runs when**, and nothing else |

They share this manifest and a test tree, and nothing else — neither imports
the other.

## Layout

```
agent_sys/
├── pyproject.toml       declares both packages; ruff and pytest settings
├── env_mgr/
│   └── README.md
├── task_graph/
│   ├── README.md
│   └── docs/            spec.md, design.md
└── tests/
    ├── env_mgr/
    └── task_graph/
```

Each component's own README is the entry point for it; this file only says
which is which.

## Running

```bash
pip install -e agent_sys              # once, from the repository root
pytest agent_sys                      # both components
pytest agent_sys/tests/task_graph     # one of them
```

The repository's own `pyproject.toml` is untouched: its
`[tool.setuptools.packages.find] include = ["infera*"]` does not cover
`agent_sys`, and does not need to.
