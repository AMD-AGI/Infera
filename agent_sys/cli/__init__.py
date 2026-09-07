"""`cli` — the program's command-line entry point.

It began as the demo's runner, and the demo is still what it is measured
against. Two artefacts, and the split is what makes `demo` spec §1.1 checkable:

| | |
|---|---|
| `examples/demo/` | the **task package** — YAML specs and the programs they name. Not installed, not importable, on nobody's `sys.path` |
| `cli/` | the **runner** — `run`, `show`, `--dry-run`. An ordinary top-level package, and where `[project.scripts]` points |

`cli` imports all eight components. **Nothing imports `cli`**, and
`tests/cli/test_package_loads.py::test_no_component_imports_demo` greps every
component package for the token to prove it.

Nothing is exported from here. There is no public API: the entry point is
`cli.main:main`, and a component that imported anything from this package would
be criterion 15's violation whatever the name was.
"""
