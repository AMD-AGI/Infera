# `closure` — the predefined binding of the four objects

```
closure = < handoff spec set, task spec, agent spec, validator set >
```

Three things and nothing more: **a composition of four parts**, **a load
checker**, and **read-only query helpers**. Nothing at runtime — a closure is
consulted when a graph is assembled and never again.

| | |
|---|---|
| Specification | [`docs/spec.md`](docs/spec.md) — 12 acceptance criteria |
| Design | [`docs/design.md`](docs/design.md) |
| Seam | [`../docs/interfaces.md`](../docs/interfaces.md) §4.5 |
| Contract | [`protocols.py`](protocols.py) + `protocols.pyi` |
| Tests | `../tests/closure/` |

## Who uses it

`cli` builds a root `Task` from a closure. `spec_loader` hands it the parsed
documents. The load check runs once, in the composition root, over the whole
registry set.

**This package imports `spec_loader` and nothing else of ours** — which is worth
stating, because its whole job is looking at four other modules' objects and it
would be the easiest place in the system to justify an import. It reaches them
through `Registries` instead.

## The interface

```python
from closure import (
    ClosureDoc, ClosureRegistry,          # a closure, and where closures live
    TaskSpec, TaskSpecRegistry,           # the task spec registry lives here — see below
    check_closures,                       # the load check. Returns problems, raises nothing
    agent_of, declared_handoffs,          # read-only queries over a closure
    named_kinds, permissions_of, phase_validators,
)
```

`check_closures` returns a list of problems rather than raising, so a caller can
report every fault in a package at once instead of the first.

## What is inside

| | |
|---|---|
| `protocols.py` / `.pyi` | the public surface, declarations only |
| `model.py` | the closure document |
| `check.py` | the load checks — the closure is consistent, and its four parts agree |
| `query.py` | the read-only accessors above |
| `registry.py` | the closure registry |
| `task_registry.py` | the **task** spec registry |

## There is no `task/` package, and that is deliberate

A task spec is not independently loadable — the closure declares it, as the
`task` key — so a `task/` package would contain one registry and no other reason
to exist. The four spec registries are still four objects; three have their own
package and one is homed with the document that declares its contents.

**Stated rather than left implicit**, because a reader counting packages will
otherwise count four and find three.

## A gap worth knowing about

A closure's `validators` key is described by the schema as the checks that run in
this task's validation phases. **Nothing runs them** — the phase runner builds
its set from the handoff kind's own `validators`, and this package's accessors
over the closure's list are read by its own code and nothing else. So a package
author can declare a phase validator here, load cleanly, and have it silently
never execute. [`../docs/TODO.md`](../docs/TODO.md) item 10a.
