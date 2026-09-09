# BUG 001 — `agent_sys` does not import on Python 3.10

Found 2026-09-01 by teammate `env`, confirmed first-hand by the leader.
Recorded per series task book rule 1.1 (`../temp/mission.md`).

**Status:** real bug, not yet fixed upstream. Worked around locally.

## Symptom

Any `agent-sys` invocation — including `--help` — dies at import:

```
File "<...>/agent_sys/cli/main.py", line 62, in <module>
    from validator import PhaseRunner, StrictLevel
[...]
pydantic.errors.PydanticUserError: Please use `typing_extensions.TypedDict`
instead of `typing.TypedDict` on Python < 3.12.
```

## Cause

`agent_sys/spec_loader/protocols.py:15`:

```python
from typing import Any, Protocol, TypeAlias, TypedDict
```

pydantic ≥2 refuses to build a schema from a `typing.TypedDict` on Python
< 3.12, because on those versions `typing.TypedDict` does not record which keys
are `Required`/`NotRequired` in a way pydantic can read. The requirement is
pydantic's and is documented; this is not a pydantic bug.

The tree imports `typing_extensions` in exactly one file — this same one — so
the dependency is already present and declared.

## Environment where it reproduces

| | |
|---|---|
| Image | `lmsysorg/sglang:v0.5.14-rocm720-mi30x` |
| Python | **3.10.12** |
| pydantic | 2.13.4 |
| typing_extensions | 4.15.0 |

It does **not** reproduce on Python ≥ 3.12, which is presumably where
`agent_sys` is developed — `pyproject.toml` should be checked for what it
actually claims to support.

## Workaround in use

A one-line change in a **local copy** at `/tmp/yihou/repo`, not in the
repository:

```python
from typing import Any, Protocol, TypeAlias

from typing_extensions import TypedDict  # LOCAL PATCH: pydantic requires this on py<3.12
```

## Proposed fix

The same change, upstream. It is safe on every supported version:
`typing_extensions.TypedDict` is the recommended import on ≥3.12 as well, and
`typing_extensions` is already a dependency of this module.

**Not applied to the repository by this task**, because the mission scopes work
to `agent_sys/examples/llm_e2e_performance_optimization/` and this is a
`spec_loader` file. Raise it with `agent_sys`'s owner.

## Second, unrelated finding in the same session

`pip install -e agent_sys` **fails when the repository is on NFS and the
container runs as root**:

```
error: could not create 'agent_sys_helper.egg-info': Permission denied
```

`root_squash` maps container root to nobody, so the editable install cannot
write its egg-info beside the source. Two routes work and both were verified:

- `docker exec -u $(id -u):$(id -g)` — run as the host user, who owns the tree;
- install from a copy on local disk (what `env` did, at `/tmp/yihou/repo`).

This is an environment property rather than an `agent_sys` defect, but it will
hit every reviewer on this class of host, so it belongs in the same note.
