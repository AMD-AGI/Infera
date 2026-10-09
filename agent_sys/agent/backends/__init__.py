"""Backend resolution — design §6.3.

`backend_entry` takes both forms, because the surveyed projects that use both do
so for different jobs and neither substitutes for the other:

| Form | Example | For |
|---|---|---|
| Dotted path | `agent.backends.program:ProgramExecutor` | naming *this exact one* |
| Entry point | `claude_sdk` in group `agent_sys.backends` | discovering *what exists* |

**Resolution is not availability.** Resolution answers "does this name denote
something"; availability answers "can it run here", and the second is a runtime
question whose answer changes after `env_mgr` deploys. Conflating them would
take the availability reading at the one moment it is guaranteed to be wrong —
SQLAlchemy draws the same line, and `get_dialect()` resolves with the driver
absent (design §4, check 2).

**Nothing here imports a concrete backend at module scope**, which is what keeps
`backends/claude_sdk.py`'s 376 MB extra out of every entry point (§8.1).
"""

from __future__ import annotations

import importlib
from importlib import metadata

from agent.backend import BackendUnsupported

__all__ = ["BUILTIN", "GROUP", "resolve"]

#: The entry-point group a third party adds a backend through.
GROUP = "agent_sys.backends"

#: The two this package ships, so a bare key resolves without an installed
#: entry point. A dotted path always wins over both.
BUILTIN: dict[str, str] = {
    "program": "agent.backends.program:ProgramExecutor",
    "claude_sdk": "agent.backends.claude_sdk:ClaudeSdkBackend",
}


def resolve(entry: str, *, key: str = "", err: str = "") -> type:
    """Resolve a `backend_entry` to the class behind it.

    Raises `BackendUnsupported` naming the entry. `err` is the declaration's own
    message and replaces the generic one when present — the one thing a dotted
    string buys that an entry point cannot, and it is worth the whole cost.
    """
    name = key or entry
    target = BUILTIN.get(entry, entry) if ":" not in entry else entry
    if ":" in target:
        return _dotted(target, name, err)
    return _entry_point(target, name, err)


def _dotted(target: str, key: str, err: str) -> type:
    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise BackendUnsupported(key, "be resolved", err or f"{target}: {exc}") from exc
    try:
        found = getattr(module, attribute)
    except AttributeError as exc:
        raise BackendUnsupported(
            key, "be resolved", err or f"{module_name} has no {attribute!r}"
        ) from exc
    if not isinstance(found, type):
        raise BackendUnsupported(key, "be resolved", err or f"{target} is not a class")
    return found


def _entry_point(target: str, key: str, err: str) -> type:
    found = [point for point in metadata.entry_points(group=GROUP) if point.name == target]
    if not found:
        available = sorted({point.name for point in metadata.entry_points(group=GROUP)})
        available += sorted(BUILTIN)
        raise BackendUnsupported(
            key,
            "be resolved",
            err or f"no backend {target!r}; known: {', '.join(sorted(set(available))) or 'none'}",
        )
    try:
        loaded = found[0].load()
    except ImportError as exc:
        raise BackendUnsupported(key, "be resolved", err or f"{target}: {exc}") from exc
    if not isinstance(loaded, type):
        raise BackendUnsupported(key, "be resolved", err or f"{target} is not a class")
    return loaded
