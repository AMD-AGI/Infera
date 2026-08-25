"""Component registry and the recovery protocol.

Components are registered by name and resolved at use time, never injected
through a constructor. That is what lets a test swap an implementation after
the system is wired, and it is what keeps the import graph acyclic: no manager
imports another manager.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = ["Registry", "Resumable", "RESUME_ORDER", "resume_all"]


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> None:
        """Register, replacing any existing entry.

        Replacement is deliberate: it is the swap mechanism. The protection
        against a typo is `get`'s loud failure, not a registration guard.
        """
        self._items[name] = component

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"no component registered as {name!r}") from None

    def resolve(self, pattern: str) -> list[Any]:
        """`"resource:*"` -> the whole group, in registration order.

        Honours a `prefix:*` suffix and nothing else — no globbing, no regex.
        """
        if pattern.endswith(":*"):
            prefix = pattern[:-1]  # "resource:*" -> "resource:"
            return [c for n, c in self._items.items() if n.startswith(prefix)]
        return [self.get(pattern)]

    def __contains__(self, name: str) -> bool:
        return name in self._items


@runtime_checkable
class Resumable(Protocol):
    def resume_system(self) -> None: ...


# Only "scheduler last" is load-bearing: it recomputes eligibility and needs
# every handoff already reloaded.
RESUME_ORDER = ["handoff_mgr", "agent_mgr", "task_mgr", "resource:*", "scheduler"]


def resume_all(registry: Registry) -> None:
    for pattern in RESUME_ORDER:
        if not pattern.endswith(":*") and pattern not in registry:
            continue
        for component in registry.resolve(pattern):
            # `isinstance` against a runtime-checkable Protocol matches on
            # method *name* alone. That is why the scheduler's per-task resume
            # is `resume_task` — otherwise it would satisfy this by accident.
            if isinstance(component, Resumable):
                component.resume_system()
