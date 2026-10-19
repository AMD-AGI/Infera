"""closure — the predefined binding of a task's handoffs, its agent, and its
validators.

Three things and nothing more: a composition of four parts, a load checker, and
read-only query helpers. **Nothing at runtime** — a closure is consulted when a
graph is assembled and never again.

The public surface is `docs/interfaces.md` §4.5's, and it is exactly what
`protocols.py` declares. Everything else in this package is internal.
"""

from .check import check_closures
from .model import (
    ClosureDoc,
    TaskSpec,
    agent_of,
    declared_handoffs,
    named_kinds,
    permissions_of,
    phase_validators,
)
from .registry import ClosureRegistry
from .task_registry import TaskSpecRegistry

__all__ = [
    "ClosureDoc",
    "ClosureRegistry",
    "TaskSpec",
    "TaskSpecRegistry",
    "agent_of",
    "check_closures",
    "declared_handoffs",
    "named_kinds",
    "permissions_of",
    "phase_validators",
]
