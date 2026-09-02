"""The hook seam — the producer cannot read the checking standard.

Spec §8.1: by hook, not by convention. A convention is a prompt instruction, and
an agent complies with prompt instructions literally.

Measured against `claude-agent-sdk` 0.2.144: a single **synchronous**
`PreToolUse` callback logs every attempt before deciding and then denies — so
criterion 10's "spy" and criterion 10's "the hook denies" are the same object,
not two. The async form cannot block (*"async outputs can't block, modify, or
inject context into the operation since the agent has already moved on"*), so
logging-only is not an available optimisation. Composition is safe: any hook
returning `deny` wins.

**The SDK is behind this Protocol because the repository has not committed to
it** — 376 MB installed, 26 extra packages, ~1.3 s to import, and `agent` §8.1
made it an optional extra for the import cost alone. One implementation would
adapt the SDK; `ZoneBoundaryHook` is the one that needs no SDK, and it is what
the tests exercise.

**This layer is attributable, not enforcing.** Measured: `Bash{'command':
'python3 reader.py'}` returns ALLOW, because there is no path in the payload and
therefore nothing to match. Anthropic documents the same for declarative rules —
deny rules *"don't apply to arbitrary subprocesses that read or write files
indirectly"*. `env_mgr`'s allow-list is the layer that makes the standard
unreachable at all, and it says nothing about who tried. Criterion 10 asserts
both halves and this file must not read as though it were both.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "BoundaryHook",
    "Decision",
    "ToolUseEvent",
    "ZoneBoundaryHook",
    "paths_in",
]


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class ToolUseEvent:
    """One attempt, as the hook saw it.

    `agent_id` is **optional**, and that is the SDK's shape rather than ours: the
    union of every field over every `HookInput` type in 0.2.144 contains nothing
    denoting a stack, caller, frame, origin or parent, and the SDK's own
    docstring says `agent_id` is *"present only when the hook fires from inside a
    Task-spawned sub-agent; absent on the main thread."* So a phase running on
    the main thread is unattributable — which is why `environment.py` requires
    each phase to carry one rather than assuming a mechanism `agent` has not
    chosen.
    """

    tool: str
    payload: Mapping[str, Any]
    agent_id: str | None = None
    decision: Decision | None = None
    reason: str = ""


def paths_in(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    """Every filesystem path the payload names outright.

    Deliberately narrow. A `Bash` command's argv is not parsed into paths: doing
    so would turn a shell-quoting bug into an isolation hole, and the measured
    limitation — an indirect read through a subprocess is invisible here — is a
    property to state rather than to almost-close.
    """
    keys = ("file_path", "path", "notebook_path")
    return tuple(Path(payload[k]) for k in keys if isinstance(payload.get(k), str))


@dataclass
class ZoneBoundaryHook:
    """Denies any tool call naming a path inside the checking standard.

    Logs **before** deciding, and logs allowed attempts too. That ordering is
    what makes the log evidence rather than a summary of denials, and it is what
    lets one object answer both halves of criterion 10.
    """

    standards: tuple[Path, ...]
    events: list[ToolUseEvent] = field(default_factory=list)

    def on_tool_use(self, event: ToolUseEvent) -> Decision:
        for candidate in paths_in(event.payload):
            for standard in self.standards:
                if _inside(standard, candidate):
                    self._note(event, Decision.DENY, f"{candidate} is the checking standard")
                    return Decision.DENY
        self._note(event, Decision.ALLOW, "")
        return Decision.ALLOW

    def log(self) -> Sequence[ToolUseEvent]:
        return tuple(self.events)

    def _note(self, event: ToolUseEvent, decision: Decision, reason: str) -> None:
        self.events.append(
            ToolUseEvent(
                tool=event.tool,
                payload=dict(event.payload),
                agent_id=event.agent_id,
                decision=decision,
                reason=reason,
            )
        )


class BoundaryHook(Protocol):
    def on_tool_use(self, event: ToolUseEvent) -> Decision: ...

    def log(self) -> Sequence[ToolUseEvent]: ...


def _inside(root: Path, candidate: Path) -> bool:
    """Fail-closed the *other* way from `separation.reaches`: an unresolvable
    candidate under a standard is denied, because here containment means deny."""
    try:
        target = os.path.realpath(candidate, strict=True)
    except OSError:
        target = os.path.abspath(candidate)
    base = os.path.realpath(root)
    return target == base or target.startswith(base + os.sep)
