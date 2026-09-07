"""One stream, N renderers. `demo` design §7.2.

The shape is `logging`'s fan-out without `logging`'s record: a `LogRecord`
carries a formatted string plus arbitrary `extra`, and criterion 14 needs a
closed set of kinds and a versioned payload. Enforcing that on top of `logging`
is more code than `Event` plus this class.

**Imports nothing of ours**, for the reason `events.py` gives.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from cli.events import Event, EventKind

__all__ = ["Renderer", "Stream"]


class Renderer(Protocol):
    def on_event(self, event: Event) -> None: ...


class Stream:
    """Stamp an event and fan it out, in attach order.

    A third renderer — a progress bar, a web view — is one `attach` call and
    changes nothing else. That is the same substitutability argument `agent`
    design §7.1 makes about the runner, and it is why the two shipped renderers
    are subscribers rather than two branches inside one writer.
    """

    def __init__(self) -> None:
        self._renderers: list[Renderer] = []
        self._events: list[Event] = []

    def attach(self, renderer: Renderer) -> None:
        self._renderers.append(renderer)

    def emit(self, kind: EventKind, message: str, /, **fields: Any) -> Event:
        """Stamp, record, and fan out. Returns the event, for a caller that
        wants to assert over what it just said.

        **`kind` and `message` are positional-only, and that is not style.**
        `kind` is also what this system calls a handoff kind, so
        `emit(HANDOFF_TRANSITION, "...", kind="facts")` is an ordinary thing
        to write — and with an ordinary signature it is a `TypeError` about
        multiple values, at the one call site a reviewer is watching. The
        `/` puts every keyword into `fields`, where it was meant to go.
        Found by `test_reserved_keys_cannot_be_overwritten_by_a_field`, which
        was written to check the *renderer* and caught this instead.
        """
        event = Event(
            kind=kind, message=message, fields=dict(fields), at=datetime.now(timezone.utc)
        )
        self._events.append(event)
        for renderer in self._renderers:
            renderer.on_event(event)
        return event

    @property
    def events(self) -> tuple[Event, ...]:
        """Everything emitted, in order.

        A copy, not the live list: handing out the list would be
        `engineer_principle.md` §1's *never hand out a mutable handle to
        internal state*, one indirection later.
        """
        return tuple(self._events)

    def count(self, kind: EventKind) -> int:
        """How many of one kind were emitted.

        A question rather than a field to interrogate: every caller that wanted
        `events` in order to filter and count was going to write the same
        comprehension, and `demo` asserts this over four different kinds.
        """
        return sum(1 for event in self._events if event.kind is kind)

    def of_kind(self, kind: EventKind) -> tuple[Event, ...]:
        return tuple(event for event in self._events if event.kind is kind)
