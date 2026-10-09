"""The event and its persistence — design §3.

Spec §8: a record is **a persisted value written through `task_graph`'s
`StoreMgr`, never a log line**. A test that satisfies criterion 9 with `caplog`
is testing the logging configuration; logging is a projection of this, rendered
at the severity the record already carries.

`EventKind` and `PLANNED` are *not* redeclared here. They are the seam and live
in `protocols.py`; a second declaration of a closed enum is the two-writers
failure `engineer_principle.md` §1 names, and the routing rule has to be the same
object the reporters see.

This module may not import `buffer.py`: the record survives the buffer (spec
§5.2 rule 3), and the import graph is what makes that structural.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field, model_validator

from task_graph.ids import AgentId, HandoffId, Id, TaskId
from task_graph.models import Model
from task_graph.store import StoreMgr

from .protocols import EventKind

__all__ = [
    "SET_KIND",
    "EVENT_KIND",
    "EventId",
    "EventRecord",
    "Recorder",
    "default_fingerprint",
    "event",
    "rekeyed",
]

#: The two store kinds. Two, because **absence is a signal** (spec §8.3): the
#: marker present with no occurrences reads as "nothing was recorded here", and
#: the marker absent reads as "something is wrong". `handoff` applies the
#: identical rule to `validation.yaml`, which is created empty at publication.
SET_KIND = "event_set"
EVENT_KIND = "event"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EventId(Id):
    """A fourth typed identity, declared here rather than in `task_graph`.

    A fourth id in `task_graph/ids.py` would make that package carry a monitor
    concept, which is the dependency `Pushable` exists to avoid. So the subclass
    lives here and the base is imported.

    **The base is `Id`, and it is public.** This read `_Id` and design §13
    recorded the private-name import as *"the smaller of two costs"* — a
    deviation, not a clean choice. `task_graph` has since made the name public,
    declining the ruling to merely export `_Id` on the ground that a public name
    spelled with a leading underscore contradicts `interfaces.md` §1.2, where the
    underscore means *named in one package*. `_Id = Id` remains as an alias with
    a retirement date; **this no longer uses it**, so the deviation is closed
    rather than carried.
    """

    __slots__ = ()


class EventRecord(Model):
    """One occurrence. Satisfies `protocols.EventRecord` structurally.

    The field vocabulary is the three-source hybrid of spec §8.2, at zero
    dependency cost: OpenTelemetry's stable `exception.*` names and
    `SeverityNumber` supply the naming, Erlang's SASL supervisor record the
    structure (`kind` is its `Context`, `reported_by` its `Supervisor`, the
    correlation fields its `Offender`), and Sentry's event/fingerprint split the
    identity.
    """

    # -- identity --
    id: EventId = Field(default_factory=EventId.new)
    #: A tuple of strings and not a hash: `JsonFileStoreMgr` writes one readable
    #: JSON file per record, and being able to `cat` one while the schema is
    #: still moving is the whole reason files were chosen over sqlite.
    fingerprint: tuple[str, ...] = ()

    # -- correlation. No standard supplies this; it is ours --
    task_id: TaskId
    attempt: int = 0  # (task_id, attempt) IS the Execution
    agent_id: AgentId | None = None
    handoff_id: HandoffId | None = None  # which declared output, when one

    # -- classification --
    kind: EventKind
    reported_by: str = ""

    # -- payload (OTel `exception.*`, as naming only) --
    exception_type: str | None = None
    exception_message: str | None = None
    exception_stacktrace: str | None = None  # only when one was raised

    # -- severity (OTel SeverityNumber; the number, not the text) --
    severity: int = 13  # 13 WARN handled, 17 ERROR gave up

    at: datetime = Field(default_factory=_now)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_fingerprint(self) -> EventRecord:
        if not self.fingerprint:
            # Written through `__dict__` rather than assigned: `Model` sets
            # `validate_assignment=True`, so an assignment here re-enters this
            # validator. The field exists and is already the right type.
            self.__dict__["fingerprint"] = default_fingerprint(self)
        return self


def default_fingerprint(record: EventRecord) -> tuple[str, ...]:
    """What groups two occurrences into one issue.

    **`attempt` is excluded, and it is a choice rather than a detail.** Excluding
    it groups the same failure across attempts into one issue; including it makes
    every fingerprint unique and grouping a no-op. Spec §11 leaves it open — this
    is the alpha's answer, in one function so that reversing it is one edit.
    """
    return (
        record.kind.value,
        str(record.task_id),
        str(record.handoff_id or ""),
        record.exception_type or "",
    )


def event(kind: EventKind, task_id: TaskId, **fields: Any) -> EventRecord:
    """Build a record. A convenience for reporters, and nothing more."""
    return EventRecord(kind=kind, task_id=task_id, **fields)


def rekeyed(record: EventRecord, task_id: TaskId, kind: EventKind | None = None) -> EventRecord:
    """The same occurrence, re-addressed to another task.

    Escalation (§7.1) and the non-leaf re-entry (§7.4) both hand a record to
    *another task's* monitor, and that monitor runs it through `_transition`,
    which refuses a task that is not the one it is handling. A record keeping the
    child's `task_id` would therefore be a scope violation on arrival.

    The child's id survives in `attributes["from_task"]`, which is also what tells
    the receiving monitor that the hop already happened.
    """
    data = record.model_dump()
    data["id"] = EventId.new()
    data["task_id"] = task_id
    data["fingerprint"] = ()  # recomputed for the task it now names
    data["attributes"] = {**record.attributes, "from_task": str(record.task_id)}
    if kind is not None:
        data["kind"] = kind
    return EventRecord(**data)


class Recorder:
    """Writes records through `StoreMgr`. Owns no policy.

    **Append-only, one store record per occurrence — never read-modify-write.**
    The alternative, a single container record holding a list, would have the
    runner's gate thread and the monitor's loop thread reading and rewriting one
    JSON file. `JsonFileStoreMgr` is atomic *per record* (`tmp.replace(path)`)
    and not across a read, so that shape needs a lock the append-only shape does
    not. Choosing a key that is unique per occurrence removes the race instead of
    guarding it.
    """

    def __init__(self, store: StoreMgr) -> None:
        self._store = store

    @staticmethod
    def _set_key(task_id: TaskId, attempt: int) -> str:
        return f"{task_id}#{attempt}"

    def open(self, task_id: TaskId, attempt: int) -> None:
        """Create the (empty) record set for one attempt. Idempotent."""
        key = self._set_key(task_id, attempt)
        if self._store.exists(SET_KIND, key):
            return
        self._store.create(
            SET_KIND,
            key,
            {"task_id": str(task_id), "attempt": attempt, "opened_at": _now().isoformat()},
        )

    def write(self, record: EventRecord) -> None:
        """Persist one occurrence.

        Opens the set first, so a record can never exist without the marker that
        makes its absence readable (criterion 14).
        """
        self.open(record.task_id, record.attempt)
        key = f"{self._set_key(record.task_id, record.attempt)}#{record.id}"
        self._store.create(EVENT_KIND, key, record.model_dump(mode="json"))

    def read(self, task_id: TaskId, attempt: int) -> list[EventRecord]:
        """Every record for that attempt, in written order. `[]` if none.

        Ordering is `at`, sorted stably, so records sharing a timestamp keep the
        order the store returned them in. **It is a scan of the kind** — design
        O3: fine at alpha scale, on the decision path of every push, and the
        first thing that will hurt.
        """
        prefix = self._set_key(task_id, attempt) + "#"
        rows = [
            EventRecord.model_validate(r)
            for r in self._store.read_all(EVENT_KIND)
            if f"{r.get('task_id')}#{r.get('attempt')}#" == prefix
        ]
        return sorted(rows, key=lambda r: r.at)

    def is_open(self, task_id: TaskId, attempt: int) -> bool:
        return self._store.exists(SET_KIND, self._set_key(task_id, attempt))
