"""The record — criteria 5, 9, 13, 14, and the routing exhaustiveness check."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from monitor import PLANNED, EventKind, EventRecord, Recorder, event
from monitor.record import EVENT_KIND, SET_KIND, default_fingerprint
from task_graph.ids import HandoffId, TaskId
from task_graph.store import MemoryStoreMgr


@pytest.fixture
def recorder() -> Recorder:
    return Recorder(MemoryStoreMgr())


def test_open_creates_an_empty_set(recorder: Recorder) -> None:
    """Criterion 14: an attempt against which nothing was recorded has an **empty
    record set, not a missing one**, so "never attempted" and "the store lost it"
    are distinguishable. `handoff` creates `validation.yaml` empty at publication
    for the identical reason."""
    tid = TaskId.new()
    assert not recorder.is_open(tid, 0)

    recorder.open(tid, 0)

    assert recorder.is_open(tid, 0)
    assert recorder.read(tid, 0) == []
    recorder.open(tid, 0)  # idempotent
    assert recorder.read(tid, 0) == []


def test_a_record_never_exists_without_its_marker(recorder: Recorder) -> None:
    """The absence signal only works if the container is never skipped."""
    tid = TaskId.new()
    recorder.write(event(EventKind.OUTPUT_ABSENT, tid, attempt=2))
    assert recorder.is_open(tid, 2)


def test_push_attempted_ineffective_never(recorder: Recorder) -> None:
    """Criterion 9: a push that produced no change is distinguishable from a push
    that was never attempted.

    The three states are read off the *sequence*, exactly as `Task.history` is:
    never → no `PUSH_ATTEMPTED` row; ineffective → both rows; worked → the first
    without the second.
    """
    never, ineffective, worked = TaskId.new(), TaskId.new(), TaskId.new()
    recorder.open(never, 0)
    recorder.write(event(EventKind.PUSH_ATTEMPTED, ineffective))
    recorder.write(event(EventKind.PUSH_INEFFECTIVE, ineffective))
    recorder.write(event(EventKind.PUSH_ATTEMPTED, worked))

    kinds = {t: [r.kind for r in recorder.read(t, 0)] for t in (never, ineffective, worked)}

    assert kinds[never] == []
    assert kinds[ineffective] == [EventKind.PUSH_ATTEMPTED, EventKind.PUSH_INEFFECTIVE]
    assert kinds[worked] == [EventKind.PUSH_ATTEMPTED]


def test_records_are_per_attempt(recorder: Recorder) -> None:
    """`(task_id, attempt)` is the `Execution`, so a second attempt's records do
    not answer questions about the first."""
    tid = TaskId.new()
    recorder.write(event(EventKind.PUSH_ATTEMPTED, tid, attempt=0))
    assert [r.kind for r in recorder.read(tid, 1)] == []
    assert len(recorder.read(tid, 0)) == 1


def test_absent_output_kind_does_not_claim_malformed() -> None:
    """Criterion 5, this module's half.

    Spec §4.1.1, measured: `put` raises `Malformed` **before anything is
    created**, inside the producing agent's own zone, so a malformed handoff
    never reaches storage and the gate sees only an absence. There is no phase in
    which this module could observe it — hence no `OUTPUT_MALFORMED` kind, and
    hence `OUTPUT_ABSENT` must not be read as one.

    **The positive half is `agent`'s**: the record that distinguishes "attempted
    and refused" from "never attempted" is written producer-side or not at all.
    """
    assert not hasattr(EventKind, "OUTPUT_MALFORMED")
    assert EventKind.OUTPUT_ABSENT.value == "output_absent"


def test_suite_passes_with_logging_disabled(recorder: Recorder) -> None:
    """Criterion 13: **the record is a persisted value, not a log line.**

    A suite that needs `caplog` is testing the logging configuration (spec §8.1),
    so this one asserts with logging switched off entirely.
    """
    logging.disable(logging.CRITICAL)
    try:
        tid = TaskId.new()
        recorder.write(event(EventKind.VALIDATION_FAILED, tid))
        assert [r.kind for r in recorder.read(tid, 0)] == [EventKind.VALIDATION_FAILED]
    finally:
        logging.disable(logging.NOTSET)


def test_the_store_holds_two_kinds_and_one_file_per_occurrence() -> None:
    """Append-only, one store record per occurrence, never read-modify-write.

    A single container holding a list would put the runner's gate thread and the
    monitor's loop thread in a read/rewrite race that `JsonFileStoreMgr`'s
    per-record atomicity does not cover.
    """
    store = MemoryStoreMgr()
    recorder = Recorder(store)
    tid = TaskId.new()
    for _ in range(3):
        recorder.write(event(EventKind.OUTPUT_ABSENT, tid))

    assert len(store.read_all(EVENT_KIND)) == 3
    assert len(store.read_all(SET_KIND)) == 1


def test_every_kind_is_routed(monitor) -> None:
    """Every `EventKind` member reaches exactly one queue.

    A kind added later cannot fall through both, and one absent from `PLANNED`
    lands on the unplanned queue — the safe direction, because the worst outcome
    is an event that gets decided instead of switched.
    """
    for kind in EventKind:
        before = (len(monitor._planned), len(monitor._buffer))
        # A fresh task per kind: the unplanned buffer collapses on task id, so
        # reusing one would make the second kind look like it reached nothing.
        monitor.report(event(kind, TaskId.new()))
        after = (len(monitor._planned), len(monitor._buffer))

        moved = [a - b for a, b in zip(after, before)]
        assert sum(moved) >= 1, f"{kind.value} reached neither queue"
        assert (moved[0] > 0) is (kind in PLANNED), f"{kind.value} took the wrong queue"

    assert PLANNED == frozenset({EventKind.PHASE_DONE, EventKind.SUBGRAPH_DONE})
    assert PLANNED < set(EventKind)


def test_no_kind_defaults_to_benign() -> None:
    """Spec §8.2.1: `kind` is Erlang's `Context` — a closed enum naming *in which
    phase* the child failed — and **none of its values may default to a benign
    one**. A record cannot be built without saying which."""
    with pytest.raises(ValidationError):
        EventRecord(task_id=TaskId.new())  # type: ignore[call-arg]


def test_fingerprint_groups_across_attempts_not_within() -> None:
    """Design §3.3: excluding `attempt` groups the same failure across attempts
    into one issue; including it would make every fingerprint unique and grouping
    a no-op. Spec §11 leaves it open — this pins the alpha's answer so that
    reversing it is a visible change."""
    tid, hid = TaskId.new(), HandoffId.new()
    first = event(EventKind.OUTPUT_ABSENT, tid, attempt=0, handoff_id=hid)
    second = event(EventKind.OUTPUT_ABSENT, tid, attempt=1, handoff_id=hid)

    assert first.fingerprint == second.fingerprint
    assert first.id != second.id
    assert "0" not in default_fingerprint(first)


def test_a_record_survives_the_store_round_trip(recorder: Recorder) -> None:
    """It is a persisted value, so it has to come back as one."""
    tid = TaskId.new()
    written = event(
        EventKind.BUDGET_EXCEEDED,
        tid,
        attempt=1,
        reported_by="runner",
        severity=17,
        attributes={"max_seconds": 30},
    )
    recorder.write(written)

    (read,) = recorder.read(tid, 1)
    assert read.id == written.id
    assert read.kind is EventKind.BUDGET_EXCEEDED
    assert read.attributes == {"max_seconds": 30}
    assert read.severity == 17
