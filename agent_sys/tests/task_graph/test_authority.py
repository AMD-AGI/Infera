"""The authority boundary — criterion 14.

The scheduler decides *when*, never *what*. It never writes handoff state. This
is the one mechanical check that the boundary has not eroded: a spy HandoffMgr
logs every call, `produce` brackets its own writes with a marker, and the
assertion is that every write falls inside a pair of markers.

"Who called this" is recorded rather than inferred from the call stack, which
would be unimplementable in any honest way.
"""

from task_graph.bootstrap import build_registry
from task_graph.handoff import HandoffMgr
from task_graph.models import TaskStatus
from task_graph.runner import FakeRunner
from task_graph.store import MemoryStoreMgr

from .conftest import make_task, new_handoffs

WRITES = {"persist"}
# `get` and `latest` hand back a *live, mutable* Handoff, so they are only reads
# by convention — a drifting scheduler would reach state through exactly these.
# Nothing in `scheduler.py` calls `handoff_mgr.get`, which
# `test_the_scheduler_never_takes_a_mutable_handle` pins down; `latest` it does
# call, and reads one integer off. See design O12 for what the spy cannot see.
HANDLE_RETURNING = {"get", "latest"}
READS = {"declare", "check_if_latest_valid", "get_many", "all_ids"} | HANDLE_RETURNING


class SpyHandoffMgr(HandoffMgr):
    def __init__(self, registry) -> None:
        super().__init__(registry)
        self.log: list[str] = []

    def declare(self, ids, producer_task_id, types=None):
        self.log.append("declare")
        return super().declare(ids, producer_task_id, types)

    def check_if_latest_valid(self, hid):
        self.log.append("check_if_latest_valid")
        return super().check_if_latest_valid(hid)

    def latest(self, hid):
        self.log.append("latest")
        return super().latest(hid)

    def get(self, hid):
        self.log.append("get")
        return super().get(hid)

    def persist(self, hid):
        self.log.append("persist")
        return super().persist(hid)


class MarkingRunner(FakeRunner):
    """Brackets the agent's writes so the log says who made them."""

    def produce(self, registry, task_id, **kw):
        log = registry.get("handoff_mgr").log
        log.append("<agent>")
        try:
            super().produce(registry, task_id, **kw)
        finally:
            log.append("</agent>")


def spans(log: list[str]) -> list[range]:
    """Index ranges between an `<agent>` marker and its close."""
    out, start = [], None
    for i, entry in enumerate(log):
        if entry == "<agent>":
            start = i
        elif entry == "</agent>":
            out.append(range(start, i))
            start = None
    return out


def test_only_the_agent_writes_handoff_state(store):
    """A full submit -> dispatch -> complete -> resume -> re-dispatch cycle."""
    registry = build_registry(store=store, runner=MarkingRunner())
    registry.register("handoff_mgr", SpyHandoffMgr(registry))
    registry.get("agent_mgr").register("profiler")
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    spy = registry.get("handoff_mgr")

    (mid,) = new_handoffs(1)
    producer = make_task(outputs=[mid])
    consumer = make_task(inputs=[mid], outputs=new_handoffs(1))
    scheduler.submit(producer)
    scheduler.submit(consumer)

    runner.produce(registry, producer.id)
    runner.finish(producer.id, TaskStatus.FAILED)

    scheduler.resume_task(producer.id)
    runner.produce(registry, producer.id)
    runner.finish(producer.id)

    runner.produce(registry, consumer.id)
    runner.finish(consumer.id)

    agent_spans = spans(spy.log)
    assert len(agent_spans) == 3
    inside = {i for span in agent_spans for i in span}

    for i, entry in enumerate(spy.log):
        if entry in WRITES:
            assert i in inside, f"{entry} at {i} was called outside an agent"
        elif entry in READS:
            pass  # a read from anywhere is fine
        elif entry not in ("<agent>", "</agent>"):
            raise AssertionError(f"unexpected call {entry!r}")


def test_the_scheduler_alone_only_reads(store):
    """No agent runs at all: the log must contain no write."""
    registry = build_registry(store=store, runner=MarkingRunner())
    registry.register("handoff_mgr", SpyHandoffMgr(registry))
    registry.get("agent_mgr").register("profiler")
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    spy = registry.get("handoff_mgr")

    producer = make_task(outputs=new_handoffs(1))
    scheduler.submit(producer)
    scheduler.submit(make_task(inputs=producer.outputs))
    runner.finish(producer.id)  # completes without writing anything

    assert not any(entry in WRITES for entry in spy.log)
    assert "check_if_latest_valid" in spy.log


def test_the_spy_would_catch_a_scheduler_write(store):
    """The test is only meaningful if it can fail. Simulate the erosion."""
    registry = build_registry(store=store, runner=MarkingRunner())
    registry.register("handoff_mgr", SpyHandoffMgr(registry))
    registry.get("agent_mgr").register("profiler")
    spy = registry.get("handoff_mgr")

    task = make_task(outputs=new_handoffs(1))
    registry.get("scheduler").submit(task)
    spy.persist(task.outputs[0])  # a write from outside any agent span

    inside = {i for span in spans(spy.log) for i in span}
    offenders = [i for i, e in enumerate(spy.log) if e in WRITES and i not in inside]
    assert offenders, "the spy failed to notice a write outside an agent"


def test_scheduling_never_looks_at_a_handoff_payload():
    """A payload that raises if anyone inspects it must still flow through."""

    class Untouchable(dict):
        """A dict, so it serialises — but it screams if anyone looks inside."""

        def __getitem__(self, key):
            raise AssertionError("something read the content")

        def __bool__(self):
            raise AssertionError("something tested the content for truth")

    registry = build_registry(store=MemoryStoreMgr())
    registry.get("agent_mgr").register("profiler")
    scheduler, runner = registry.get("scheduler"), registry.get("runner")
    payload = Untouchable(verdict="do not read me")

    producer = make_task(outputs=new_handoffs(1))
    consumer = make_task(inputs=producer.outputs)
    scheduler.submit(producer)
    scheduler.submit(consumer)

    runner.produce(registry, producer.id, content=payload)
    runner.finish(producer.id)

    assert registry.get("task_mgr").get(consumer.id).status is TaskStatus.RUNNING


def test_a_payload_must_be_serialisable(store):
    """Design open question O9. The *scheduler* is content-agnostic, but
    `persist` dumps the whole handoff, so an arbitrary Python object cannot be
    a payload today. Asserted so the constraint is visible rather than
    discovered by the first agent that returns one."""
    import pytest
    from pydantic_core import PydanticSerializationError

    class Opaque:
        pass

    registry = build_registry(store=store)
    registry.get("agent_mgr").register("profiler")
    task = make_task(outputs=new_handoffs(1))
    registry.get("scheduler").submit(task)

    with pytest.raises(PydanticSerializationError):
        registry.get("runner").produce(registry, task.id, content=Opaque())


def test_the_scheduler_never_takes_a_mutable_handle():
    """`persist` is not the only way to write handoff state.

    `HandoffMgr.get` returns a live `Handoff`, on which `open_next` and `seal`
    are one call away — and those live on the model, where the spy cannot see
    them (design O12). The boundary therefore rests on the scheduler never
    taking such a handle in the first place, which is a property of the source,
    not of a run. Checked statically because no test can observe it.
    """
    import inspect

    from task_graph import scheduler as scheduler_module

    source = inspect.getsource(scheduler_module)
    assert "handoff_mgr.get(" not in source, (
        "the scheduler took a mutable Handoff; open_next and seal are one call away "
        "and the authority spy cannot see them"
    )
    for verb in ("open_next", "seal("):
        assert verb not in source, f"the scheduler calls {verb} directly"
