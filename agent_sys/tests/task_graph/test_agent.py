"""AgentMgr — criteria 28 and 34.

Two collections: the spec table of what kinds exist, and the instances of what
has been created. The mgr keeps what it creates; a factory that instantiated and
forgot would leave every `Execution.agent_id` pointing at nothing.
"""

import pytest

from task_graph.agent import AgentMgr
from task_graph.ids import AgentId, HandoffId, TaskId
from task_graph.models import HandoffRef
from task_graph.registry import Registry
from task_graph.store import MemoryStoreMgr


@pytest.fixture
def store():
    return MemoryStoreMgr()


@pytest.fixture
def registry(store):
    r = Registry()
    r.register("store_mgr", store)
    return r


@pytest.fixture
def mgr(registry):
    manager = AgentMgr(registry)
    registry.register("agent_mgr", manager)
    manager.register("profiler")
    return manager


# -------------------------------------------------------------- spec table


def test_register_and_list_specs(mgr):
    mgr.register("tuner", model="opus")
    assert sorted(mgr.specs()) == ["profiler", "tuner"]
    assert mgr.is_registered("tuner")
    assert not mgr.is_registered("nope")


def test_a_spec_config_reaches_the_instance(mgr):
    mgr.register("tuner", model="opus", retries=2)
    agent = mgr.instantiate("tuner", TaskId.new())
    assert agent.config == {"model": "opus", "retries": 2}


def test_registering_a_spec_twice_replaces_its_config(mgr):
    mgr.register("tuner", model="sonnet")
    mgr.register("tuner", model="opus")
    assert mgr.instantiate("tuner", TaskId.new()).config == {"model": "opus"}


# ------------------------------------------------------------- instantiate


def test_instantiate_retains_the_agent(mgr):
    """Criterion 28: otherwise the audit trail names agents nobody can resolve."""
    agent = mgr.instantiate("profiler", TaskId.new())
    assert mgr.get(agent.id) is agent
    assert mgr.all() == [agent]


def test_instantiate_binds_the_task(mgr):
    tid = TaskId.new()
    agent = mgr.instantiate("profiler", tid)
    assert agent.task_id == tid
    assert agent.spec == "profiler"
    assert agent.handoffs == []


def test_instantiate_mints_a_fresh_id_every_call(mgr):
    """Criterion 21: after a resume the stack top must report a different
    agent_id than the entry beneath it."""
    tid = TaskId.new()
    first, second = mgr.instantiate("profiler", tid), mgr.instantiate("profiler", tid)
    assert first.id != second.id
    assert len(mgr.all()) == 2


def test_instantiate_an_unknown_spec_names_the_registered_ones(mgr):
    with pytest.raises(KeyError, match="profiler"):
        mgr.instantiate("nope", TaskId.new())


def test_instantiate_persists(mgr, store):
    agent = mgr.instantiate("profiler", TaskId.new())
    assert store.read("agent", str(agent.id))["spec"] == "profiler"


# --------------------------------------------------------------- retrieval


def test_get_by_id(mgr):
    agent = mgr.instantiate("profiler", TaskId.new())
    assert mgr.get(agent.id) is agent


def test_get_an_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.get(AgentId.new())


def test_get_by_spec_name_makes_an_unbound_agent(mgr):
    """The `get(name) -> agent` mission.md asks for."""
    agent = mgr.get("profiler")
    assert agent.spec == "profiler"
    assert agent.task_id is None
    assert mgr.get(agent.id) is agent


def test_get_by_an_unknown_spec_name_raises(mgr):
    with pytest.raises(KeyError):
        mgr.get("nope")


def test_by_spec(mgr):
    mgr.register("tuner")
    profilers = [mgr.instantiate("profiler", TaskId.new()) for _ in range(2)]
    tuner = mgr.instantiate("tuner", TaskId.new())

    assert sorted(a.id for a in mgr.by_spec("profiler")) == sorted(a.id for a in profilers)
    assert [a.id for a in mgr.by_spec("tuner")] == [tuner.id]
    assert mgr.by_spec("nope") == []


def test_by_task_returns_every_attempt_in_order(mgr):
    tid = TaskId.new()
    first, second = mgr.instantiate("profiler", tid), mgr.instantiate("profiler", tid)
    mgr.instantiate("profiler", TaskId.new())

    assert [a.id for a in mgr.by_task(tid)] == [first.id, second.id]
    assert mgr.by_task(TaskId.new()) == []


def test_retire(mgr, store):
    agent = mgr.instantiate("profiler", TaskId.new())
    mgr.retire(agent.id)

    assert mgr.all() == []
    assert not store.exists("agent", str(agent.id))
    with pytest.raises(KeyError):
        mgr.retire(agent.id)


# -------------------------------------------------------------- persistence


def test_persist_writes_back_what_the_agent_appended(mgr, store):
    agent = mgr.instantiate("profiler", TaskId.new())
    hid = HandoffId.new()
    agent.handoffs.append(HandoffRef(handoff_id=hid, version=1))

    assert store.read("agent", str(agent.id))["handoffs"] == []
    mgr.persist(agent.id)
    assert store.read("agent", str(agent.id))["handoffs"] == [
        {"handoff_id": str(hid), "version": 1}
    ]


def test_persist_an_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.persist(AgentId.new())


def test_instances_are_restored_so_the_audit_trail_resolves(mgr, registry):
    """Criterion 34."""
    tid, hid = TaskId.new(), HandoffId.new()
    agent = mgr.instantiate("profiler", tid)
    agent.handoffs.append(HandoffRef(handoff_id=hid, version=0))
    mgr.persist(agent.id)

    fresh = AgentMgr(registry)
    assert fresh.all() == []
    fresh.resume_system()

    restored = fresh.get(agent.id)
    assert restored.spec == "profiler"
    assert restored.task_id == tid
    assert restored.handoffs[0].handoff_id == hid
    assert [a.id for a in fresh.by_task(tid)] == [agent.id]


def test_the_spec_table_is_not_restored(mgr, registry):
    """It is configuration. Reading it back would resurrect a spec the operator
    has since removed."""
    mgr.instantiate("profiler", TaskId.new())

    fresh = AgentMgr(registry)
    fresh.resume_system()

    assert fresh.specs() == []
    with pytest.raises(KeyError):
        fresh.instantiate("profiler", TaskId.new())


def test_a_restored_instance_is_a_record_not_a_live_agent(mgr, registry):
    """Nothing dispatches against one: instantiate always creates a new one."""
    agent = mgr.instantiate("profiler", TaskId.new())

    fresh = AgentMgr(registry)
    fresh.resume_system()
    fresh.register("profiler")
    replacement = fresh.instantiate("profiler", TaskId.new())

    assert replacement.id != agent.id
    assert len(fresh.all()) == 2


def test_resume_replaces_rather_than_merging(mgr, store):
    agent = mgr.instantiate("profiler", TaskId.new())
    store.delete("agent", str(agent.id))

    mgr.resume_system()
    assert mgr.all() == []
