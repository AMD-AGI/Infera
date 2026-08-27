"""HandoffMgr — the collection, and nothing more.

The mgr decides nothing: `check_if_latest_valid` delegates to the handoff, which
delegates to the version. One definition of "usable", on the object that has it.
"""

import pytest

from task_graph.handoff import HandoffMgr
from task_graph.ids import AgentId, HandoffId, TaskId
from task_graph.models import HandoffStatus
from task_graph.registry import Registry
from task_graph.store import MemoryStoreMgr


@pytest.fixture
def store():
    return MemoryStoreMgr()


@pytest.fixture
def mgr(store):
    registry = Registry()
    registry.register("store_mgr", store)
    manager = HandoffMgr(registry)
    registry.register("handoff_mgr", manager)
    return manager


def seal(mgr, hid, status=HandoffStatus.VALID, content=None):
    """What an agent does: take a version and seal it."""
    version = mgr.get(hid).open_next(TaskId.new(), AgentId.new())
    version.seal(status, content)
    mgr.persist(hid)
    return version


# ----------------------------------------------------------------- declare


def test_declare_creates_one_created_v0(mgr):
    hid, tid = HandoffId.new(), TaskId.new()
    mgr.declare([hid], producer_task_id=tid)

    handoff = mgr.get(hid)
    assert len(handoff.versions) == 1
    assert handoff.latest.version == 0
    assert handoff.latest.status is HandoffStatus.CREATED
    assert handoff.latest.producer_task_id == tid
    assert handoff.latest.producer_agent_id is None


def test_declare_persists(mgr, store):
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    assert store.exists("handoff", str(hid))


def test_declare_accepts_types(mgr):
    a, b = HandoffId.new(), HandoffId.new()
    mgr.declare([a, b], producer_task_id=TaskId.new(), types={a: "profile"})
    assert mgr.get(a).type == "profile"
    assert mgr.get(b).type == ""


def test_declare_is_idempotent_and_never_overwrites(mgr):
    """`update_task` re-declares; overwriting would delete written versions."""
    hid, first_task = HandoffId.new(), TaskId.new()
    mgr.declare([hid], producer_task_id=first_task)
    seal(mgr, hid, content="already written")

    mgr.declare([hid], producer_task_id=TaskId.new())

    handoff = mgr.get(hid)
    assert len(handoff.versions) == 1
    assert handoff.latest.status is HandoffStatus.VALID
    assert handoff.latest.content == "already written"


def test_declare_of_an_empty_list_is_fine(mgr):
    mgr.declare([], producer_task_id=TaskId.new())
    assert mgr.all_ids() == []


# ------------------------------------------------------ check_if_latest_valid


def test_an_unknown_id_is_not_valid_rather_than_an_error(mgr):
    """A consumer may be submitted before its producer declares the slot."""
    assert mgr.check_if_latest_valid(HandoffId.new()) is False


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda m, h: None, False),  # CREATED
        (lambda m, h: m.get(h).open_next(TaskId.new(), AgentId.new()), False),  # GENERATING
        (lambda m, h: seal(m, h, HandoffStatus.VALID), True),
        (lambda m, h: seal(m, h, HandoffStatus.INVALID), False),
    ],
    ids=["created", "generating", "valid", "invalid"],
)
def test_validity_tracks_the_latest_version(mgr, make, expected):
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    make(mgr, hid)
    assert mgr.check_if_latest_valid(hid) is expected


def test_a_re_run_makes_a_valid_handoff_not_valid_again(mgr):
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    seal(mgr, hid)
    assert mgr.check_if_latest_valid(hid)

    mgr.get(hid).open_next(TaskId.new(), AgentId.new())
    assert not mgr.check_if_latest_valid(hid)


# -------------------------------------------------------------------- reads


def test_latest_returns_none_for_an_unknown_id(mgr):
    assert mgr.latest(HandoffId.new()) is None


def test_latest_returns_the_last_version(mgr):
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    seal(mgr, hid)
    seal(mgr, hid)
    assert mgr.latest(hid).version == 1


def test_get_raises_for_an_unknown_id(mgr):
    """Unlike `check_if_latest_valid`: asking for the object is a bug."""
    with pytest.raises(KeyError):
        mgr.get(HandoffId.new())


def test_get_many_preserves_order(mgr):
    ids = [HandoffId.new() for _ in range(3)]
    mgr.declare(ids, producer_task_id=TaskId.new())
    assert [h.id for h in mgr.get_many(ids)] == ids


def test_all_ids(mgr):
    ids = [HandoffId.new() for _ in range(3)]
    mgr.declare(ids, producer_task_id=TaskId.new())
    assert sorted(mgr.all_ids(), key=str) == sorted(ids, key=str)


def test_produced_by_task(mgr):
    first, second = TaskId.new(), TaskId.new()
    a, b, c = HandoffId.new(), HandoffId.new(), HandoffId.new()
    mgr.declare([a, b], producer_task_id=first)
    mgr.declare([c], producer_task_id=second)

    assert sorted(mgr.produced_by_task(first), key=str) == sorted([a, b], key=str)
    assert mgr.produced_by_task(second) == [c]
    assert mgr.produced_by_task(TaskId.new()) == []


def test_produced_by_task_follows_the_latest_producer(mgr):
    """A re-run by a different task moves the handoff's attribution."""
    original, rerun = TaskId.new(), TaskId.new()
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=original)
    seal(mgr, hid)

    mgr.get(hid).open_next(rerun, AgentId.new())
    mgr.persist(hid)

    assert mgr.produced_by_task(rerun) == [hid]
    assert mgr.produced_by_task(original) == []


# -------------------------------------------------------------- persistence


def test_persist_writes_the_whole_handoff_including_its_versions(mgr, store):
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    seal(mgr, hid, content={"p50": 7})
    seal(mgr, hid, HandoffStatus.INVALID)

    record = store.read("handoff", str(hid))
    assert len(record["versions"]) == 2
    assert record["versions"][0]["content"] == {"p50": 7}
    assert record["versions"][1]["status"] == "invalid"


def test_persist_of_an_unknown_id_raises(mgr):
    with pytest.raises(KeyError):
        mgr.persist(HandoffId.new())


def test_resume_reloads_everything_with_versions_intact(mgr, store):
    ids = [HandoffId.new() for _ in range(3)]
    mgr.declare(ids, producer_task_id=TaskId.new(), types={ids[0]: "profile"})
    seal(mgr, ids[0], content="v0")
    seal(mgr, ids[0], content="v1")
    seal(mgr, ids[1], HandoffStatus.INVALID)

    registry = Registry()
    registry.register("store_mgr", store)
    fresh = HandoffMgr(registry)
    assert fresh.all_ids() == []

    fresh.resume_system()

    assert sorted(fresh.all_ids(), key=str) == sorted(ids, key=str)
    assert fresh.get(ids[0]).type == "profile"
    assert [v.content for v in fresh.get(ids[0]).versions] == ["v0", "v1"]
    assert fresh.check_if_latest_valid(ids[0])
    assert not fresh.check_if_latest_valid(ids[1])
    assert not fresh.check_if_latest_valid(ids[2])


def test_resume_needs_no_regrouping_because_a_record_is_a_whole_handoff(mgr, store):
    """One record per handoff, not per version: directory order is not version
    order, and nesting removes the need to sort on the way back."""
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    for i in range(5):
        seal(mgr, hid, content=i)

    assert len(store.read_all("handoff")) == 1

    registry = Registry()
    registry.register("store_mgr", store)
    fresh = HandoffMgr(registry)
    fresh.resume_system()
    assert [v.version for v in fresh.get(hid).versions] == [0, 1, 2, 3, 4]
    assert [v.content for v in fresh.get(hid).versions] == [0, 1, 2, 3, 4]


def test_resume_replaces_rather_than_merges(mgr, store):
    hid = HandoffId.new()
    mgr.declare([hid], producer_task_id=TaskId.new())
    store.delete("handoff", str(hid))

    mgr.resume_system()
    assert mgr.all_ids() == []
