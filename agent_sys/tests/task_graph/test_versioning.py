"""Versioning — criteria 16, 17 and 20.

A re-run appends; it never overwrites. Nothing invalidates a downstream handoff,
because a consumer that has already run has already recorded which version it
saw, and one that has not will ask again.
"""

import pytest

from task_graph.bootstrap import build_registry
from task_graph.models import HandoffStatus, TaskStatus
from task_graph.store import MemoryStoreMgr

from .conftest import make_task, new_handoffs


def test_a_second_producer_appends_rather_than_overwriting(
    scheduler, handoff_mgr, runner, registry
):
    """Criterion 16."""
    (hid,) = new_handoffs(1)
    first = make_task(outputs=[hid])
    scheduler.submit(first)
    runner.produce(registry, first.id, content="first")
    runner.finish(first.id)

    second = make_task(outputs=[hid])
    scheduler.submit(second)
    runner.produce(registry, second.id, content="second")
    runner.finish(second.id)

    handoff = handoff_mgr.get(hid)
    assert [v.version for v in handoff.versions] == [0, 1]
    assert handoff.get(0).content == "first"
    assert handoff.get(1).content == "second"


def test_the_earlier_version_is_untouched(scheduler, handoff_mgr, runner, registry):
    """Criterion 17."""
    (hid,) = new_handoffs(1)
    first = make_task(outputs=[hid])
    scheduler.submit(first)
    runner.produce(registry, first.id, content="first")
    runner.finish(first.id)
    before = handoff_mgr.get(hid).get(0).model_copy(deep=True)

    second = make_task(outputs=[hid])
    scheduler.submit(second)
    runner.produce(registry, second.id, valid=False, content="second")
    runner.finish(second.id)

    assert handoff_mgr.get(hid).get(0) == before


def test_a_version_records_which_task_and_which_agent_wrote_it(
    scheduler, handoff_mgr, task_mgr, runner, registry
):
    (hid,) = new_handoffs(1)
    first, second = make_task(outputs=[hid]), make_task(outputs=[hid])
    for task in (first, second):
        scheduler.submit(task)
        runner.produce(registry, task.id)
        runner.finish(task.id)

    handoff = handoff_mgr.get(hid)
    assert handoff.get(0).producer_task_id == first.id
    assert handoff.get(1).producer_task_id == second.id
    assert handoff.get(0).producer_agent_id == task_mgr.get(first.id).current.agent_id
    assert handoff.get(1).producer_agent_id == task_mgr.get(second.id).current.agent_id
    assert handoff.get(0).producer_agent_id != handoff.get(1).producer_agent_id


def test_a_re_run_makes_the_slot_temporarily_not_valid(scheduler, handoff_mgr, runner, registry):
    """Which is what blocks a consumer that has not started yet."""
    (hid,) = new_handoffs(1)
    first = make_task(outputs=[hid])
    scheduler.submit(first)
    runner.produce(registry, first.id)
    runner.finish(first.id)
    assert handoff_mgr.check_if_latest_valid(hid)

    second = make_task(outputs=[hid])
    scheduler.submit(second)  # dispatched, agent has not written yet
    consumer = make_task(inputs=[hid])
    scheduler.submit(consumer)
    assert handoff_mgr.check_if_latest_valid(hid)  # v0 is still the latest

    runner.produce(registry, second.id)  # now v1 exists and is valid
    assert handoff_mgr.latest(hid).version == 1


def test_a_consumer_submitted_after_a_re_run_sees_the_new_version(
    scheduler, task_mgr, runner, registry
):
    """Criterion 20."""
    (hid,) = new_handoffs(1)
    for content in ("first", "second"):
        producer = make_task(outputs=[hid])
        scheduler.submit(producer)
        runner.produce(registry, producer.id, content=content)
        runner.finish(producer.id)

    consumer = make_task(inputs=[hid])
    scheduler.submit(consumer)
    assert task_mgr.get(consumer.id).current.input_versions == {hid: 1}


def test_nothing_invalidates_a_downstream_handoff(
    scheduler, handoff_mgr, task_mgr, runner, registry
):
    """A consumer that already ran keeps its recorded version and its own
    output stays VALID: cascade invalidation is deliberately absent."""
    (upstream,) = new_handoffs(1)
    (downstream,) = new_handoffs(1)

    producer = make_task(outputs=[upstream])
    scheduler.submit(producer)
    runner.produce(registry, producer.id, content="v0")
    runner.finish(producer.id)

    consumer = make_task(inputs=[upstream], outputs=[downstream])
    scheduler.submit(consumer)
    runner.produce(registry, consumer.id)
    runner.finish(consumer.id)

    refresher = make_task(outputs=[upstream])
    scheduler.submit(refresher)
    runner.produce(registry, refresher.id, content="v1")
    runner.finish(refresher.id)

    assert handoff_mgr.check_if_latest_valid(downstream)
    assert handoff_mgr.get(downstream).latest.status is HandoffStatus.VALID
    assert task_mgr.get(consumer.id).status is TaskStatus.SUCCEEDED
    assert task_mgr.get(consumer.id).history[0].input_versions == {upstream: 0}


def test_a_resumed_task_writes_a_new_version_not_over_its_old_one(
    scheduler, handoff_mgr, runner, registry
):
    (hid,) = new_handoffs(1)
    task = make_task(outputs=[hid])
    scheduler.submit(task)
    runner.produce(registry, task.id, content="attempt 0")
    runner.finish(task.id, TaskStatus.FAILED)

    scheduler.resume_task(task.id)
    runner.produce(registry, task.id, content="attempt 1")
    runner.finish(task.id)

    handoff = handoff_mgr.get(hid)
    assert [v.content for v in handoff.versions] == ["attempt 0", "attempt 1"]
    assert handoff.get(0).producer_task_id == handoff.get(1).producer_task_id == task.id
    assert handoff.get(0).producer_agent_id != handoff.get(1).producer_agent_id


def test_versions_survive_a_restart(scheduler, runner, registry, store):
    from .conftest import rebuild

    (hid,) = new_handoffs(1)
    for content in ("first", "second"):
        producer = make_task(outputs=[hid])
        scheduler.submit(producer)
        runner.produce(registry, producer.id, content=content)
        runner.finish(producer.id)

    fresh = rebuild(store)
    fresh.get("handoff_mgr").resume_system()

    handoff = fresh.get("handoff_mgr").get(hid)
    assert [v.content for v in handoff.versions] == ["first", "second"]
    assert fresh.get("handoff_mgr").check_if_latest_valid(hid)


def test_resuming_over_an_abandoned_generating_version_appends(
    scheduler, handoff_mgr, task_mgr, runner, registry
):
    """Criterion 20's other half: the abandoned version is not reused, so the
    record still shows that an attempt was made and left unfinished.

    NOTE the seal below. This test drives the case where *something* closes the
    abandoned version off; it does NOT cover a crash, where nobody is alive to
    make that call — see `test_recovery.py`'s deadlock test and design §14 O10."""
    (hid,) = new_handoffs(1)
    task = make_task(outputs=[hid])
    scheduler.submit(task)

    # the agent opened the slot and died without sealing
    first_agent = task_mgr.get(task.id).current.agent_id
    handoff_mgr.get(hid).open_next(task.id, first_agent)
    handoff_mgr.persist(hid)
    runner.finish(task.id, TaskStatus.FAILED)
    assert handoff_mgr.get(hid).latest.status is HandoffStatus.GENERATING

    scheduler.resume_task(task.id)
    second_agent = task_mgr.get(task.id).current.agent_id
    # `open_next` refuses a slot someone else has open — an abandoned version
    # must be sealed off before the retry can take the slot.
    handoff_mgr.get(hid).latest.seal(HandoffStatus.INVALID)
    runner.produce(registry, task.id, content="retry")
    runner.finish(task.id)

    handoff = handoff_mgr.get(hid)
    assert [v.status for v in handoff.versions] == [
        HandoffStatus.INVALID,
        HandoffStatus.VALID,
    ]
    assert handoff.get(0).producer_agent_id == first_agent
    assert handoff.get(1).producer_agent_id == second_agent


# ----------------------------------------- the two version spaces, and the gap


@pytest.fixture
def stored_registry(tmp_path):
    """A registry with a real `handoff_store`, which the shared one has not."""
    r = build_registry(store=MemoryStoreMgr(), handoff_root=str(tmp_path / "store"))
    r.get("agent_mgr").register("profiler")
    return r


def test_the_slot_reuses_a_store_version_the_store_has_burned(stored_registry):
    """**The two counters have incompatible reuse rules**, so they part company.

    `handoff.FilesystemStore.allocate` takes each number with an `os.mkdir`
    token, so a number it hands out is burned whether or not anything is written
    into it. `Handoff.open_next` adopts a `CREATED` latest **in place**. So a
    dispatch that allocates and never writes burns a store number that the slot
    then goes on to use for a different artefact.

    **One dispatch that does not write is enough** — and `handoff` records that
    as the ordinary case rather than the exceptional one, since the directory
    has to exist before `env_mgr.prepare` resolves the grant and a refused
    `prepare` is *"no isolation, no start"*.

    This is not the defect; it is the mechanism the defect rests on, and it is
    correct on both sides. Asserted so that a later change to either allocator
    has to come past it.
    """
    scheduler, runner = stored_registry.get("scheduler"), stored_registry.get("runner")
    handoff_mgr, task_mgr = stored_registry.get("handoff_mgr"), stored_registry.get("task_mgr")
    (hid,) = new_handoffs(1)
    task = make_task(outputs=[hid])
    scheduler.submit(task)

    # A dispatch that allocates and never writes: no produce(), so nothing on
    # the slot side advances at all.
    runner.finish(task.id, TaskStatus.FAILED)
    scheduler.resume_task(task.id)
    runner.produce(stored_registry, task.id)

    attempts = task_mgr.get(task.id).history
    assert [e.output_versions[hid] for e in attempts] == [0, 1], "the store burned 0, then gave 1"
    assert handoff_mgr.latest(hid).version == 0, "the slot adopted the burned number"


@pytest.mark.xfail(
    strict=True,
    reason="ruled and not yet built: `input_versions` is slot-space while the grant "
    "resolves it as a store path. The fix moves the gate and the grant together and "
    "spans task_graph, env_mgr and handoff — see task_graph/README.md. This xfail is "
    "strict so the day it lands, this goes red and asks to be turned into an assertion.",
)
def test_a_consumer_is_pinned_to_the_version_its_producer_actually_published(stored_registry):
    """**The defect, executable.** Nothing else in the suite can catch it.

    `env_mgr/grants.py` resolves `input_versions` through `handoff_version_dir`
    — a **store** path — but `scheduler.py` fills it from `HandoffMgr.latest`, a
    **slot** number. Once the two have parted, a consumer is granted and staged
    the directory of a version that was allocated, never written and never
    sealed.

    **And every guard reports valid.** `allocate` must create `v<N>/content/`
    for the ruleset to open it, so the directory exists and confinement builds;
    `env_mgr`'s staging skips only an *absent* `content/`; staging is a
    `copytree` rather than `handoff.copy_out`, so no digest is verified. The
    body receives an empty directory presented as the artefact.

    Measured end to end in
    `scratch/impl-2026-08/task_graph/probe_consumer_staging.py`, which stages
    both numbers against one store and shows `[]` versus the artefact's files.
    """
    scheduler, runner = stored_registry.get("scheduler"), stored_registry.get("runner")
    task_mgr = stored_registry.get("task_mgr")
    (hid,) = new_handoffs(1)
    producer = make_task(outputs=[hid])
    scheduler.submit(producer)
    runner.finish(producer.id, TaskStatus.FAILED)  # allocated v0, wrote nothing
    scheduler.resume_task(producer.id)
    runner.produce(stored_registry, producer.id)  # writes, and is pinned to v1
    runner.finish(producer.id, TaskStatus.SUCCEEDED)

    published = task_mgr.get(producer.id).history[-1].output_versions[hid]

    consumer = make_task(inputs=[hid])
    scheduler.submit(consumer)

    assert task_mgr.get(consumer.id).current.input_versions[hid] == published
