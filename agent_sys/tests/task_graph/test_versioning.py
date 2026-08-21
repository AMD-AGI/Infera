"""Versioning — criteria 16, 17 and 20.

A re-run appends; it never overwrites. Nothing invalidates a downstream handoff,
because a consumer that has already run has already recorded which version it
saw, and one that has not will ask again.
"""

from task_graph.models import HandoffStatus, TaskStatus

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
