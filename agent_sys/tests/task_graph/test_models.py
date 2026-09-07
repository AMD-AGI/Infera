"""Domain models and their state machines — criteria 26 and 30.

The guards are the design: `seal` only from GENERATING, `open_next` refuses a
slot someone else has open, `push_execution` refuses to stack a second attempt.
Everything downstream trusts these to hold.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from task_graph.ids import AgentId, HandoffId, TaskId
from task_graph.models import (
    RESUMABLE,
    WAITING,
    Agent,
    Execution,
    Handoff,
    HandoffRef,
    HandoffStateError,
    HandoffStatus,
    HandoffVersion,
    Task,
    TaskStateError,
    TaskStatus,
)


def make_handoff(**kw) -> Handoff:
    """A freshly declared slot: one CREATED v0, as `declare` makes it."""
    kw.setdefault("id", HandoffId.new())
    kw.setdefault("versions", [HandoffVersion(version=0)])
    return Handoff(**kw)


# ---------------------------------------------------------------- enums


def test_enums_are_plain_strings_when_dumped():
    assert TaskStatus.RUNNING.value == "running"
    assert HandoffStatus.CREATED.value == "created"
    assert json.dumps(TaskStatus.RUNNING) == '"running"'


def test_waiting_and_resumable_sets():
    assert WAITING == {TaskStatus.WAITING_HANDOFF, TaskStatus.WAITING_RESOURCE}
    assert RESUMABLE == {TaskStatus.FAILED, TaskStatus.SUSPENDED}


# ---------------------------------------------------------- model config


def test_an_unknown_field_is_rejected_not_dropped():
    with pytest.raises(ValidationError):
        Task(agent_spec="profiler", agnet_spec="typo")


def test_assignment_is_validated():
    task = Task(agent_spec="profiler")
    with pytest.raises(ValidationError):
        task.status = "not-a-status"


def test_a_valid_assignment_keeps_the_enum_member():
    task = Task(agent_spec="profiler")
    task.status = "running"
    assert task.status is TaskStatus.RUNNING


# ------------------------------------------------------- HandoffVersion


def test_a_fresh_version_is_created_and_not_valid():
    version = HandoffVersion(version=0)
    assert version.status is HandoffStatus.CREATED
    assert not version.is_valid
    assert version.producer_task_id is None
    assert version.producer_agent_id is None
    assert isinstance(version.timestamp, datetime)


def test_seal_from_generating_to_valid():
    version = HandoffVersion(version=0, status=HandoffStatus.GENERATING)
    version.seal(HandoffStatus.VALID, content={"latency": 12})
    assert version.status is HandoffStatus.VALID
    assert version.is_valid
    assert version.content == {"latency": 12}


def test_seal_from_generating_to_invalid():
    version = HandoffVersion(version=0, status=HandoffStatus.GENERATING)
    version.seal(HandoffStatus.INVALID)
    assert version.status is HandoffStatus.INVALID
    assert not version.is_valid


@pytest.mark.parametrize(
    "status", [HandoffStatus.CREATED, HandoffStatus.VALID, HandoffStatus.INVALID]
)
def test_seal_requires_generating(status):
    version = HandoffVersion(version=0, status=status)
    with pytest.raises(HandoffStateError):
        version.seal(HandoffStatus.VALID)


@pytest.mark.parametrize("verdict", [HandoffStatus.CREATED, HandoffStatus.GENERATING])
def test_seal_requires_a_verdict(verdict):
    version = HandoffVersion(version=0, status=HandoffStatus.GENERATING)
    with pytest.raises(HandoffStateError):
        version.seal(verdict)


# --------------------------------------------------------------- Handoff


def test_latest_is_the_last_version():
    handoff = make_handoff()
    assert handoff.latest is handoff.versions[-1]
    assert handoff.latest.version == 0


def test_versions_may_not_be_empty():
    with pytest.raises(ValidationError):
        Handoff(id=HandoffId.new(), versions=[])


def test_get_by_version_number():
    handoff = make_handoff()
    assert handoff.get(0) is handoff.versions[0]
    with pytest.raises(IndexError):
        handoff.get(7)


def test_is_latest_valid_tracks_the_latest_only():
    handoff = make_handoff()
    assert not handoff.is_latest_valid

    handoff.open_next(TaskId.new(), AgentId.new()).seal(HandoffStatus.VALID)
    assert handoff.is_latest_valid

    handoff.open_next(TaskId.new(), AgentId.new())  # a re-run reopens the slot
    assert not handoff.is_latest_valid


def test_open_next_adopts_a_created_latest_in_place():
    handoff = make_handoff()
    tid, aid = TaskId.new(), AgentId.new()

    version = handoff.open_next(tid, aid)

    assert len(handoff.versions) == 1
    assert version is handoff.versions[0]
    assert version.version == 0
    assert version.status is HandoffStatus.GENERATING
    assert version.producer_task_id == tid
    assert version.producer_agent_id == aid


@pytest.mark.parametrize("verdict", [HandoffStatus.VALID, HandoffStatus.INVALID])
def test_open_next_appends_after_a_sealed_latest(verdict):
    handoff = make_handoff()
    first_task, first_agent = TaskId.new(), AgentId.new()
    handoff.open_next(first_task, first_agent).seal(verdict, content="v0")

    second_task, second_agent = TaskId.new(), AgentId.new()
    version = handoff.open_next(second_task, second_agent)

    assert len(handoff.versions) == 2
    assert version.version == 1
    assert version.status is HandoffStatus.GENERATING
    assert version.producer_task_id == second_task
    assert version.producer_agent_id == second_agent

    # the earlier version is untouched — criterion 17
    assert handoff.get(0).status is verdict
    assert handoff.get(0).content == "v0"
    assert handoff.get(0).producer_task_id == first_task
    assert handoff.get(0).producer_agent_id == first_agent


def test_open_next_refuses_a_slot_someone_else_has_open():
    handoff = make_handoff()
    handoff.open_next(TaskId.new(), AgentId.new())
    with pytest.raises(HandoffStateError):
        handoff.open_next(TaskId.new(), AgentId.new())


def test_the_list_index_is_the_version_number():
    handoff = make_handoff()
    for _ in range(4):
        handoff.open_next(TaskId.new(), AgentId.new()).seal(HandoffStatus.VALID)
    assert [v.version for v in handoff.versions] == [0, 1, 2, 3]
    assert all(handoff.get(i).version == i for i in range(4))


# ------------------------------------------------------------- Execution


def test_an_execution_is_open_until_it_ends():
    execution = Execution(attempt=0, agent_id=AgentId.new())
    assert execution.is_open
    assert execution.outcome is None
    execution.ended_at = datetime.now()
    assert not execution.is_open


# ------------------------------------------------------------------ Task


def test_a_fresh_task_has_an_id_and_waits_for_handoffs():
    task = Task(agent_spec="profiler")
    assert isinstance(task.id, TaskId)
    assert task.status is TaskStatus.WAITING_HANDOFF
    assert task.history == []
    assert task.current is None
    assert not task.is_running
    assert not task.expedited


def test_push_execution_numbers_attempts_from_zero():
    task = Task(agent_spec="profiler")
    hid = HandoffId.new()

    first = task.push_execution(agent_id=AgentId.new(), input_versions={hid: 0})
    assert first.attempt == 0
    assert task.current is first
    assert task.is_running
    assert first.input_versions == {hid: 0}

    task.close_execution(TaskStatus.FAILED)
    second = task.push_execution(agent_id=AgentId.new(), input_versions={hid: 1})
    assert second.attempt == 1
    assert task.current is second
    assert len(task.history) == 2


def test_push_execution_refuses_to_stack_on_an_open_attempt():
    task = Task(agent_spec="profiler")
    task.push_execution(agent_id=AgentId.new(), input_versions={})
    with pytest.raises(TaskStateError):
        task.push_execution(agent_id=AgentId.new(), input_versions={})


def test_close_execution_seals_the_stack_top():
    task = Task(agent_spec="profiler")
    hid = HandoffId.new()
    task.push_execution(agent_id=AgentId.new(), input_versions={}, output_versions={hid: 2})

    task.close_execution(TaskStatus.SUCCEEDED, detail="done")

    top = task.history[-1]
    assert not top.is_open
    assert top.ended_at is not None
    assert top.outcome is TaskStatus.SUCCEEDED
    assert top.output_versions == {hid: 2}
    assert top.detail == "done"
    assert not task.is_running
    assert task.current is top  # `current` is the stack top, open or not


def test_close_execution_requires_an_open_attempt():
    task = Task(agent_spec="profiler")
    with pytest.raises(TaskStateError):
        task.close_execution(TaskStatus.SUCCEEDED)

    task.push_execution(agent_id=AgentId.new(), input_versions={})
    task.close_execution(TaskStatus.SUCCEEDED)
    with pytest.raises(TaskStateError):
        task.close_execution(TaskStatus.SUCCEEDED)


def test_history_is_a_stack_and_earlier_attempts_survive():
    task = Task(agent_spec="profiler")
    first_agent, second_agent = AgentId.new(), AgentId.new()

    task.push_execution(agent_id=first_agent, input_versions={})
    task.close_execution(TaskStatus.FAILED)
    task.push_execution(agent_id=second_agent, input_versions={})

    assert [e.agent_id for e in task.history] == [first_agent, second_agent]
    assert task.current.agent_id == second_agent


# ----------------------------------------------------------------- Agent


def test_an_agent_records_its_task_and_what_it_touched():
    tid, hid = TaskId.new(), HandoffId.new()
    agent = Agent(spec="profiler", task_id=tid)
    agent.handoffs.append(HandoffRef(handoff_id=hid, version=1))

    assert isinstance(agent.id, AgentId)
    assert agent.task_id == tid
    assert agent.handoffs[0].handoff_id == hid
    assert agent.handoffs[0].version == 1
    assert agent.knowledge is None
    assert agent.config == {}


# --------------------------------------------------------- serialisation
# Criterion 30: round trip through JSON, nothing hand-written per model.


def round_trip(model):
    return type(model).model_validate(json.loads(model.model_dump_json()))


def test_task_round_trip_including_handoff_id_dict_keys():
    hid_in, hid_out = HandoffId.new(), HandoffId.new()
    task = Task(
        agent_spec="profiler",
        inputs=[hid_in],
        outputs=[hid_out],
        depends_on=[TaskId.new()],
        resources={"gpu": 2.0},
        expedited=True,
    )
    task.push_execution(
        agent_id=AgentId.new(), input_versions={hid_in: 0}, output_versions={hid_out: 3}
    )
    task.close_execution(TaskStatus.SUCCEEDED, detail="ok")

    back = round_trip(task)
    assert back == task
    assert isinstance(back.id, TaskId)
    assert isinstance(back.inputs[0], HandoffId)
    assert isinstance(back.depends_on[0], TaskId)
    assert back.status is TaskStatus.WAITING_HANDOFF
    assert back.history[0].outcome is TaskStatus.SUCCEEDED
    assert back.history[0].input_versions == {hid_in: 0}
    assert back.history[0].output_versions == {hid_out: 3}
    assert isinstance(next(iter(back.history[0].input_versions)), HandoffId)
    assert back.created_at == task.created_at


def test_handoff_round_trip():
    handoff = make_handoff(type="profile")
    handoff.open_next(TaskId.new(), AgentId.new()).seal(HandoffStatus.VALID, content={"p50": 7})
    handoff.open_next(TaskId.new(), AgentId.new())

    back = round_trip(handoff)
    assert back == handoff
    assert back.get(0).status is HandoffStatus.VALID
    assert back.get(0).content == {"p50": 7}
    assert back.latest.status is HandoffStatus.GENERATING
    assert isinstance(back.get(0).producer_task_id, TaskId)
    assert isinstance(back.get(0).producer_agent_id, AgentId)


def test_agent_round_trip():
    agent = Agent(spec="profiler", task_id=TaskId.new(), config={"model": "opus"})
    agent.handoffs.append(HandoffRef(handoff_id=HandoffId.new(), version=2))

    back = round_trip(agent)
    assert back == agent
    assert isinstance(back.handoffs[0].handoff_id, HandoffId)


def test_a_restored_model_is_still_guarded():
    """Validation is not a one-off at construction — the state machine survives."""
    handoff = round_trip(make_handoff())
    handoff.open_next(TaskId.new(), AgentId.new())
    with pytest.raises(HandoffStateError):
        handoff.open_next(TaskId.new(), AgentId.new())
