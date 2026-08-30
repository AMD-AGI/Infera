"""The event stream. Criteria 4, 5, 10 and 14, plus the guards on the format.

Criterion 14 makes the machine-readable stream **an interface**: the acceptance
criteria are assertions over it, so a change to `EventKind`, to `Event`'s
fields, or to what `render/machine.py` emits is a change to a contract. That is
why `SCHEMA_VERSION` exists and why two of the tests here are about the format
rather than about the run.
"""

from __future__ import annotations

import datetime
import io
import json
import pathlib
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest

from cli import expectations as cli_expectations
from cli import main as cli_main
from cli.events import SCHEMA_VERSION, Event, EventKind
from cli.render.human import HumanRenderer, line_for
from cli.render.machine import JsonLinesRenderer, as_object
from cli.stream import Stream
from handoff import Verdict
from monitor import EventKind as MonitorEventKind
from monitor import event as monitor_event
from task_graph import TaskStatus
from tests.cli.conftest import by_closure

#: `examples/demo`'s expectation set, which every accounting test here drives.
DEMO = cli_expectations.DEMO

# --------------------------------------------------------------------------- #
# A driven run, with no model, no credential and no sandbox.


def _content(tmp, kind: str, body: str):
    """A publishable content directory for one of the demo's two kinds.

    Built to what the **content type** requires — `structured_text` wants a
    `Purpose` and a `Schema` section plus one of the three text items; `text`
    wants a `Purpose` and an item called `content` — because `put` runs the
    README check and the items check before it creates anything, and a demo
    fixture that dodged them would be testing a different `put`.
    """
    root = tmp / f"content-{kind}-{body[:6]}"
    (root / "items").mkdir(parents=True, exist_ok=True)
    if kind == "facts":
        (root / "README.md").write_text(
            "# facts\n\n## Purpose\n\nA manifest.\n\n## Schema\n\nrows and totals.\n"
        )
        (root / "items" / "text.json").write_text(body)
    else:
        (root / "README.md").write_text(
            "# summary\n\n## Purpose\n\nA summary.\n\n## Grounding\n\nEvery number is in facts.\n"
        )
        (root / "items" / "content").write_text(body)
        # **`grounding` is a required item now** — the task declaration passing
        # its own input through, so `check_grounded` can judge a summary from
        # the summary alone (the `summary` kind in `steps/describe.yaml`). A tree, because the
        # body copies the facts artefact verbatim rather than selecting from it.
        # Built here rather than schema-dodged for the reason above: `put` runs
        # the items check, and a fixture that skipped it would test a different
        # `put`.
        (root / "items" / "grounding").mkdir(exist_ok=True)
        (root / "items" / "grounding" / "text.json").write_text(body)
    return root


@pytest.fixture()
def run(registry: Any, submitted: Any, tmp_path, stream: Stream) -> Stream:
    """The demo's graph driven to quiescence, emitting the demo's own events.

    `FakeRunner` stands in for the executors — it is the only thing in the suite
    that writes handoff state, which is what makes the authority assertions
    below mean something. The publication and the verdicts go through the real
    `FilesystemStore`, so `put` → `copy_out` → `record_verdict` is exercised
    here rather than mocked.

    The root is the only task submitted; the three subtasks arrive when its main
    phase unfolds, which is how a run actually grows a graph.
    """
    runner, store = registry.get("runner"), registry.get("handoff_store")
    task_mgr, agent_mgr = registry.get("task_mgr"), registry.get("agent_mgr")

    runner.advance(registry, submitted.id)  # main -> RUNNING; the unfold is here
    tasks = task_mgr.all()
    assert len(tasks) == 4, [t.closure for t in tasks]

    for closure, kind, body, valid in (
        ("produce", "facts", json.dumps({"rows": [], "totals": {"files": 0, "lines": 0}}), True),
        ("describe", "summary", "The tree holds 0 files and took 12 seconds.", False),
    ):
        task = by_closure(tasks, closure)
        registry.get("scheduler").try_dispatch()
        stream.emit(
            EventKind.TASK_DISPATCHED,
            f"{task.closure} dispatched to agent {task.agent_spec!r}",
            task=str(task.id),
            closure=task.closure,
            agent=task.agent_spec,
        )
        runner.advance(registry, task.id)  # -> RUNNING
        hid = task.outputs[0]
        version = store.put(hid, _content(tmp_path, kind, body), producer=task.id)
        runner.produce(registry, task.id, valid=valid)
        runner.advance(registry, task.id)  # -> OUTPUT_VALIDATING

        # The verdict is written by the **validation phase**, never by the agent
        # that produced the content (`task_graph` spec §3.1). Here the phase is
        # stood in for, and `agent_id` is a real one: a `None` there is written
        # and then unreadable, which is reported to `handoff`.
        agent = agent_mgr.get(task.current.agent_id)
        store.record_verdict(
            hid,
            version,
            Verdict(
                validator="check_facts" if kind == "facts" else "check_grounded",
                result=valid,
                strength="strong",
                dimension="completeness" if kind == "facts" else "trustworthiness",
                task_id=task.id,
                agent_id=agent.id,
                environment={"source": "global"},
                at=datetime.datetime.now(datetime.timezone.utc),
            ),
        )
        runner.finish(task.id, TaskStatus.SUCCEEDED)

    tasks = task_mgr.all()
    cli_main._emit_graph(stream, tasks, resumed=False)
    cli_main._describe(registry, tasks, stream)
    cli_main._report(registry, stream, _layout(tmp_path), DEMO)
    return stream


def _layout(tmp_path):
    from cli.environment import layout_for

    return layout_for(tmp_path / "demo-root")


def _fields(stream: Stream, kind: EventKind) -> list[dict]:
    return [dict(event.fields) for event in stream.of_kind(kind)]


# --------------------------------------------------------------------------- #
# Criterion 4 — one dispatch per task, and no validator in any pool


def test_one_dispatch_per_task(run: Stream) -> None:
    """`describe` has **both** validation phases populated and is dispatched
    once. The runner carries all three phases on one lease and never returns to
    the scheduler between them, which is what makes a validation phase invisible
    rather than merely unimportant."""
    dispatched = [f["closure"] for f in _fields(run, EventKind.TASK_DISPATCHED)]
    assert dispatched == ["produce", "describe"]
    assert len(dispatched) == len(set(dispatched))


def test_no_validator_in_any_pool(registry: Any, run: Stream) -> None:
    """The other half, asserted where it would show: the scheduler's pools hold
    tasks, and every id in one resolves to a task instantiated from a closure.

    There are exactly three surfaces a validation phase could appear on —
    `runner.start`, `resource:<name>.take`, `policy.select` — and it appears on
    none of them, *because the runner never returns to the scheduler between
    phases*.
    """
    scheduler, task_mgr = registry.get("scheduler"), registry.get("task_mgr")
    closures = {"main", "produce", "describe", "consume"}
    pooled = [tid for pool in scheduler.pools.values() for tid in pool]
    assert pooled, "the run should end with `consume` still pooled"
    assert all(task_mgr.get(tid).closure in closures for tid in pooled)
    # And no validator name ever became a task: the pools are keyed by TaskId,
    # so the failure would look like a task whose closure is a validator's name.
    assert not (
        {task_mgr.get(tid).closure for tid in pooled} & set(registry.get("validator_specs").names())
    )


# --------------------------------------------------------------------------- #
# Criterion 5 — a failing verdict, and the consumer left WAITING_HANDOFF


def test_grounded_fails_and_consume_waits(registry: Any, run: Stream) -> None:
    verdicts = _fields(run, EventKind.VERDICT_RECORDED)
    failing = [v for v in verdicts if not v["result"]]
    assert [v["validator"] for v in failing] == ["check_grounded"]
    assert failing[0]["expected"] is True

    final = {f["closure"]: f for f in _fields(run, EventKind.TASK_FINAL_STATE)}
    assert final["consume"]["status"] == TaskStatus.WAITING_HANDOFF.value
    # Reported as the expected outcome, not as a crash — and it says why.
    assert final["consume"]["expected"] is True
    assert "summary is invalid" in final["consume"]["reason"]
    assert registry.get("task_mgr").get(_id(final["consume"]["task"], registry)).history == []


def _id(text: str, registry: Any):
    return next(t.id for t in registry.get("task_mgr").all() if str(t.id) == text)


def test_verdict_author_is_the_phase(run: Stream) -> None:
    """`task_graph` spec §3.1: the verdict is written by the validation phase,
    never by the agent that produced the content.

    Asserted structurally, because that is where it holds: a verdict names the
    task and the agent it was recorded *against*, and the demo's own event
    carries `phase`, so a verdict that arrived from an agent would have no phase
    to name. The mechanism itself is `validator`'s and is tested there.
    """
    for verdict in _fields(run, EventKind.VERDICT_RECORDED):
        assert verdict["phase"] == "output_validation"
        assert verdict["validator"] in {"check_facts", "check_grounded"}


def test_the_expected_failure_is_reported_as_one(run: Stream) -> None:
    observed = {f["expectation"] for f in _fields(run, EventKind.EXPECTED_FAILURE)}
    assert observed == {"grounded_verdict_fails", "consumer_waits"}
    assert run.count(EventKind.UNEXPECTED_SUCCESS) == 0
    assert _fields(run, EventKind.RUN_COMPLETE)[-1]["exit_code"] == cli_main.OK


# --------------------------------------------------------------------------- #
# §7.5 — the most important test in this directory


def test_expected_failure_that_passes_fails_the_run(stream: Stream, tmp_path) -> None:
    """An expected failure that **passes** is a FAILURE, and it exits 3.

    A demo that prints "all good" because the sandbox stopped blocking, or
    because the validator stopped failing, is the single worst outcome available
    to this artefact: it would assert, in the most visible place in the
    repository, that a safety property holds when it does not. pytest calls this
    `xfail(strict=True)`; here it is not optional.

    Driven at the accounting rather than through a run, because the thing under
    test is what the demo does when an expectation is **unmet**, and arranging
    for `check_grounded` to pass would be arranging the very thing that must not
    be arrangeable.
    """
    code = cli_main._strict(
        stream, set(), unreachable=set(), layout=_layout(tmp_path), promises=DEMO
    )
    assert code == cli_main.UNEXPECTED_SUCCESS == 3

    unexpected = _fields(stream, EventKind.UNEXPECTED_SUCCESS)
    assert {f["expectation"] for f in unexpected} == set(DEMO.promises)
    assert all(f["expected"] is True and f["observed"] is False for f in unexpected)
    assert stream.count(EventKind.EXPECTED_FAILURE) == 0
    assert _fields(stream, EventKind.RUN_COMPLETE)[-1]["ok"] is False


def test_a_half_observed_expectation_still_fails(stream: Stream, tmp_path) -> None:
    """One of two is not a pass. Stated as its own test because "some expected
    failures happened" is the shape a lenient accounting would accept."""
    assert (
        cli_main._strict(stream, {"consumer_waits"}, set(), _layout(tmp_path), DEMO)
        == cli_main.UNEXPECTED_SUCCESS
    )
    assert stream.count(EventKind.EXPECTED_FAILURE) == 1
    assert stream.count(EventKind.UNEXPECTED_SUCCESS) == 1


# --------------------------------------------------------------------------- #
# Criterion 10 — every verdict carries its dimension and strength


def test_machine_verdict_carries_taxonomy(run: Stream) -> None:
    verdicts = _fields(run, EventKind.VERDICT_RECORDED)
    assert verdicts
    for verdict in verdicts:
        assert verdict["dimension"] in {"completeness", "usability", "trustworthiness"}
        assert verdict["strength"] in {"strong", "long_term_strong", "weak"}


def test_human_verdict_line_carries_taxonomy(run: Stream) -> None:
    """Spelled out rather than abbreviated: spec §4.2 asks the demo to **teach**
    the taxonomy by using it, and `trust/S` teaches nothing."""
    lines = [line_for(event) for event in run.of_kind(EventKind.VERDICT_RECORDED)]
    assert lines
    assert any("trustworthiness / strong" in line and "FAIL" in line for line in lines)
    assert any("completeness / strong" in line and "PASS" in line for line in lines)
    assert all("verdict" in line for line in lines)


# --------------------------------------------------------------------------- #
# Criterion 14 — the machine output answers criteria 2–10 without parsing prose


def test_machine_output_answers_criteria_2_to_10(registry: Any, run: Stream) -> None:
    """One test, nine assertions, and every one reads `fields` only.

    `message` is never consulted. If any of these needed a substring of a
    sentence, criterion 14 would be false and this is where that would show.
    """
    objects = [as_object(event) for event in run.events]
    assert all(o["schema"] == SCHEMA_VERSION for o in objects)

    def of(kind: EventKind) -> list[dict]:
        return [o for o in objects if o["kind"] == kind.value]

    graph = of(EventKind.GRAPH_BUILT)[0]
    phases = of(EventKind.PHASE_START)

    # 2 — a root with `parent = None`, and subtasks with a non-None parent.
    assert graph["root_parent"] is None
    assert graph["tasks"] == 4
    assert len(graph["subtasks"]) == 3
    assert all(p["parent"] is not None for p in phases if p["closure"] != "main")

    # 3 — one input phase empty, another populated.
    produce_in = _one(phases, closure="produce", phase="input_validation")
    describe_in = _one(phases, closure="describe", phase="input_validation")
    assert produce_in["validators"] == []
    assert describe_in["validators"] == ["check_facts"]

    # 4 — one dispatch per task, and nothing dispatched that is not a task.
    dispatched = [o["closure"] for o in of(EventKind.TASK_DISPATCHED)]
    assert len(dispatched) == len(set(dispatched))
    assert set(dispatched) <= {"main", "produce", "describe", "consume"}

    # 5 — a failing verdict, and the consumer waiting.
    assert any(o["result"] is False for o in of(EventKind.VERDICT_RECORDED))
    consume = _one(of(EventKind.TASK_FINAL_STATE), closure="consume")
    assert consume["status"] == "waiting_handoff" and consume["expected"] is True

    # 6 — no fake backend: the loaded agents are the two the package declares.
    assert of(EventKind.PACKAGE_LOADED) == [] or "fake" not in str(
        of(EventKind.PACKAGE_LOADED)[0]["agents"]
    )

    # 7 — one program node and one SDK node, both producing a handoff.
    specs = registry.get("agent_specs")
    kinds = {o["closure"]: specs.spec(o["agent"]).kind.value for o in of(EventKind.TASK_DISPATCHED)}
    assert kinds == {"produce": "program", "describe": "ai"}
    assert {o["kind_"] if "kind_" in o else o["kind"] for o in of(EventKind.HANDOFF_TRANSITION)}

    # 8 — the isolation event's shape is asserted in `test_isolation_shown.py`;
    #     the demo's run does not reach it without a sandbox.

    # 9 — likewise the refusal.

    # 10 — every verdict carries its dimension and its strength.
    assert all(o["dimension"] and o["strength"] for o in of(EventKind.VERDICT_RECORDED))


def _one(objects: list[dict], **match: Any) -> dict:
    found = [o for o in objects if all(o.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise AssertionError(f"expected one event matching {match}, found {len(found)}")
    return found[0]


# --------------------------------------------------------------------------- #
# The format itself


def test_schema_version_matches_the_emitted_field(stream: Stream) -> None:
    """A constant and the thing it describes drift the moment they are two
    facts. This is the only mechanism that notices."""
    event = stream.emit(EventKind.RUN_START, "started")
    assert as_object(event)["schema"] == SCHEMA_VERSION


@pytest.mark.parametrize("kind", list(EventKind), ids=lambda k: k.value)
def test_every_event_kind_has_a_human_rendering(kind: EventKind) -> None:
    """A new kind that renders as `<Event object>` fails here rather than in
    front of a reviewer."""
    event = Event(
        kind=kind,
        message="something happened",
        fields={},
        at=datetime.datetime.now(datetime.timezone.utc),
    )
    line = line_for(event)
    assert "something happened" in line
    assert "object at 0x" not in line


def test_no_event_field_holds_a_model(run: Stream) -> None:
    """`fields` holds JSON scalars, lists and dicts — never a `Task`, an
    `Execution` or a `Handoff`. A renderer that could reach into a model would
    start rendering one."""
    for event in run.events:
        for key, value in event.fields.items():
            assert _is_json(value), f"{event.kind.value}.{key} is {type(value).__name__}"


def _is_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json(v) for k, v in value.items())
    return False


def test_the_two_renderings_come_from_one_call(stream: Stream) -> None:
    """S5, as a test: **one stream rendered twice, not two writers.**

    A demo whose narration and whose JSON disagree fails at the one job it has,
    so there is no path through `emit` that produces one without the other.
    """
    human, machine = io.StringIO(), io.StringIO()
    stream.attach(HumanRenderer(human))
    stream.attach(JsonLinesRenderer(machine))
    stream.emit(EventKind.VERDICT_RECORDED, "check_grounded: FAIL", result=False, strength="strong")

    assert human.getvalue().count("\n") == machine.getvalue().count("\n") == 1
    assert "check_grounded: FAIL" in human.getvalue()
    assert json.loads(machine.getvalue())["message"] == "check_grounded: FAIL"
    assert json.loads(machine.getvalue())["result"] is False


def test_the_json_lines_survive_a_truncated_run(stream: Stream) -> None:
    """Criterion 12 needs this: a run killed with `os._exit` must still have
    emitted valid, complete output for everything before the interrupt. A single
    top-level JSON document would be truncated and unparseable, which would make
    the resume demonstration unassertable."""
    out = io.StringIO()
    stream.attach(JsonLinesRenderer(out))
    stream.emit(EventKind.RUN_START, "one")
    stream.emit(EventKind.TASK_DISPATCHED, "two")
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["message"] for line in lines] == ["one", "two"]


def test_reserved_keys_cannot_be_overwritten_by_a_field(stream: Stream) -> None:
    """`schema`, `kind`, `at` and `message` always mean what they say, because
    they are written after the flatten. A field colliding with one would
    otherwise win silently — and `kind` is a word this system uses for handoff
    kinds too, so the collision is not hypothetical."""
    event = stream.emit(EventKind.RUN_START, "hello", kind="facts", schema="9.9")
    obj = as_object(event)
    assert obj["kind"] == "run_start" and obj["schema"] == SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# `final` must say why, not only what


def test_a_failed_task_reports_its_reason(registry: Any, submitted: Any) -> None:
    """A `final` line naming a status without the reason answers *what* and not
    *why*, and this artefact is where that costs the most: the demo is what a
    reviewer runs, and every wall this stage was found here and then diagnosed by
    somebody reading a different package's source.

    **Measured before it was built.** The reason IS recorded — `monitor`'s
    `Recorder` holds a `HANDLING_FAILED` record — while `Execution.detail`, the
    field whose comment says *"from the runner; for a human"*, is empty because
    `agent`'s `_crash` does not pass the `detail` `on_task_done` accepts.
    Reported there; read from the recorder here, because that is the copy that
    exists.
    """
    runner, task_mgr = registry.get("runner"), registry.get("task_mgr")
    runner.advance(registry, submitted.id)  # unfold
    produce = by_closure(task_mgr.all(), "produce")

    registry.get("scheduler").try_dispatch()
    runner.advance(registry, produce.id)  # -> RUNNING
    registry.get("recorder").open(produce.id, produce.current.attempt)
    registry.get("recorder").write(
        monitor_event(
            MonitorEventKind.HANDLING_FAILED,
            produce.id,
            attempt=produce.current.attempt,
            reported_by="agent.Runner",
            exception_type="UnresolvedGrant",
            exception_message="grant for kind 'facts' matches no handoff",
        )
    )
    runner.finish(produce.id, TaskStatus.FAILED)

    assert produce.status is TaskStatus.FAILED
    assert produce.current.detail == ""  # `FakeRunner` passes none; the real one does
    reason, source = cli_main._why_failed(produce, registry)
    assert "UnresolvedGrant" in reason
    assert "matches no handoff" in reason
    # **And it says which copy answered.** The recorder read was reached on every
    # failure until `agent` landed `detail`, which made it the primary path
    # wearing a fallback's name — and a fallback that is silently load-bearing is
    # the same class of defect as the one this whole line exists to fix.
    assert source == "recorder"


def test_the_reason_is_a_field_and_not_only_prose(registry: Any, submitted: Any, tmp_path) -> None:
    """Criterion 14 reads `fields`, never `message`. A reason a reviewer can see
    and a parser cannot is half the fix."""
    stream = Stream()
    cli_main._report(registry, stream, _layout(tmp_path), DEMO)
    final = [dict(e.fields) for e in stream.of_kind(EventKind.TASK_FINAL_STATE)]
    assert final
    assert all("reason" in f for f in final)
    # And which copy answered, so a reappearance of the fallback is assertable
    # rather than only visible.
    assert all("reason_source" in f for f in final)


def test_detail_wins_over_the_recorder_when_it_is_populated(registry: Any) -> None:
    """`Execution.detail` first, so the fallback stops being reached the day
    `agent` passes it — and nothing here needs changing when they do."""
    from task_graph import Task

    task = Task(agent_spec="collect")
    registry.get("task_mgr").add(task)
    task.push_execution(agent_id=registry.get("agent_mgr").instantiate("collect", task.id).id)
    task.current.detail = "the runner said so"
    assert cli_main._why_failed(task, registry) == ("the runner said so", "execution.detail")


def test_an_unreached_expectation_is_untested_and_not_a_broken_property(
    stream: Stream, tmp_path
) -> None:
    """**The third outcome, and it is the one this accounting got wrong.**

    `main` ran the demo in a state where `produce` failed before `describe`
    could produce a summary, so `check_grounded` never ran — and the run
    reported *"did NOT happen … a property stopped holding"*, which was true
    about what was observed and false about what it meant.

    A promise the run never reached is **untested**, not broken. It is not green
    either: exit 4, *an unexpected failure*, because something else stopped the
    run before its promises could be tested. Exit 3 is reserved for the claim it
    actually makes — a property that stopped holding — and crying wolf about one
    is the mirror of the failure this artefact exists to prevent.
    """
    code = cli_main._strict(
        stream, set(), unreachable=set(DEMO.promises), layout=_layout(tmp_path), promises=DEMO
    )
    assert code == cli_main.UNEXPECTED_FAILURE == 4

    assert stream.count(EventKind.UNEXPECTED_SUCCESS) == 0
    assert stream.count(EventKind.EXPECTATION_UNREACHED) == len(DEMO.promises)
    for fields in _fields(stream, EventKind.EXPECTATION_UNREACHED):
        assert fields["reachable"] is False and fields["observed"] is False
    done = _fields(stream, EventKind.RUN_COMPLETE)[-1]
    assert done["unreached"] == sorted(DEMO.promises)
    assert done["unobserved"] == []
    assert done["ok"] is False


def test_a_published_summary_with_no_verdict_is_not_judged(
    registry: Any, submitted: Any, tmp_path
) -> None:
    """**A summary existing is not `check_grounded` having judged it**, and the
    old predicate could not tell those apart.

    It asked *does a summary version exist in the store*, which since §4.14 is
    true from the moment `_seal_outputs` publishes — before the gate, well
    before `OUTPUT_VALIDATING`. Measured on a real run: `check_grounded` crashed
    with a `KeyError` and recorded nothing, a summary existed, and the run
    reported *"did NOT happen … a property stopped holding"* and exited 3.
    **`UNTESTED` wearing `UNEXPECTED`'s label**, from the artefact whose purpose
    is telling those apart.

    So this publishes a summary and records **no** verdict. The old
    implementation returns `True` here; the correct one returns `False`, and the
    run reports `EXPECTATION_UNREACHED` rather than claiming a lost property.
    """
    runner, store = registry.get("runner"), registry.get("handoff_store")
    runner.advance(registry, submitted.id)
    task = by_closure(registry.get("task_mgr").all(), "describe")

    assert not DEMO.promises["grounded_verdict_fails"].was_judged(registry)
    store.put(
        task.outputs[0],
        _content(tmp_path, "summary", "The tree holds 0 files."),
        producer=task.id,
    )
    # Published, readable, and nothing has judged it.
    assert store.list_versions(task.outputs[0])
    assert not DEMO.promises["grounded_verdict_fails"].was_judged(registry)


def test_consumer_reachability_is_asked_of_the_graph(registry: Any, submitted: Any) -> None:
    """`consume` cannot be expected to wait before the root has grown it.

    Deliberately still the weaker *does the subject exist* shape — see
    `_consumer_exists`, which records why and that it is reported rather than
    fixed here."""
    assert not DEMO.promises["consumer_waits"].was_judged(registry)
    registry.get("runner").advance(registry, submitted.id)
    assert DEMO.promises["consumer_waits"].was_judged(registry)


def test_awaiting_a_decision_is_not_reported_as_a_stall(registry: Any, submitted: Any) -> None:
    """**A resting state that looks identical to a hang is not a resting state.**

    `monitor` measured the end state after fixing F-D16: a handled gate failure
    legitimately leaves the task `running` — criterion 4 says the gate cycle must
    not move task status — and once the escalation reaches the root, *what the
    alpha does at the top of an escalation chain* is their spec §11, open.
    `NullUserSink` records the arrival and does nothing, deliberately. So the
    task sits in `running` **as specified**.

    This package's stall detection ends the run by timing out on absence of
    change, which is a heuristic and **cannot tell waiting-for-a-human from
    deadlocked**. It does not have to: the escalation that reaches the top is a
    record carrying `target: "user"`, and the demo holds a `Recorder`.
    """
    runner, task_mgr = registry.get("runner"), registry.get("task_mgr")
    runner.advance(registry, submitted.id)
    produce = by_closure(task_mgr.all(), "produce")
    registry.get("scheduler").try_dispatch()
    runner.advance(registry, produce.id)  # -> RUNNING, and it stays there

    recorder = registry.get("recorder")
    recorder.open(produce.id, produce.current.attempt)
    # An escalation that has NOT reached the top reads as an ordinary failure.
    recorder.write(
        monitor_event(
            MonitorEventKind.ESCALATED,
            produce.id,
            attempt=produce.current.attempt,
            reported_by="default",
            attributes={"why": "output_absent: nothing to push"},
        )
    )
    assert cli_main._awaiting_a_decision(produce, registry) == ""

    # The one that reached the root carries `target: "user"`.
    recorder.write(
        monitor_event(
            MonitorEventKind.ESCALATED,
            produce.id,
            attempt=produce.current.attempt,
            reported_by="default",
            attributes={"target": "user", "why": "output_absent: nothing to push"},
        )
    )
    assert "nothing to push" in cli_main._awaiting_a_decision(produce, registry)

    # And it is `monitor.reached_the_user` that is asked, not two strings this
    # package read out of their design document. The keys are theirs and are
    # exported; a rename on their side is now an import error here rather than a
    # silent False that would report a resting state as a hang.
    from monitor import ESCALATION_TARGET, TARGET_USER, reached_the_user

    assert (ESCALATION_TARGET, TARGET_USER) == ("target", "user")
    assert reached_the_user(recorder.read(produce.id, produce.current.attempt)[-1])

    stream = Stream()
    cli_main._report(registry, stream, _layout(pathlib.Path(tempfile.mkdtemp())), DEMO)
    final = {f["closure"]: f for f in _fields(stream, EventKind.TASK_FINAL_STATE)}
    assert final["produce"]["reason_source"] == "escalation"
    assert (
        "awaiting a decision"
        in [
            e.message for e in stream.of_kind(EventKind.TASK_FINAL_STATE) if "produce" in e.message
        ][0]
    )


def test_a_parked_escalated_task_does_not_hold_the_run_open(registry: Any, submitted: Any) -> None:
    """**"Holding a thread" is not "making progress".**

    Measured in a live run: after a gate failure `TaskAttempt._main` parks on
    `_await_wake()`, so the thread is alive and unhalted and `is_running` stays
    True for ever. `_settle` required `not holding` before it would call a stall,
    so the branch never fired and only the 300 s timeout ended the run — with the
    escalation records already 20 s old and nothing left to happen.

    A task awaiting a decision is **parked, not working**, and this package
    already knew how to tell those apart; it just had not wired the distinction
    into the exit condition, only into the reporting. One genuinely mid-model-call
    still counts as holding, which is what stops a 20 s window calling a slow
    backend a stall.
    """

    class _Parked:
        """The real runner's shape: **one** parked leaf, and a released non-leaf.

        Faithful on purpose. A first version returned `is_running=True` for every
        task, which made `main` hold the run open and the test fail for a reason
        the real run does not have — a stub encoding an assumption instead of the
        neighbour's behaviour, `interfaces.md` §8.1's fifth row. A non-leaf calls
        `release()`, which sets `_halted`, so `is_running` is False for it.
        """

        def __init__(self, parked: Any) -> None:
            self._parked = parked

        def attempt_of(self, tid: Any) -> Any:
            return SimpleNamespace(is_running=tid == self._parked)

    runner, task_mgr = registry.get("runner"), registry.get("task_mgr")
    runner.advance(registry, submitted.id)
    produce = by_closure(task_mgr.all(), "produce")
    registry.get("scheduler").try_dispatch()
    runner.advance(registry, produce.id)  # -> RUNNING, and it stays

    recorder = registry.get("recorder")
    recorder.open(produce.id, produce.current.attempt)
    recorder.write(
        monitor_event(
            MonitorEventKind.ESCALATED,
            produce.id,
            attempt=produce.current.attempt,
            reported_by="default",
            attributes={"target": "user", "why": "output_absent: nothing to push"},
        )
    )
    registry.register("runner", _Parked(produce.id))

    stream = Stream()
    started = time.monotonic()
    cli_main._settle(registry, stream, timeout=30.0, period=5.0, stall_after=0.2)
    # Ends on the stall, not on the timeout.
    assert time.monotonic() - started < 10.0
    done = _fields(stream, EventKind.RUN_COMPLETE)[-1]
    assert done["settled"] is False
    assert any("produce" in s for s in done["stalled_tasks"])
