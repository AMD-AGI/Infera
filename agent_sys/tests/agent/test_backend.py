"""Criteria 4, 7, 8, 12 and 14 — design §12.

**Four of these five are structural**, and the tests say so: they assert over
the protocol's signature rather than over behaviour. A structural criterion
tested behaviourally passes for the wrong reason as soon as someone adds the
field.
"""

from __future__ import annotations

import ast
import inspect
import threading
import time
from pathlib import Path

import pytest

from agent.backend import (
    TERMINAL,
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendUnsupported,
    Executor,
)
from agent.backends.program import ProgramExecutor
from task_graph.models import TaskStatus

from .conftest import BrokenBackend, ScriptedBackend

CONFIGURATION_VERBS = (
    "configure",
    "reload",
    "set_rules",
    "set_hooks",
    "set_model",
    "set_permission_mode",
)
SUBAGENT_WORDS = ("subagent", "sub_agent", "child_agent")


# --------------------------------------------------------------------------- #
# Criterion 4


def test_unsupported_method_raises() -> None:
    """Spec §3.3.1: no capability matrix, no per-capability degradation. **The
    error names the adapter that is incomplete, not the interface.**"""
    broken = BrokenBackend()
    broken.start_async(lambda: None)
    with pytest.raises(BackendUnsupported) as caught:
        broken.mainloop()
    assert caught.value.key == "broken"
    assert "broken" in str(caught.value)


def test_a_program_executor_has_nothing_to_raise_from() -> None:
    """Criterion 4's other half, and design §5.3's warning: **this is about an
    incomplete adapter, not about an executor that legitimately has no level
    2.** A program is not an incomplete AI harness."""
    program = ProgramExecutor(assignment=_entry())
    for verb in ("interrupt", "instruct", "query"):
        assert not hasattr(program, verb)


# --------------------------------------------------------------------------- #
# Criterion 7


def test_start_async_returns_before_started() -> None:
    """`start_async` returns immediately and its callback fires when the agent
    *really* starts — deploying takes long enough that "started" and "asked to
    start" are different events, and the difference is what a monitor sees."""
    backend = ScriptedBackend()
    fired: list[AgentStatus] = []
    backend.start_async(lambda: fired.append(backend.status))

    assert backend.status is AgentStatus.PENDING
    assert fired == []

    backend.mainloop()
    assert fired == [AgentStatus.RUNNING]


def test_start_equals_async_plus_wait() -> None:
    """`start()` is sugar over the asynchronous form, and the two paths produce
    the same result — spec §4.3."""
    direct = ScriptedBackend().start()

    threaded = ScriptedBackend()
    threaded.start_async(lambda: None)
    driver = threading.Thread(target=threaded.mainloop)
    driver.start()
    through_wait = threaded.wait()
    driver.join(5)

    assert direct.status is through_wait.status is AgentStatus.FINISHED
    assert direct.usage == through_wait.usage


def test_every_synchronous_verb_is_sugar_this_level_wraps() -> None:
    """Design §5.1.1, as **one rule rather than a per-method convention**: an
    adapter implements the asynchronous form and inherits the rest, so it
    cannot ship a `stop()` that blocks differently from every other
    backend's."""
    inherited = {"start", "wait", "stop", "start_async", "mainloop"}
    assert inherited.isdisjoint(vars(ScriptedBackend))
    assert inherited.isdisjoint(vars(ProgramExecutor))


def test_stop_settles_a_running_agent() -> None:
    program = ProgramExecutor(config={"command": ["/bin/sleep", "30"]})
    program.start_async(lambda: None)
    driver = threading.Thread(target=program.mainloop)
    driver.start()
    deadline = time.monotonic() + 5
    while program.status is not AgentStatus.RUNNING and time.monotonic() < deadline:
        time.sleep(0.005)
    program.stop()
    driver.join(5)
    assert program.status in TERMINAL


def test_a_loud_body_does_not_hang_the_executor() -> None:
    """**The pipe had no reader**, so a body that filled it blocked in `write()`
    and `_run` polled a child that could not exit — measured at 256 KiB in
    `scratch/impl-2026-08/agent/probe_program_output_is_lost.py`, where
    `start()` never returned. A quiet body was fine, which is why every test
    here passed over it.

    Well over Linux's 64 KiB pipe buffer, and the assertion is that this test
    finishes at all.
    """
    loud = "dd if=/dev/zero bs=1024 count=256 2>/dev/null | tr '\\0' 'x'"
    program = ProgramExecutor(config={"command": ["/bin/sh", "-c", loud]})
    done: list[AgentResult] = []
    driver = threading.Thread(target=lambda: done.append(program.start()), daemon=True)
    driver.start()
    driver.join(20)

    assert not driver.is_alive(), "the executor hung on a body that filled the pipe"
    assert done[0].status is AgentStatus.FINISHED


def test_a_failed_body_says_what_it_said_before_it_stopped() -> None:
    """`exit 1` alone ends the investigation where it starts — `demo` measured
    an hour lost to a body whose `KeyError` reached no reader."""
    script = "echo on-stdout; echo the-traceback >&2; exit 3"
    program = ProgramExecutor(config={"command": ["/bin/sh", "-c", script]})

    result = program.start()

    assert result.status is AgentStatus.FAILED
    assert result.detail.startswith("exit 3: ")
    assert "the-traceback" in result.detail


# --------------------------------------------------------------------------- #
# Criterion 8


def test_status_sequence() -> None:
    """`pending → deploying → running → finished`."""
    seen: list[AgentStatus] = []

    class Watching(ScriptedBackend):
        def _deploy(self) -> None:
            seen.append(self.status)

        def _run(self) -> AgentResult:
            seen.append(self.status)
            return AgentResult(status=AgentStatus.FINISHED)

    backend = Watching()
    seen.append(backend.status)
    backend.start()
    seen.append(backend.status)
    assert seen == [
        AgentStatus.PENDING,
        AgentStatus.DEPLOYING,
        AgentStatus.RUNNING,
        AgentStatus.FINISHED,
    ]


def test_task_status_is_superset() -> None:
    """`Task.status` is a superset of the stack-top agent's: every agent status
    has a task status that covers it, and the task adds the states that exist
    when no agent is bound — waiting, cancelled — and the phase states."""
    covering = {
        AgentStatus.PENDING: TaskStatus.WAITING_RESOURCE,
        AgentStatus.DEPLOYING: TaskStatus.INPUT_VALIDATING,
        AgentStatus.RUNNING: TaskStatus.RUNNING,
        AgentStatus.FINISHED: TaskStatus.SUCCEEDED,
        AgentStatus.FAILED: TaskStatus.FAILED,
        AgentStatus.INTERRUPTED: TaskStatus.STOPPING,
    }
    assert set(covering) == set(AgentStatus)
    unreached = set(TaskStatus) - set(covering.values())
    assert TaskStatus.WAITING_HANDOFF in unreached
    assert TaskStatus.CANCELLED in unreached
    assert TaskStatus.OUTPUT_VALIDATING in unreached


# --------------------------------------------------------------------------- #
# Criteria 12 and 14, both structural


def test_no_interface_reaches_a_subagent() -> None:
    """Criterion 12. **Only the main backend agent may be interacted with** —
    at any point, however visible a subagent is. Non-trivial rather than
    automatic: the SDK emits subagent tool blocks into our stream by default
    and subagents inherit the parent's permission mode."""
    for protocol in (Executor, AgentBackend):
        for name in _members(protocol):
            signature = inspect.signature(getattr(protocol, name))
            for parameter in signature.parameters:
                assert not any(word in parameter.lower() for word in SUBAGENT_WORDS)
    for name in dir(ScriptedBackend):
        assert not any(word in name.lower() for word in SUBAGENT_WORDS)


def test_backend_has_no_configuration_method() -> None:
    """Criterion 14. **An agent spec arrives fully prepared**, and there is no
    runtime interface for changing it. Satisfied structurally, like criterion 5:
    there is nowhere to put one.

    `set_permission_mode()` and `set_model()` exist on `ClaudeSDKClient` and
    are deliberately unused — adding a passthrough would create the interface
    §4.4 says must not exist (design §5.4).
    """
    for protocol in (Executor, AgentBackend):
        assert not set(_members(protocol)) & set(CONFIGURATION_VERBS)
    for adapter in (ScriptedBackend, ProgramExecutor):
        assert not {v for v in CONFIGURATION_VERBS if hasattr(adapter, v)}


# --------------------------------------------------------------------------- #
# The split itself — design D5


def test_the_two_levels_are_two_protocols() -> None:
    assert set(_members(AgentBackend)) - set(_members(Executor)) == {
        "interrupt",
        "instruct",
        "query",
    }
    assert isinstance(ScriptedBackend(), AgentBackend)
    assert isinstance(ProgramExecutor(assignment=_entry()), Executor)
    assert not isinstance(ProgramExecutor(assignment=_entry()), AgentBackend)


def _members(protocol: type) -> set[str]:
    """The protocol's own callable members.

    Not `__protocol_attrs__`, which is 3.12+, and the floor here is 3.10.
    `status` is an annotation rather than an entry in `vars`, so it drops out
    without being named.
    """
    found: set[str] = set()
    for klass in protocol.__mro__:
        if klass.__name__ in {"object", "Protocol", "Generic"}:
            continue
        found |= {n for n, v in vars(klass).items() if not n.startswith("_") and callable(v)}
    return found


def _entry():
    from agent.backend import Assignment

    return Assignment(entry="/bin/true")


def test_backend_py_imports_nothing_of_ours() -> None:
    """`docs/implementation-stage.md` §4.2 and design §2: **`backend.py` imports
    nothing of ours**, which is what lets `selection`, `backends/` and `runner`
    all depend on it without a cycle. It is the one file in the package that
    could be written before anything else existed."""
    ours = {
        "spec_loader",
        "handoff",
        "validator",
        "agent",
        "closure",
        "env_mgr",
        "task_graph",
        "monitor",
        "demo",
    }
    tree = ast.parse((Path(__file__).resolve().parents[2] / "agent" / "backend.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & ours


def test_instruct_refuses_once_the_loop_has_returned() -> None:
    """**A push into a dead queue is a silent deadlock, and it cost 65 minutes.**

    Measured 2026-08-31 on the first end-to-end run of
    `examples/single_real_task`: the agent finished, the seal was refused, the
    monitor decided to push *continue, do it until finished*, `PUSH_ATTEMPTED`
    was written to the store — and the message was never read. `mainloop` had
    already broken out on terminal status (`backend.py:410-413`), and
    `queue.Queue` is unbounded, so `_enqueue_instruction`'s `put` succeeded into
    a queue with no consumer. The run then sat for 65 minutes with both
    processes alive, ~1 s CPU per 20 s, and nothing written anywhere.

    **The defect is the silence, not the failed push.** A push that arrives
    after the agent has settled is an ordinary race the monitor cannot win from
    outside. Reporting it as attempted and then waiting for ever for an answer
    is not.

    The store made this indistinguishable: it holds `push_attempted` and no
    counterpart — no `push_delivered`, no `push_failed` — so the event reads the
    same whether the message landed or vanished. Raising gives it one, because
    `monitor.base._run_guarded` catches and records `handling_failed`.

    Probe: `scratch/single-real-task-2026-08/probe_push_after_settle.py`.
    """
    from agent.backend import AgentNotListening

    backend = ScriptedBackend()
    backend.start()  # runs mainloop to settlement, as the runner does
    assert backend.status in TERMINAL
    assert backend.delivered == []

    with pytest.raises(AgentNotListening) as caught:
        backend.instruct("continue, do it until finished")

    # The message names the agent, its status, and what to do instead — a
    # deadlock's replacement has to be readable or it is just a different hang.
    text = str(caught.value)
    assert "scripted" in text and "finished" in text and "start_async" in text
    # And it is not quietly queued behind the refusal.
    assert backend._inbox.empty()
    assert backend.delivered == []


def test_instruct_is_still_delivered_while_the_loop_runs() -> None:
    """**The non-vacuity control for the test above.**

    A refusal that fires on every `instruct` would pass that test and break the
    feature — the monitor could never push at all, which is worse than the
    defect it replaces. So the same call, made while a loop is running, must
    still reach `_deliver`.

    Driven on a second thread because `mainloop` is the consumer and this
    thread is the producer; that is the shipped arrangement — `runner` drives
    the loop and a monitor instructs from its own.
    """
    backend = ScriptedBackend(config={"results": [AgentResult(status=AgentStatus.RUNNING)]})
    backend.start_async(lambda: None)
    driver = threading.Thread(target=backend.mainloop, daemon=True)
    driver.start()

    deadline = time.monotonic() + 5.0
    while not backend._looping and time.monotonic() < deadline:
        time.sleep(0.01)
    assert backend._looping, "the loop never started; the control proves nothing"

    backend.instruct("keep going")

    while backend.delivered != ["keep going"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert backend.delivered == ["keep going"], "a live loop must still deliver"

    backend.stop()
    driver.join(timeout=5.0)


def test_instruct_before_the_loop_starts_is_still_queued() -> None:
    """**The second control, and the suite found it before I did.**

    A first version of the refusal keyed on `_looping`, and
    `test_claude_sdk.py::test_instruct_does_not_end_run` went red: the shipped
    order `start_async()` → `instruct()` → `mainloop()` — queue the work, then
    lend the loop a thread — has no loop turning at the moment of the call, and
    the message is nonetheless about to be consumed.

    **"Has not started yet" and "has already finished" are different states**,
    and only the second is the defect. Pinned here rather than left to the
    claude_sdk test, because that one is about the SDK adapter and this is about
    the predicate.
    """
    backend = ScriptedBackend(config={"results": [AgentResult(status=AgentStatus.FINISHED)]})
    backend.start_async(lambda: None)
    assert not backend._looping, "the premise: no loop is turning yet"

    backend.instruct("queued before the loop")  # must NOT raise
    backend.mainloop()

    assert backend.delivered == ["queued before the loop"]
