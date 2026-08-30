"""Stubs, one per Protocol the runner resolves.

Wave 1's one unbendable rule: **import the Protocol, never a sibling's in-flight
implementation.** Where a test needs a neighbour's behaviour it satisfies the
Protocol here, in `agent`'s own tests — `docs/implementation-stage.md` §4.1.

Nothing here makes a model call, opens a socket, or needs a credential.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.backend import AgentResult, AgentStatus, Assignment, ExecutorBase
from agent.registry import AgentSpecRegistry
from task_graph.ids import HandoffId
from task_graph.models import Agent, Task, TaskStatus
from task_graph.registry import Registry
from task_graph.task import TaskMgr

# --------------------------------------------------------------------------- #
# Executors


class ScriptedBackend(ExecutorBase):
    """An `AgentBackend` whose whole harness is a list of scripted results.

    Level 2, so it is what the `Executor`-versus-`AgentBackend` tests contrast
    `ProgramExecutor` against.
    """

    def __init__(
        self,
        key: str = "scripted",
        config: dict[str, Any] | None = None,
        assignment: Assignment | None = None,
    ) -> None:
        super().__init__(key, assignment)
        self.config = dict(config or {})
        if self.config.get("unavailable"):
            from agent.backend import BackendUnsupported

            raise BackendUnsupported(key, "run here", str(self.config["unavailable"]))
        self.deployed = 0
        self.delivered: list[str] = []
        self.interrupted = 0
        self.results: list[AgentResult] = list(self.config.get("results") or [])

    def _deploy(self) -> None:
        self.deployed += 1

    def _run(self) -> AgentResult:
        if self.results:
            return self.results.pop(0)
        return AgentResult(status=AgentStatus.FINISHED, usage={"tokens": 1.0}, detail="ok")

    def _deliver(self, message: str) -> None:
        self.delivered.append(message)

    def interrupt(self) -> None:
        self.interrupted += 1
        self.status = AgentStatus.INTERRUPTED

    def instruct(self, message: str) -> None:
        self._enqueue_instruction(message)

    def query(self) -> Any:
        from agent.backend import history_of

        return history_of([{"role": "assistant"}], "session-1")


class BrokenBackend(ExecutorBase):
    """An adapter that declared a method it does not implement — design §5.3.

    It inherits `_run`, whose base raises `BackendUnsupported` naming the
    adapter. That is what criterion 4 is about, and it is *not* what a program
    executor is: a program has no level 2 and therefore no method to raise from.
    """

    def __init__(
        self,
        key: str = "broken",
        config: dict[str, Any] | None = None,
        assignment: Assignment | None = None,
    ) -> None:
        super().__init__(key, assignment)
        self.config = dict(config or {})

    def _deploy(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Neighbours, by Protocol


@dataclass
class StubEvidence:
    value: str


@dataclass
class StubOutcome:
    kind: str
    passed: bool = True
    empty: bool = False
    evidence: StubEvidence | None = None
    ran: tuple[Any, ...] = ()
    reused: tuple[Any, ...] = ()
    skipped: tuple[Any, ...] = ()

    @property
    def blocks_the_task(self) -> bool:
        """`validator`'s, computed the same way: **only a real failure blocks.**

        Not a stub constant — `not empty and not passed` is the rule, and a stub
        that hard-coded `False` would pass every test here while the real one
        blocked, which is this week's recurring failure.
        """
        return not self.empty and not self.passed


class StubPhaseRunner:
    """Satisfies `validator.protocols.PhaseRunner`."""

    def __init__(
        self, *, passes: bool = True, empty: bool = False, evidence: str | None = "established"
    ) -> None:
        self.passes = passes
        self.empty = empty
        self.evidence = evidence
        self.calls: list[tuple[Any, Any]] = []

    def run_phase(self, kind: Any, task: Any, registry: Any) -> StubOutcome:
        # `validator` coerces a bare string at the boundary — `PhaseKind(kind)`
        # — after finding five `is` comparisons that would have taken the wrong
        # branch. The stub records what it was handed so the seam stays visible.
        self.calls.append((kind, task.id))
        return StubOutcome(
            kind=str(kind),
            passed=self.passes,
            empty=self.empty,
            evidence=StubEvidence(self.evidence) if self.evidence else None,
        )


class StubMonitor:
    """Satisfies `monitor.protocols.Monitor`'s inbound surface.

    **It advances the phase and wakes the attempt**, because that is what
    `monitor` spec §5.3 made a monitor action: the attempt reports and waits,
    and `task.enter_phase(next)` is the monitor's call, never the runner's.
    """

    name = "stub"
    last_beat = 0.0

    def __init__(self, runner: Any = None) -> None:
        self.runner = runner
        self.records: list[Any] = []
        self.watching: list[Any] = []
        self.advance = True
        self._lock = threading.Lock()

    def set_task(self, task_id: Any) -> None:
        """Idempotent, as the real one is — `resume` takes a second thread and
        calls this again."""
        if task_id not in self.watching:
            self.watching.append(task_id)

    def report(self, record: Any) -> None:
        """**Enforces the scope guard**, because a stub that accepts anything
        proves nothing about `set_task` being called.

        The real `_run_guarded` raises `ScopeViolation` for a task it was never
        given, and `demo` F-D8 was exactly that firing on the first planned
        advance. A permissive stub is how the gap survived eight packages'
        unit suites.
        """
        if record.task_id not in self.watching:
            raise AssertionError(
                f"reported {record.kind} for {record.task_id}, which set_task never named"
            )
        with self._lock:
            self.records.append(record)
        if not self.advance or self.runner is None:
            return
        if record.kind.value != "phase_done":
            return
        attempt = self.runner.attempt_of(record.task_id)
        if attempt is None:
            return
        task = attempt.task
        nxt = _next_phase(task.status)
        if nxt is not None:
            task.enter_phase(nxt)
            attempt.wake()
        else:
            attempt.wake()

    def kinds(self) -> list[str]:
        return [r.kind.value for r in self.records]

    def mainloop(self) -> None:  # pragma: no cover — the stub is driven by report
        return None

    def stop(self) -> None:
        return None


def _next_phase(status: TaskStatus) -> TaskStatus | None:
    order = [TaskStatus.INPUT_VALIDATING, TaskStatus.RUNNING, TaskStatus.OUTPUT_VALIDATING]
    if status not in order:
        return None
    index = order.index(status)
    return order[index + 1] if index + 1 < len(order) else None


@dataclass
class StubManifest:
    """**No `items`, because the real one has none.**

    It had one for weeks, and `tests/agent/test_gate_against_the_real_store.py`
    is what found it: `handoff.protocols.Manifest` is `digest` / `algorithm` /
    `kind` / `producer` / `created_at`, so the gate's pre-check read a field
    that never existed and `OUTPUT_NOT_EXECUTABLE` was unreachable. Keeping the
    field here would let the same shape come back.
    """

    done_by_self_check: Any = None


@dataclass
class StubContent:
    """What `copy_out` returns — `handoff.content.Content`'s `items`."""

    items: dict[str, Any] = field(default_factory=dict)


class StubStore:
    """Satisfies the part of `handoff.protocols.HandoffStore` the gate uses.

    It models the real store's **refusals**, not only its successes: `copy_out`
    creates its destination and raises on one that exists, and a stub that
    ignored `dst` hid a real crash in the gate.
    """

    def __init__(self, present: dict[HandoffId, StubManifest] | None = None) -> None:
        self.present = dict(present or {})
        self.contents: dict[HandoffId, Any] = {}
        #: What the body "wrote", by handoff id. `seal` publishes only these,
        #: which is the real store's rule: `content/` exists because `allocate`
        #: created it, so **emptiness** is what says the attempt produced
        #: nothing. Tests put an id here to mean "the body wrote something".
        self.written: set[HandoffId] = set()
        self.sealed: list[tuple[HandoffId, int, Any]] = []

    def exists(self, hid: HandoffId, version: int | None = None) -> bool:
        return hid in self.present

    def list_versions(self, hid: HandoffId) -> list[int]:
        return [0] if hid in self.present else []

    def get_manifest(self, hid: HandoffId, version: int) -> StubManifest:
        return self.present[hid]

    def copy_out(self, hid: HandoffId, version: int, dst: Path) -> Any:
        if Path(dst).exists():
            raise AssertionError(f"{dst} already exists; copy_out creates its destination")
        Path(dst).mkdir(parents=True)
        return self.contents.get(hid, StubContent())

    def seal(self, hid: HandoffId, version: int, *, producer: Any) -> str | None:
        """Publish, or refuse — and **the refusal is the half that matters**.

        This method was missing when the runner started calling it, and the
        runner's `except Exception` turned every `AttributeError` into a
        "refusal": 169 tests stayed green with the seal never running once.
        Same failure as the `FakeClient` and `Prepared` doubles on 2026-08-29,
        and `test_doubles_conform.py` now checks this store's surface for it.

        Refuses with the real store's distinction (`handoff/store.py:380`): an
        empty `content/` is *the agent wrote nothing*, which is `monitor`
        criterion 5's "never attempted" and is not the same as malformed.

        **Returns the refusal, and raising it was this double's second version
        of the same mistake.** `handoff` moved the boundary in `fd31a6c` — a
        refusal is a return value and only `NotSealable` escapes — because
        `agent` may not import `handoff` and so cannot name a type to catch. A
        double that still raises would keep the runner's deleted `except` alive
        in the tests after the code stopped having one.
        """
        if hid not in self.written:
            return (
                f"cannot seal {hid} v{version}: nothing was written to content/. "
                f"That directory is the agent's grant and it is empty, so this "
                f"attempt produced no content at all"
            )
        self.sealed.append((hid, version, producer))
        self.present.setdefault(hid, StubManifest())
        return None


class StubRecorder:
    """Satisfies the part of `monitor.protocols.Recorder` the attempt calls.

    **Absence is a signal**, which is why the runner now requires it rather than
    skipping: a marker present with no occurrences reads as "nothing was
    recorded here", and a marker absent reads as "something is wrong".
    """

    def __init__(self) -> None:
        self.opened: list[tuple[Any, int]] = []

    def open(self, task_id: Any, attempt: int) -> None:
        self.opened.append((task_id, attempt))

    def write(self, record: Any) -> None:
        return None

    def read(self, task_id: Any, attempt: int) -> list[Any]:
        return []

    def is_open(self, task_id: Any, attempt: int) -> bool:
        return (task_id, attempt) in self.opened


class StubEnvManager:
    """Satisfies `env_mgr.protocols.EnvManager`. See `agent/README.md` on the
    two-versus-three argument disagreement."""

    def __init__(self, zone_root: str = "/tmp/zone") -> None:
        self.zone_root = zone_root
        self.calls: list[Any] = []
        self.executions: list[Any] = []
        self.placed: list[Any] = []

    def place_zone(self, task: Any, execution: Any) -> Any:
        """`prepare`'s first step and none of the rest — `env_mgr` `6fa6a6e`."""
        self.placed.append((task.id, execution))
        return type("Zone", (), {"root": self.zone_root})()

    def prepare(self, task: Any, execution: Any, agent_spec: Any = None) -> Any:
        """**All seven fields**, because the real `Prepared` has all seven.

        It used to return `zone` alone, and a probe over the suite counted the
        runner's `getattr` defaults being taken 34 times for `confinement` and
        17 for `environment` — **every one of them from this stub and none from
        production**. The defaults existed to paper over a thin double, and the
        `confinement` one meant a dropped field would have started a task
        unconfined instead of raising. `env_mgr.protocols.Prepared` is
        `zone` / `workspace` / `policy` / `confinement` / `sync` / `environment`
        / `agent_cli`.

        **`agent_cli` was the seventh, and its absence here took eleven tests
        red the moment the runner read it** (`runner.py:617`). That is the same
        failure this docstring already describes, one field later and from the
        other direction: a double that has drifted from the object it stands
        in for. It is `None` rather than a path because these tests do not
        exercise the real backend's refusal — `test_claude_sdk.py` does.
        """
        self.calls.append((task.id, agent_spec))
        self.executions.append(execution)
        return SimpleNamespace(
            zone=SimpleNamespace(root=self.zone_root),
            workspace=None,
            policy=None,
            confinement=None,
            sync=None,
            environment={},
            agent_cli=None,
            staged_package=None,
            permissions_enforced=True,
            output_paths={},
        )


class StubSpecs:
    """Satisfies the part of `spec_loader.protocols.SpecRegistry` that a `Task`
    transition and the runner reach.

    Stands in for both `closures` (which `Task.has_subgraph` reads) and
    `task_specs` (which the runner reads) — `closure/check.py` keys the nested
    task spec under the closure's own name, so in the real system one name
    indexes both and the stub mirrors that."""

    def __init__(self, docs: dict[str, Any] | None = None) -> None:
        self._docs = dict(docs or {})

    def get(self, name: str) -> Any:
        return self._docs[name]

    def names(self) -> list[str]:
        return sorted(self._docs)

    def __contains__(self, name: str) -> bool:
        return name in self._docs


class StubScheduler:
    """`Task.enter_phase` reaches the scheduler through `_move`. This is the
    whole of what the phase transitions need, and no more."""

    def __init__(self) -> None:
        self.tasks: dict[Any, Task] = {}
        self.submitted: list[Task] = []

    def track(self, task: Task) -> None:
        self.tasks[task.id] = task

    def _move(self, task_id: Any, status: TaskStatus) -> None:
        self.tasks[task_id].status = status

    def submit(self, task: Task) -> None:
        self.submitted.append(task)
        self.track(task)


# --------------------------------------------------------------------------- #
# Fixtures


def ai_spec(name: str = "writer", **over: Any) -> dict[str, Any]:
    record = {
        "name": name,
        "version": "1",
        "description": "writes",
        "kind": "ai",
        "backends": [{"key": "scripted", "backend_entry": "tests.agent.conftest:ScriptedBackend"}],
    }
    record.update(over)
    return record


def program_spec(name: str = "runner", **over: Any) -> dict[str, Any]:
    record = {"name": name, "version": "1", "description": "runs", "kind": "program"}
    record.update(over)
    return record


@pytest.fixture()
def specs() -> AgentSpecRegistry:
    registry = AgentSpecRegistry()
    registry.add("writer", ai_spec(), origin="tests")
    registry.add("runner", program_spec(), origin="tests")
    return registry


@pytest.fixture()
def wired(specs: AgentSpecRegistry, tmp_path: Path):
    """A registry with every component the runner resolves, and a task."""
    from agent.runner import Runner

    entry = tmp_path / "entry.sh"
    entry.write_text("#!/bin/sh\nexit 0\n")
    entry.chmod(0o755)
    # **A real file, because `readme` is a path and the runner now reads it.**
    # The schema always said so (`_common.schema.json`); until `4d43017` the
    # runner passed the declared string through as the agent's prompt, so a
    # fixture could say `"R"` and nothing noticed.
    readme = tmp_path / "readme.md"
    readme.write_text("Do the thing.\n")

    specs_by_closure = StubSpecs(
        {
            "leaf": {
                "agent": "runner",
                "task": {"goal": "do it", "body": {"readme": str(readme), "entry": str(entry)}},
            },
            "leaf_ai": {
                "agent": "writer",
                "task": {"goal": "do it", "body": {"readme": str(readme)}},
            },
            # A subgraph entry is `{closure, is_start?, is_end?}` and the
            # closure must be declared — `Task.unfold` raises otherwise, before
            # the runner's main phase is reached at all. The first version of
            # this fixture named a closure that did not exist, and the
            # `TaskStateError` looked like a runner fault.
            "branch": {
                "agent": "writer",
                "task": {
                    "goal": "expand",
                    "body": {"readme": str(readme)},
                    "subgraph": [{"closure": "leaf_ai"}],
                },
            },
        }
    )

    r = Registry()
    scheduler = StubScheduler()
    r.register("scheduler", scheduler)
    # **The real `TaskMgr`, not a stub.** `Task.has_subgraph` resolves it
    # (`task_graph/models.py:473`) since `task_graph` made `unfold` idempotent
    # for `--resume` (`cc23f98`) — before that, a non-leaf's declaration was the
    # whole answer and this harness needed nothing. `task_graph` rejected a
    # tolerant lookup with the reason worth repeating: *a harness with no task
    # graph is not the same fact as a task with no children*, so the fixture
    # supplies the component rather than the runner shrugging at its absence.
    r.register("task_mgr", TaskMgr(r))
    runner = Runner(r)
    monitor = StubMonitor(runner)
    phase_runner = StubPhaseRunner()
    r.register("agent_specs", specs)
    r.register("closures", specs_by_closure)
    r.register("task_specs", StubSpecs({k: v["task"] for k, v in specs_by_closure._docs.items()}))
    r.register("phase_runner", phase_runner)
    r.register("env_mgr", StubEnvManager(str(tmp_path)))
    r.register("handoff_store", StubStore())
    r.register("monitor:stub", monitor)
    recorder = StubRecorder()
    r.register("recorder", recorder)
    r.register("runner", runner)
    return type(
        "Wired",
        (),
        {
            "registry": r,
            "runner": runner,
            "monitor": monitor,
            "phase_runner": phase_runner,
            "scheduler": scheduler,
            "recorder": recorder,
            "entry": entry,
            "tmp_path": tmp_path,
        },
    )()


def dispatched(agent_spec: str, closure: str, wired: Any = None) -> tuple[Task, Agent]:
    """A task in the state the scheduler leaves it in, and **everything the
    scheduler does on the way**.

    `INPUT_VALIDATING`, one open `Execution`, its registry supplied — which is
    what `TaskMgr` does and what `enter_phase` needs — and `Monitor.set_task`,
    which `Scheduler._dispatch_pass` calls immediately before `runner.start`
    (`interfaces.md` §2.1 rev. 5).

    **The `set_task` line is here because it moved out of the runner**, and
    modelling it matters: `StubMonitor.report` enforces the scope guard, so a
    harness that skipped it would fail every test — which is how this fixture
    caught the move rather than papering over it.
    """
    task = Task(agent_spec=agent_spec, closure=closure, monitor_spec="stub")
    agent = Agent(spec=agent_spec, task_id=task.id)
    task.push_execution(agent.id)
    task.status = TaskStatus.INPUT_VALIDATING
    if wired is not None:
        task._registry = wired.registry
        wired.scheduler.track(task)
        wired.runner.monitor_for(task).set_task(task.id)
    return task, agent


# --------------------------------------------------------------------------- #
# Structural assertions, over the AST rather than the text
#
# `tests/interfaces/test_import_rules.py` gives the reason and it applies here
# unchanged: a substring check fails for the wrong reason the moment a docstring
# mentions the word. "retry" appears in a sentence explaining that the runner
# does not retry.


def _tree(*objects: Any) -> list[ast.AST]:
    return [ast.parse(textwrap.dedent(inspect.getsource(obj))) for obj in objects]


def called_attributes(*objects: Any) -> set[str]:
    """Every `x.name(...)` these objects call."""
    found: set[str] = set()
    for tree in _tree(*objects):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                found.add(node.func.attr)
    return found


def assigned_attributes(*objects: Any) -> set[str]:
    """Every `x.name = ...` these objects perform."""
    found: set[str] = set()
    for tree in _tree(*objects):
        for node in ast.walk(tree):
            targets = getattr(node, "targets", []) or (
                [node.target] if hasattr(node, "target") else []
            )
            found |= {t.attr for t in targets if isinstance(t, ast.Attribute)}
    return found
