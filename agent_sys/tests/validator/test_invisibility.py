"""Criterion 5 — a validation phase is invisible to the scheduler.

Three devices, all three from `tests/task_graph/test_authority.py`:
subclass-and-log rather than stack inspection, assert set membership rather than
a call count, and a third test that **plants the erosion** so the spy is known to
be able to fail.

There are exactly three surfaces to spy, verified against the real scheduler:
`runner.start` (`scheduler.py`), `resource:<name>.take`, and `policy.select`. A
validation phase adds to none of them, **because the runner never returns to the
scheduler between phases** — that structural fact is what makes the criterion
assertable rather than aspirational.

Two things this test has to get right, both measured:

*`select` fires more often than there are dispatches* — four passes for two tasks,
two of them with an empty eligible list. So the assertion is *"no validator name
ever appears in a `select` argument"*, not a call count.

*"Pool" means the **resource** pool.* `Scheduler.pools` comprehends over the whole
`TaskStatus` enum, so `INPUT_VALIDATING` and `OUTPUT_VALIDATING` create two index
pools **by construction** — a test asserting "no validator occupies a pool" over
`Scheduler.pools` would fail on a correct implementation. §15 O1 records that the
criterion's wording should be tightened; the real assertion is one lease taken
once and held across all three phases.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from task_graph.bootstrap import build_registry
from task_graph.ids import HandoffId
from task_graph.models import Task, TaskStatus
from task_graph.policy import FifoPolicy
from task_graph.resource import GpuMgr
from task_graph.runner import FakeRunner
from tests.validator.conftest import (
    CLOSURE,
    DictSpecRegistry,
    MemoryHandoffStore,
    StubClosureRegistry,
    bind_kind,
    validator_record,
    write_body,
)
from validator.phase import PhaseRunner
from validator.protocols import PhaseKind, StrictLevel
from validator.registry import ValidatorSpecRegistry

ROOT = Path(__file__).resolve().parents[2]


class SpyPolicy(FifoPolicy):
    """Subclass-and-log. Records every candidate it was ever offered."""

    def __init__(self) -> None:
        super().__init__()
        self.offered: list[str] = []
        self.passes = 0

    def select(self, eligible, snapshot):
        self.passes += 1
        self.offered += [str(t.id) for t in eligible]
        self.offered += [t.agent_spec for t in eligible]
        return super().select(eligible, snapshot)


class SpyGpu(GpuMgr):
    def __init__(self, registry, capacity: float) -> None:
        super().__init__(registry, capacity=capacity)
        self.takes: list[float] = []

    def take(self, amount: float) -> None:
        self.takes.append(amount)
        super().take(amount)


class PhaseDrivingRunner(FakeRunner):
    """The runner that does what `agent.Runner` will: three phases, one dispatch.

    `start` runs input validation, then the main work, then output validation,
    and **never returns to the scheduler in between**. That is the property under
    test, so it is the thing the fake has to reproduce faithfully.
    """

    def __init__(self, phase: PhaseRunner, registry_ref: dict) -> None:
        super().__init__()
        self.phase = phase
        self.registry_ref = registry_ref
        self.phases_run: list[PhaseKind] = []

    def start(self, task, agent, on_done) -> None:
        super().start(task, agent, on_done)
        registry = self.registry_ref["r"]
        for kind in (PhaseKind.INPUT, PhaseKind.OUTPUT):
            self.phase.run_phase(kind, task, registry)
            self.phases_run.append(kind)


@pytest.fixture
def spied(tmp_path: Path) -> Any:
    ref: dict = {}
    policy = SpyPolicy()
    phase = PhaseRunner(
        StrictLevel.DEFAULT,
        zone_root=tmp_path / "zones",
        package_root=write_body(tmp_path / "pkg"),
    )
    runner = PhaseDrivingRunner(phase, ref)
    r = build_registry(runner=runner, policy=policy, resources=[])
    r.register("resource:gpu", SpyGpu(r, capacity=8))
    r.register("handoff_store", MemoryHandoffStore())
    r.register("handoff_specs", DictSpecRegistry("handoff"))
    r.register("validator_specs", ValidatorSpecRegistry())
    r.register("closures", StubClosureRegistry())
    r.get("agent_mgr").register("producer")
    ref["r"] = r
    return r, policy, runner


def run_two_tasks(registry) -> list[Task]:
    tasks = []
    for _ in range(2):
        hid = HandoffId.new()
        task = Task(
            agent_spec="producer",
            outputs=[hid],
            kinds={hid: "trace"},
            resources={"gpu": 1},
            closure=CLOSURE,
        )
        registry.get("scheduler").submit(task)
        tasks.append(task)
    return tasks


def test_one_dispatch_for_three_phases(spied) -> None:
    """Criterion 5. The scheduler dispatches **one** task and gets one completion.

    And the real assertion behind "no validator occupies a pool": `task_graph`
    criterion 40 — one lease, taken once, held across all three phases.
    """
    registry, _policy, runner = spied
    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])

    tasks = run_two_tasks(registry)

    assert runner.started == [t.id for t in tasks]  # two dispatches, two tasks
    assert runner.phases_run.count(PhaseKind.INPUT) == 2
    assert runner.phases_run.count(PhaseKind.OUTPUT) == 2
    assert registry.get("resource:gpu").takes == [1, 1]  # one lease each, not three
    # The phases genuinely ran: the output phase recorded a verdict per task, so
    # this is not a launch the scheduler aborted and the spy then saw nothing in.
    store = registry.get("handoff_store")
    assert [len(store.read_verdicts(t.outputs[0], 0)) for t in tasks] == [1, 1]
    # Still live and still holding its lease: the scheduler dispatched into
    # INPUT_VALIDATING and the runner has not reported back.
    assert all(t.status not in (TaskStatus.SUCCEEDED, TaskStatus.FAILED) for t in tasks)
    assert registry.get("resource:gpu").available == 6


def test_no_validator_reaches_the_policy(spied) -> None:
    """Criterion 5, as set membership rather than a call count.

    Measured: `select` fires more often than there are dispatches — extra passes
    with an empty eligible list — so a count assertion would be asserting the
    scheduler's pass structure rather than the property.
    """
    registry, policy, _runner = spied
    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])

    run_two_tasks(registry)

    assert policy.passes >= 2  # it fires more often than there are dispatches
    assert "shape" not in policy.offered
    assert not [name for name in policy.offered if name in registry.get("validator_specs").names()]


def test_the_spy_would_catch_a_dispatch(spied) -> None:
    """**Plant the erosion**, so the spy is known to be able to fail.

    Without this, a spy that observes nothing is indistinguishable from a spy that
    is wired to nothing.
    """
    registry, policy, _runner = spied
    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])

    # A validator submitted as a graph node — the thing criterion 5 forbids.
    registry.get("agent_mgr").register("shape")
    registry.get("scheduler").submit(
        Task(agent_spec="shape", resources={"gpu": 1}, closure=CLOSURE)
    )

    assert "shape" in policy.offered


def test_the_two_validating_statuses_are_index_pools_by_construction(spied) -> None:
    """§15 O1, asserted rather than left as a note.

    `Scheduler.pools` comprehends over the whole `TaskStatus` enum, so the literal
    reading of *"no validator occupies a pool"* is unsatisfiable: the two pools
    exist whether or not anything is in them. A task sitting in the
    `OUTPUT_VALIDATING` index is **correct**.
    """
    registry, _policy, _runner = spied
    pools = registry.get("scheduler").pools
    assert TaskStatus.INPUT_VALIDATING in pools
    assert TaskStatus.OUTPUT_VALIDATING in pools
    assert set(pools) == set(TaskStatus)


def test_the_runner_never_names_the_scheduler() -> None:
    """The static half, and it must walk the **AST**.

    Measured on the shipped tree: `"scheduler" in runner.py` source text is `True`
    — docstring mentions — while an AST walk over names, attributes and imports
    returns 0. `test_authority.py`'s existing static check *is* a substring grep,
    so copying it naively produces a test that fails for the wrong reason. Applied
    here to `validator/phase.py`, which is this module's runner-side surface.
    """
    source = (ROOT / "validator/phase.py").read_text()
    assert "scheduler" in source  # the prose mentions it; that is not the check

    tree = ast.parse(source)
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    resolved = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "scheduler" not in referenced
    assert not [m for m in imported if "scheduler" in m]
    assert "scheduler" not in resolved  # not even resolved by name from the registry
    assert "policy" not in resolved
