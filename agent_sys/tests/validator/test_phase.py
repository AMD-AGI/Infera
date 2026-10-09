"""Criteria 6 and 17 — the phase runs, and a failed one is an ordinary failure.

Every test here builds its own `Registry` through `bootstrap.build_registry` with
a `MemoryStoreMgr` and the shipped `FakeRunner`; nothing is process-global. The
handoff store and the two spec registries are the stubs `conftest.py` declares,
because `handoff` and `spec_loader` are being written in the same wave.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from task_graph.models import TaskStatus
from tests.validator.conftest import (
    RecordingExecutor,
    bind_kind,
    validator_record,
    write_body,
)
from validator.phase import PhaseRunner, read_verdict_file
from validator.protocols import PhaseKind, StrictLevel, ValidatorInvalid
from validator.report import Evidence


def runner(zone_root: Path, package_root: Path, level: StrictLevel = StrictLevel.DEFAULT):
    return PhaseRunner(level, zone_root=zone_root, package_root=package_root)


def register(registry: Any, name: str, **kw) -> None:
    registry.get("validator_specs").add(
        name, validator_record(name, **kw), origin=f"{name}.jsonnet"
    )


def test_phase_produces_no_handoff(registry, dispatched, zone_root, package_root) -> None:
    """Criterion 6. A validation phase *"produces no output handoff"* — it records
    a verdict beside the artefact instead."""
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    before = set(registry.get("handoff_mgr").all_ids())

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.passed is True
    assert set(registry.get("handoff_mgr").all_ids()) == before


def test_a_bare_string_kind_takes_the_right_branch(
    registry, dispatched, zone_root, package_root
) -> None:
    """`agent.Runner` cannot name `PhaseKind` and passes the **value**.

    `interfaces.md` §4.4 gives `agent` only `spec_loader` / `task_graph` /
    `monitor`, so there is no import through which the enum member could reach
    the one caller this seam has. Everything inside compares with `is`, which a
    bare string fails — and the first thing it fails is which list of handoffs
    the phase reads, so an input phase would silently validate the task's
    *outputs*. Raised by `agent-mod` as F3; `run_phase` coerces at the boundary.

    Asserted as a **difference between the two phases**, not as "it did not
    crash": the string form has to reach the same branch the enum does.
    """
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    phase = runner(zone_root, package_root, StrictLevel.STRICT)

    by_enum = phase.run_phase(PhaseKind.OUTPUT, dispatched, registry)
    by_value = phase.run_phase("output_validation", dispatched, registry)
    assert by_value.kind is PhaseKind.OUTPUT
    assert [r.verdict.validator for r in by_value.ran] == [r.verdict.validator for r in by_enum.ran]

    # The input phase reads `task.inputs`, which is empty here — so if the string
    # form fell through to the output branch this would be a pass, not empty.
    assert phase.run_phase("input_validation", dispatched, registry).empty is True

    with pytest.raises(ValueError, match="not a valid PhaseKind"):
        phase.run_phase("main", dispatched, registry)


def test_verdict_does_not_move_digest(registry, dispatched, zone_root, package_root) -> None:
    """Criterion 6. The record is a sibling of the content and outside the digest,
    so recording a verdict does not change the artefact's identity.

    `handoff` owns the digest; what this side owns is calling `record_verdict` and
    touching nothing on the version. Asserted as *the version is unchanged*, which
    is the half this module can be responsible for.
    """
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    hid = dispatched.outputs[0]
    before = registry.get("handoff_mgr").get(hid).latest.model_dump()

    runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert registry.get("handoff_mgr").get(hid).latest.model_dump() == before
    assert len(registry.get("handoff_store").read_verdicts(hid, 0)) == 1


def test_a_closure_phase_validator_runs(registry, dispatched, zone_root, package_root) -> None:
    """The defect `demo` found on the first assembly of all eight.

    `closure.schema.json` calls these *"the PHASE validators… a property of the
    task rather than of any one handoff kind, **which is why the handoff specs
    cannot carry them**"* — and this module used to build its set from the handoff
    kinds, so a closure declaring `validators: ['check_grounded']` ran **nothing**.

    Declared on the closure and on **no handoff kind**, which is the case that
    silently did nothing before.
    """
    register(registry, "grounded", inputs=["trace"])
    registry.get("closures").declare(dispatched.closure, ["grounded"])

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert [r.verdict.validator for r in outcome.ran] == ["grounded"]
    assert outcome.passed is True


def test_the_set_is_asked_for_not_joined_here(
    registry, dispatched, zone_root, package_root
) -> None:
    """`engineer_principle` §4.4: ask for the computation, not the raw material.

    `closure` already joins the phase validators with the per-handoff ones —
    *"every validator that will run"* — so this module asks for that set and does
    not read `handoff_specs` at all. Asserted as behaviour: a kind naming a
    validator the closure's set omits does **not** run it, because the closure is
    the single answer rather than one of two inputs.
    """
    register(registry, "kind_only", inputs=["trace"])
    registry.get("handoff_specs").add(
        "trace", {"name": "trace", "validators": ["kind_only"]}, origin="<trace>"
    )
    registry.get("closures").declare(dispatched.closure, [])  # the closure says none

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert outcome.empty is True
    assert outcome.passed is False  # and empty is not a pass


def test_a_task_with_no_closure_runs_nothing_and_does_not_pass(
    registry, dispatched, zone_root, package_root
) -> None:
    """A task built outside a closure has no declared set.

    The phase folds to `empty`, which is **reported and not a pass**, so nothing
    is silently admitted. That is the third state doing its job rather than a
    gap — but it is worth an assertion, because "runs nothing" and "found nothing"
    are the two readings this system exists to keep apart.
    """
    dispatched.closure = None
    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert outcome.empty is True and outcome.passed is False


def test_a_phase_with_no_bound_validators_is_empty_not_a_pass(
    registry, dispatched, zone_root, package_root
) -> None:
    """§11.2, at the phase. Nothing is bound, so nothing ran, so it is not green.

    **And since `interfaces.md` §4.15 it is more than not-green.** This task has
    an output and nothing is bound to check it, so verdicts *were* expected and
    none arrived — the fault, and it blocks.
    """
    registry.get("closures").declare(dispatched.closure, [])
    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert outcome.empty is True and outcome.passed is False
    assert outcome.verdicts_expected is True
    assert outcome.evidence is Evidence.UNCHECKED
    assert outcome.blocks_the_task is True


def test_the_same_phase_under_none_is_expected_rather_than_a_fault(
    registry, dispatched, zone_root, package_root
) -> None:
    """§4.15's first row, against the identical task. The **only** difference from
    the test above is the level, and the two empties come apart:
    `StrictLevel.NONE` means no validation was asked for.

    This is also criterion 20 measured rather than asserted — the level moved a
    phase from *ran* to *did not run*, and moved no verdict, because there is no
    verdict in either outcome to move.
    """
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    off = PhaseRunner(StrictLevel.NONE, zone_root=zone_root, package_root=package_root)
    outcome = off.run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.empty is True and outcome.passed is False  # still not a pass
    assert outcome.verdicts_expected is False
    assert outcome.evidence is Evidence.NOTHING_RAN
    assert outcome.blocks_the_task is False
    # Criterion 7 still holds: the skip is reported, whatever caused it.
    assert [s.validator for s in outcome.skipped] == ["shape"]


def test_a_task_with_no_output_has_nothing_unchecked(
    registry, dispatched, zone_root, package_root
) -> None:
    """The narrow reading of §4.15, and the reason it is narrow.

    §4.15's sentence is *nothing checked what this task **produced***. A task
    that produced nothing has nothing unchecked, so this is not the fault — and
    the wide reading is not a theoretical difference: `examples/demo/closures/`
    gives `main` (`outputs: []`, `validators: []`) and `consume` (`outputs: []`,
    one validator declared), so it would block the demo's **root** task and end
    the run before it started.

    `consume` is why the question is answered independently of what is bound: it
    declares a validator *and* has no output.
    """
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    dispatched.outputs = []
    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.empty is True and outcome.passed is False
    assert outcome.verdicts_expected is False
    assert outcome.blocks_the_task is False


def test_failed_phase_is_an_ordinary_failure(registry, dispatched, zone_root, tmp_path) -> None:
    """Criterion 17. The phase returns a `PhaseOutcome`; the *runner* turns a
    failing one into a task failure. **Nothing in this module cascades,
    invalidates or notifies** — the scheduler's response to a failure is to stop
    scheduling, and everything past that belongs to the monitor.
    """
    package_root = write_body(tmp_path / "failing", verdict=False)
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert outcome.passed is False

    # What the runner does with it, and it is the ordinary path.
    registry.get("runner").finish(dispatched.id, TaskStatus.FAILED)
    assert dispatched.status is TaskStatus.FAILED
    assert registry.get("scheduler").pools[TaskStatus.FAILED] == {dispatched.id}


def test_nothing_downstream_is_cancelled(registry, dispatched, zone_root, tmp_path) -> None:
    """Criterion 17. A consumer stays `WAITING_HANDOFF` because no output became
    valid; nothing downstream is cancelled or invalidated *by the scheduler*.

    Cascading invalidation is a **policy** about how to react to a failure, and
    putting policy in the scheduler would give it an opinion about *what*.
    """
    from task_graph.ids import HandoffId
    from task_graph.models import Task

    package_root = write_body(tmp_path / "failing", verdict=False)
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])

    # A second producer whose output never becomes valid, so the criterion is
    # about the *consumer standing still* rather than about a handoff that was
    # already usable before the phase ran.
    hid = HandoffId.new()
    producer = Task(agent_spec="producer", outputs=[hid], kinds={hid: "trace"})
    registry.get("scheduler").submit(producer)
    registry.get("runner").produce(registry, producer.id, valid=False)

    consumer = Task(agent_spec="producer", inputs=[hid], depends_on=[producer.id])
    registry.get("scheduler").submit(consumer)

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, producer, registry)
    assert outcome.passed is False
    registry.get("runner").finish(producer.id, TaskStatus.FAILED)

    assert producer.status is TaskStatus.FAILED
    assert consumer.status is TaskStatus.WAITING_HANDOFF
    assert consumer.id not in registry.get("scheduler").pools[TaskStatus.CANCELLED]


def test_the_phase_resolves_its_collaborators_by_name(
    registry, dispatched, zone_root, package_root
) -> None:
    """§2.1. `phase` is the only module that touches a manager, and it resolves
    every one of them from the component `Registry` **at call time, never by
    import**.

    `handoff_specs` is deliberately **not** among them any more: the phase asks
    `closures` for the whole validator set rather than joining the kinds' lists
    itself, and doing that join twice was the defect `demo` found.
    """
    import ast

    source = Path(__file__).resolve().parents[2] / "validator/phase.py"
    literals = {
        node.value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"handoff_mgr", "validator_specs", "handoff_store", "closures"} <= literals
    assert "handoff_specs" not in literals

    imported = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "env_mgr" not in imported and "agent" not in imported and "closure" not in imported


def test_cheap_gates_run_before_expensive_ones(
    registry, dispatched, zone_root, package_root
) -> None:
    """§5.3, spec §2 principle 2. Ordering by a declared `cost` tag has **no prior
    art** in anything surveyed, so the failure mode is owed rather than a
    citation: the tag can be wrong and nothing here detects it."""
    for name, cost in (("bench", "gpu_hours"), ("shape", "seconds"), ("load", "minutes")):
        register(registry, name, inputs=["trace"], cost=cost)
    bind_kind(registry, "trace", ["bench", "load", "shape"])

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert [r.verdict.validator for r in outcome.ran] == ["shape", "load", "bench"]


def test_a_second_run_reuses_the_recorded_verdict(
    registry, dispatched, zone_root, package_root
) -> None:
    """§7. The record answers *did this exact validator run against this exact
    version, and what did it say* — which is what a cache was going to be built to
    answer, without needing a key."""
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    phase = runner(zone_root, package_root)

    first = phase.run_phase(PhaseKind.OUTPUT, dispatched, registry)
    second = phase.run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert len(first.ran) == 1 and not first.reused
    assert not second.ran and len(second.reused) == 1
    assert second.skipped[0].validator == "shape"
    assert second.passed is True


def test_strict_never_reuses(registry, dispatched, zone_root, package_root) -> None:
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    again = runner(zone_root, package_root, StrictLevel.STRICT).run_phase(
        PhaseKind.OUTPUT, dispatched, registry
    )
    assert len(again.ran) == 1 and not again.reused


def test_none_switches_the_phase_off_and_that_is_not_a_pass(
    registry, dispatched, zone_root, package_root
) -> None:
    """Criterion 20 at the phase: the knob decides which phases *run*, and a
    switched-off phase folds to `empty` — so it can never turn a failing phase
    green."""
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])

    outcome = runner(zone_root, package_root, StrictLevel.NONE).run_phase(
        PhaseKind.OUTPUT, dispatched, registry
    )
    assert outcome.empty is True and outcome.passed is False
    assert outcome.skipped[0].reason.endswith("--validation-strict-level")
    assert registry.get("handoff_store").read_verdicts(dispatched.outputs[0], 0) == []


def test_args_reach_the_body_as_a_file(registry, dispatched, zone_root, tmp_path) -> None:
    """§10.6. `args.json` in the zone, readable by a script and referable by a
    readme. **Parameters would have worked for a callable and not for an agent**,
    which is the same reason the callable went away."""
    package_root = tmp_path / "argsy"
    write_body(package_root)
    (package_root / "entry.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
        "import json, pathlib\n"
        "zone = pathlib.Path.cwd()\n"
        "args = json.loads((zone / 'args.json').read_text())\n"
        "ids = json.loads((zone / 'inputs.json').read_text())\n"
        "(zone / 'verdict.json').write_text(json.dumps({i: args['limit_ms'] == 10 for i in ids}))\n"
        "PY\n"
    )
    record = validator_record("threshold", inputs=["trace"])
    record["args"] = {"limit_ms": 10}
    registry.get("validator_specs").add("threshold", record, origin="t.jsonnet")
    bind_kind(registry, "trace", ["threshold"])

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert outcome.passed is True


def test_agent_bodied_and_script_bodied_validators_are_substitutable(
    registry, dispatched, zone_root, package_root
) -> None:
    """§3.8. One verdict file, two ways of producing it — **the property the
    callable could not have.** A callable cannot express a validator an agent is
    responsible for without a wrapper whose whole job is to run an agent."""
    registry.register("validator_executor", RecordingExecutor(result=True))
    register(registry, "scripted", inputs=["trace"], entry="entry.sh")
    register(registry, "judged", inputs=["trace"], entry=None)
    bind_kind(registry, "trace", ["judged", "scripted"])

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.passed is True
    assert {r.verdict.validator for r in outcome.ran} == {"scripted", "judged"}
    assert registry.get("validator_executor").calls == ["judged"]


def test_a_script_body_without_a_package_root_is_refused(registry, dispatched, zone_root) -> None:
    """`package_root` had defaulted to `Path.cwd()`, which is not a default.

    A body path is **package-relative**, so resolving it against the working
    directory resolves it against wherever the process happened to start — it
    finds either nothing, with a puzzling message, or a different file of the
    same name. `interfaces.md` §4.11's family once more: a value that is wrong
    but not type-wrong, so nothing raises.

    It stays optional because §4.3 declares `PhaseRunner(strict_level)`, so the
    absence is refused at the point a body actually needs resolving. An
    agent-bodied validator is unaffected — the executor is handed the spec, not a
    resolved path — and that asymmetry is asserted here rather than assumed.
    """
    register(registry, "shape", inputs=["trace"], entry="entry.sh")
    bind_kind(registry, "trace", ["shape"])
    bare = PhaseRunner(StrictLevel.DEFAULT, zone_root=zone_root)  # no package_root

    with pytest.raises(ValidatorInvalid, match="no package root"):
        bare.run_phase(PhaseKind.OUTPUT, dispatched, registry)


def test_an_agent_body_needs_no_package_root(registry, dispatched, zone_root) -> None:
    """The other half of the asymmetry, asserted rather than assumed: the
    executor is handed the spec, not a resolved path, so an agent-bodied
    validator runs with no package root at all."""
    registry.register("validator_executor", RecordingExecutor(result=True))
    register(registry, "judged", inputs=["trace"], entry=None)
    bind_kind(registry, "trace", ["judged"])
    bare = PhaseRunner(StrictLevel.DEFAULT, zone_root=zone_root)  # no package_root

    assert bare.run_phase(PhaseKind.OUTPUT, dispatched, registry).passed is True


def test_the_verdict_names_the_checking_agent_not_the_producer(
    registry, dispatched, zone_root, package_root
) -> None:
    """Criterion 10's attribution leg, and until `agent`'s executor existed this
    module got it **backwards**.

    Every verdict carried `task.current.agent_id` — *the producing agent* — so the
    persisted record said the producer's own agent validated the artefact, which
    is the exact claim §8.1 forbids. The executor mints a fresh **unbound** agent
    per phase and returns its id; `env` is frozen and built before the body runs,
    so the return value is the only route by which it can reach a verdict.
    """
    executor = RecordingExecutor(result=True)
    registry.register("validator_executor", executor)
    register(registry, "judged", inputs=["trace"], entry=None)
    bind_kind(registry, "trace", ["judged"])

    runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    (verdict,) = registry.get("handoff_store").read_verdicts(dispatched.outputs[0], 0)
    assert verdict.agent_id == executor.agents[0]
    assert verdict.agent_id != dispatched.current.agent_id  # not the producer's


def test_a_script_verdict_records_no_agent(registry, dispatched, zone_root, package_root) -> None:
    """A script body has no agent, and the verdict says so in the field itself.

    This carried the **producer's** id until `handoff` widened
    `Verdict.agent_id` to `AgentId | None` (f9142aa) — a record asserting that
    the producer validated its own artefact, which is the claim §8.1 forbids.

    `None` rather than a sentinel: a sentinel `AgentId` is a UUID, so a reader
    who does not know it takes it for a real agent and one who looks it up in
    `agent_mgr` finds nothing. That is a plausible value flowing on undetected,
    in the one field whose entire purpose is attribution.
    """
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])

    runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    (verdict,) = registry.get("handoff_store").read_verdicts(dispatched.outputs[0], 0)
    assert verdict.agent_id is None
    assert verdict.agent_id != dispatched.current.agent_id  # never the producer's


def test_an_agent_bodied_validator_without_an_executor_fails_loudly(
    registry, dispatched, zone_root, package_root
) -> None:
    """The mechanism is `agent` design O6's and is open, so this names it rather
    than assuming one."""
    register(registry, "judged", inputs=["trace"], entry=None)
    bind_kind(registry, "trace", ["judged"])
    with pytest.raises(ValidatorInvalid, match="agent design O6"):
        runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)


def test_a_body_that_exits_zero_and_reports_nothing_is_an_error(
    registry, dispatched, zone_root, tmp_path
) -> None:
    """A body that reports nothing must not pass. JUnit XML makes pass the
    structural default and that is the shape to design against."""
    package_root = tmp_path / "silent"
    package_root.mkdir()
    (package_root / "readme.md").write_text("# x")
    (package_root / "entry.sh").write_text("#!/bin/sh\nexit 0\n")
    register(registry, "silent", inputs=["trace"])
    bind_kind(registry, "trace", ["silent"])

    with pytest.raises(ValidatorInvalid, match="wrote no verdict.json"):
        runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)


def test_a_crashed_body_reaches_no_verdict_rather_than_failing(
    registry, dispatched, zone_root, tmp_path
) -> None:
    """**A body that reports nothing must not pass — and must not fail either.**

    Only the first half was here. A nonzero exit with no verdict file had
    `{hid: False}` fabricated for it, and a fabricated `False` is byte-identical
    to a considered one — so a segfaulting validator reported *the validator
    worked and the answer is no*.

    That is the flattening `monitor` spec §2.1 exists to prevent:
    `VALIDATION_FAILED` says a branch is **judged** dead, `VALIDATION_UNREACHED`
    says it is **undetermined**, and their analysing dispatcher's whole job is
    telling those apart. A crash is not a judgement.

    Found because `monitor` asked whether a *returned* outcome could mean "no
    verdict reachable". I had told them it could not. It could.
    """
    package_root = tmp_path / "broken"
    package_root.mkdir()
    (package_root / "readme.md").write_text("# x")
    (package_root / "entry.sh").write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    register(registry, "broken", inputs=["trace"])
    bind_kind(registry, "trace", ["broken"])

    with pytest.raises(ValidatorInvalid, match="nothing was decided") as exc:
        runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert "exited 3" in str(exc.value)  # the only thing that tells a human why
    assert "boom" in str(exc.value)

    # And nothing is persisted: a crash records no verdict against the artefact.
    assert registry.get("handoff_store").read_verdicts(dispatched.outputs[0], 0) == []


def test_a_crashed_body_reports_the_tail_of_its_stderr_not_the_head(
    registry, dispatched, zone_root, tmp_path
) -> None:
    """**The end of a traceback is the part worth keeping**, and the message used
    to keep the other one.

    `[:200]` took the head, which for a Python body is
    `Traceback (most recent call last):` plus the outermost frames — the file,
    the line and the exception type all live at the tail. `demo` measured the
    consequence against a real body: the recorded message was cut mid-path inside
    the zone, before the first frame, and named none of the three, so the only
    reason anyone knew what had failed was that they had captured the child's
    stderr themselves.
    """
    package_root = tmp_path / "verbose"
    package_root.mkdir()
    (package_root / "readme.md").write_text("# x")
    (package_root / "entry.sh").write_text(
        "#!/bin/sh\n"
        "i=0; while [ $i -lt 400 ]; do echo 'a filler frame' >&2; i=$((i+1)); done\n"
        "echo 'KeyError: AGENT_SYS_DEMO_STORE' >&2\n"
        "exit 1\n"
    )
    register(registry, "verbose", inputs=["trace"])
    bind_kind(registry, "trace", ["verbose"])

    with pytest.raises(ValidatorInvalid) as exc:
        runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    message = str(exc.value)
    assert "KeyError: AGENT_SYS_DEMO_STORE" in message  # the last line survives
    assert "…" in message  # and the message says it was cut
    assert len(message) < 4000  # still bounded: this lands in a `monitor` event


def test_a_considered_failure_still_fails_the_phase(
    registry, dispatched, zone_root, tmp_path
) -> None:
    """The other side of the same line, so the distinction is a partition rather
    than a special case: a body that ran and said `false` **is** a verdict, binds,
    blocks the task, and is recorded against the artefact."""
    package_root = write_body(tmp_path / "says_no", verdict=False)
    register(registry, "says_no", inputs=["trace"])
    bind_kind(registry, "trace", ["says_no"])

    outcome = runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert outcome.passed is False
    assert outcome.blocks_the_task is True
    assert len(registry.get("handoff_store").read_verdicts(dispatched.outputs[0], 0)) == 1


@pytest.mark.parametrize(
    ("written", "because"),
    [
        ("{not json", "malformed JSON"),
        ("null", "JSON null"),
        ("[]", "a list rather than an object"),
        ('"done"', "a bare string"),
        ("42", "a number"),
    ],
)
def test_every_unusable_verdict_file_raises_validator_invalid(
    tmp_path, written: str, because: str
) -> None:
    """**`monitor` depends on this being the complete set**, and two of these
    escaped as something else until they asked.

    Their `VALIDATION_UNREACHED` is *"its `entry.sh` crashed, its agent died —
    nothing was decided"*, and `agent.Runner` catches **any** exception out of
    `run_phase` to report it. This test still matters under the broad catch: the
    two escapes below reached `_crash` as `HANDLING_FAILED` — *the monitor's own
    handler raised* — rather than being reported as a validation outcome at all.
    Measured: malformed JSON left as a `json.JSONDecodeError` and `null` as a
    `TypeError` from `"x" in None`. Both are a body producing garbage — exactly
    the case — and both would have surfaced as *the monitor's own handler raised*,
    which routes to `GiveUp` rather than escalating to the user.

    So a crashed validator was the quietest dead branch in the system, and it was
    reachable only by someone reading my raise sites against their event
    taxonomy.
    """
    from task_graph.ids import HandoffId
    from validator.environment import ConfigSource, EnvironmentConfig, ValidationEnvironment

    zone = tmp_path / "zone"
    zone.mkdir()
    env = ValidationEnvironment(
        zone=zone,
        cwd=zone,
        env={},
        config=EnvironmentConfig(ConfigSource.GLOBAL, {}),
        agent_id="a:output_validation",
    )
    env.verdict_file.write_text(written)

    with pytest.raises(ValidatorInvalid):
        read_verdict_file(env, [HandoffId.new()])


def test_a_verdict_file_missing_an_entry_raises(tmp_path) -> None:
    """`dict.get` would yield `None`, and `None` folded as falsy is
    indistinguishable from a genuine `False`."""
    from task_graph.ids import HandoffId
    from validator.environment import ConfigSource, EnvironmentConfig, ValidationEnvironment

    zone = tmp_path / "zone"
    zone.mkdir()
    env = ValidationEnvironment(
        zone=zone,
        cwd=zone,
        env={},
        config=EnvironmentConfig(ConfigSource.GLOBAL, {}),
        agent_id="a:output_validation",
    )
    a, b = HandoffId.new(), HandoffId.new()
    env.verdict_file.write_text(json.dumps({str(a): True}))
    with pytest.raises(ValidatorInvalid, match="no verdict for"):
        read_verdict_file(env, [a, b])


def test_the_registry_records_that_the_validator_ran(
    registry, dispatched, zone_root, package_root
) -> None:
    """Criterion 14's historical index, fed from the one place a run happens."""
    register(registry, "shape", inputs=["trace"])
    bind_kind(registry, "trace", ["shape"])
    assert registry.get("validator_specs").never_run() == ["shape"]

    runner(zone_root, package_root).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert registry.get("validator_specs").never_run() == []
    assert registry.get("validator_specs").has_ever_run("shape").runs == 1
