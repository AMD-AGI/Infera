"""Criteria 9, 10 and 21 — a fresh environment, the hook, and what "fresh" means.

**Criterion 21 cannot be tested by a directory check**, and this file must not
pretend otherwise. Measured: a fresh zone directory closes exactly *one* of the
channels a producer leaves state in — `/tmp`, `os.environ`, an inherited `cwd`,
`$HOME` and same-path reuse all still carry it. So the test enumerates
**channels**, which is Nix's framing: *"what matters for determinism is what the
build process can observe… we therefore specify building from the process's
perspective."* Stating it that way is what lets `env_mgr` change mechanisms later
without invalidating the criterion.

**What is asserted, and what is not.** The hook is the attributable layer and
`env_mgr`'s allow-list is the enforcing one; `env_mgr` has no sandbox
implementation today, so criterion 10's test asserts the hook half plus the
declaration half (`test_separation.py`), never the kernel half.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from task_graph.models import Task
from validator.boundary import Decision, ToolUseEvent, ZoneBoundaryHook
from validator.environment import (
    CHANNELS,
    ConfigSource,
    ValidationEnvironment,
    assert_attributable,
    assert_standard_unreachable,
    build_environment,
    choose_configuration,
)
from validator.protocols import PhaseKind, ValidatorInvalid
from validator.spec import ValidatorSpec


def env(root: Path, **kw) -> ValidationEnvironment:
    return build_environment(
        root,
        config=kw.pop("config", choose_configuration(PhaseKind.OUTPUT)),
        agent_id=kw.pop("agent_id", "agent-1:output_validation"),
        **kw,
    )


# --------------------------------------------------------------------------- #
# Criterion 9


def test_environment_is_rebuilt(tmp_path: Path) -> None:
    """Criterion 9. Two builds, two zones, and **never the same path**.

    Freshness comes from **allocation**, never from cleanup. pytest's `tmp_path`
    is a new numbered directory and its cleanup is explicitly best-effort; a
    guarantee that depends on a teardown succeeding is not a guarantee.
    """
    root = tmp_path / "zones"
    a, b = env(root), env(root)
    assert a.zone != b.zone
    assert a.zone.is_dir() and b.zone.is_dir()
    assert a.cwd == a.zone


@pytest.mark.parametrize(
    ("kind", "kw", "expected"),
    [
        (PhaseKind.INPUT, {"bound": {"X": "1"}, "consumer": {}}, ConfigSource.BOUND),
        (PhaseKind.OUTPUT, {"bound": {"X": "1"}, "producer": {}}, ConfigSource.BOUND),
        (PhaseKind.INPUT, {"consumer": {}, "producer": {}}, ConfigSource.CONSUMER),
        (PhaseKind.OUTPUT, {"consumer": {}, "producer": {}}, ConfigSource.PRODUCER),
        (PhaseKind.INPUT, {"producer": {}}, ConfigSource.GLOBAL),
        (PhaseKind.OUTPUT, {"consumer": {}}, ConfigSource.GLOBAL),
        (PhaseKind.OUTPUT, {}, ConfigSource.GLOBAL),
    ],
)
def test_configuration_chain_order(kind: PhaseKind, kw: dict, expected: ConfigSource) -> None:
    """Criterion 9's chain, spec §8.2: bound env, else the **consumer's** for
    input, else the **producer's** for output, else a predefined global one.

    The middle two are why the phases sit inside the task: the right configuration
    is the one already *resolved*, which is not the same as the one already
    running. The source is recorded on the result, so this asserts which row
    applied rather than the contents that row happened to produce.
    """
    assert choose_configuration(kind, **kw).source is expected


def test_the_bound_row_takes_the_named_agents_env(registry, dispatched, tmp_path) -> None:
    """§8.2 row 1, reachable at last: *"bound to a real **agent** with a declared
    environment → **that one**"*.

    *That one* is the agent's `env`, which `agent.schema.json` already had — so
    the validator spec names the agent rather than carrying a second copy. Three
    packages to land it: `spec_loader`'s key, this field and resolve, and
    `closure`'s catalogue check.
    """
    from tests.validator.conftest import (
        DictSpecRegistry,
        bind_kind,
        validator_record,
        write_body,
    )
    from validator.environment import ConfigSource
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    registry.register(
        "agent_specs",
        DictSpecRegistry("agent", {"profiler": {"name": "profiler", "env": {"CUDA": "12"}}}),
    )
    record = validator_record("shape", inputs=["trace"])
    record["agent"] = "profiler"
    registry.get("validator_specs").add("shape", record, origin="s.jsonnet")
    bind_kind(registry, "trace", ["shape"])

    outcome = PhaseRunner(
        StrictLevel.DEFAULT,
        zone_root=tmp_path / "zones",
        package_root=write_body(tmp_path / "pkg"),
    ).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.ran[0].verdict.environment["source"] == ConfigSource.BOUND.value


def test_an_unresolvable_agent_raises_rather_than_falling_back(
    registry, dispatched, tmp_path
) -> None:
    """**Absent and unresolvable are different questions and must not share an
    answer** — `closure`'s correction of a conflation of mine.

    Absent is the declared way to take the global row. A validator naming
    `profilr` wanted a specific environment, and falling back would give it a
    *working* one that is not the one it configured — the failure `closure`'s
    fatal load-time check exists to prevent, arriving at run time instead. The
    symptom of the silent version is the bad one: a validator that **runs**, in
    the wrong environment, producing a verdict somebody trusts.
    """
    from tests.validator.conftest import (
        DictSpecRegistry,
        bind_kind,
        validator_record,
        write_body,
    )
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    registry.register("agent_specs", DictSpecRegistry("agent", {"profiler": {"name": "profiler"}}))
    record = validator_record("shape", inputs=["trace"])
    record["agent"] = "profilr"
    registry.get("validator_specs").add("shape", record, origin="s.jsonnet")
    bind_kind(registry, "trace", ["shape"])

    with pytest.raises(ValidatorInvalid, match="does not resolve") as exc:
        PhaseRunner(
            StrictLevel.DEFAULT,
            zone_root=tmp_path / "zones",
            package_root=write_body(tmp_path / "pkg"),
        ).run_phase(PhaseKind.OUTPUT, dispatched, registry)
    assert "profiler" in str(exc.value)  # the candidates, which make it a quick fix


def test_an_empty_agent_name_raises_rather_than_reading_as_absent(
    registry, dispatched, tmp_path
) -> None:
    """`""` is **not** absent here: `_bound_environment` branches on `is None`,
    so an empty name falls through to the resolve and raises.

    Pinned because I reported the opposite to `spec-loader` — that this side
    quietly took the global row — and they measured my code and found it loud.
    The wrong version made a shared argument look stronger than it was, and it
    reached a test docstring in a third package before anyone ran it.

    Where the belief came from is the part worth keeping. This package briefly
    had **two readers of the same key that disagreed on this input**: the
    withdrawn `agent_of` normalised `""` to `None` — deliberately, the `entry: ""`
    lesson — and `_bound_environment` never called it, reading `spec.agent` off
    the model instead. I described the accessor's semantics as the package's.
    The accessor is gone, so one answer remains; the test is what keeps it one.

    It also matters beyond this file: `validator.schema.json`'s `minLength: 1`
    is the reason `""` is unreachable from a real document, and
    `tests/interfaces/test_schema_clauses.py` now cites **this** behaviour when
    it says the clause's silent dependant is `agent`'s and not this package's.
    A change here that made `""` read as absent would falsify a claim in
    somebody else's test.
    """
    from tests.validator.conftest import (
        DictSpecRegistry,
        bind_kind,
        validator_record,
        write_body,
    )
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    registry.register("agent_specs", DictSpecRegistry("agent", {"profiler": {"name": "profiler"}}))
    record = validator_record("shape", inputs=["trace"])
    record["agent"] = ""  # rejected by the schema; reachable only if that clause goes
    registry.get("validator_specs").add("shape", record, origin="s.jsonnet")
    bind_kind(registry, "trace", ["shape"])

    with pytest.raises(ValidatorInvalid, match="does not resolve"):
        PhaseRunner(
            StrictLevel.DEFAULT,
            zone_root=tmp_path / "zones",
            package_root=write_body(tmp_path / "pkg"),
        ).run_phase(PhaseKind.OUTPUT, dispatched, registry)


def test_a_validator_naming_no_agent_still_takes_the_global_row(
    registry, dispatched, tmp_path
) -> None:
    """Absent is legal, and it is the ordinary case: most validators name no
    agent and take §8.2's last row.

    This test used to say *the chain is dead in the phase runner* — three of four
    rows had no source at all, fed by `getattr(spec, "environment", None)` against
    a model with `extra="forbid"`. `bound` is now reachable; `consumer` and
    `producer` still are not, and the enumeration below is what keeps that
    visible rather than implicit.

    `test_configuration_chain_order` above exercises all four rows and passes —
    correctly, because `choose_configuration` is a pure function and its logic is
    right. It calls it **directly**. The real caller cannot: the arguments were
    `getattr(spec, "environment", None)` and `getattr(task, "environment", None)`,
    and **neither `ValidatorSpec` nor `task_graph.Task` has an `environment`
    field** — `ValidatorSpec` sets `extra="forbid"`, so no document can add one.
    Three of four rows returned `None` on every call ever made.

    A `getattr` with a default is not a field access; it is dead code that reads
    as live. Found by applying `env_mgr`'s `stubs.Task.repos` finding here — the
    same shape, and theirs had shipped too.

    Asserted through the **phase runner**, not the function, because that is the
    gap: the unit test's coverage was real and its implication was not.
    """
    from tests.validator.conftest import bind_kind, validator_record, write_body
    from validator.phase import CONFIGURATION_SOURCES, PhaseRunner
    from validator.protocols import StrictLevel

    assert "environment" not in ValidatorSpec.model_fields
    assert "environment" not in Task.model_fields
    assert ValidatorSpec.model_config["extra"] == "forbid"  # and no document may add one

    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])
    outcome = PhaseRunner(
        StrictLevel.DEFAULT,
        zone_root=tmp_path / "zones",
        package_root=write_body(tmp_path / "pkg"),
    ).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    # The global row, because the shipped `FakeRunner` has no attempts — so the
    # producer row below has nothing to report, which is an answer rather than a
    # missing field. See `test_the_producer_row_is_live_when_the_runner_has_one`.
    assert outcome.ran[0].verdict.environment["source"] == ConfigSource.GLOBAL.value

    # And what is still unsourced is enumerated rather than implicit. **One row,
    # not two, since `agent` 3155ca2** — and the one that remains is unreachable
    # in principle rather than unbuilt: `env.prepare` has a single call site
    # inside `_deploy`, which `_one_phase` reaches only in `RUNNING`, so at
    # `INPUT_VALIDATING` no `Prepared` exists for the task. §8.2 calls this row
    # *"the task about to run"*, and about-to-run is exactly before `prepare`.
    unsourced = [row for row, why in CONFIGURATION_SOURCES if "NO SOURCE" in why]
    assert unsourced == ["consumer"]


def test_the_producer_row_is_live_when_the_runner_has_one(
    registry, dispatched, tmp_path: Path
) -> None:
    """Criterion 9's third row, built by `agent` on request (`3155ca2`).

    §8.2: *"otherwise, output validation — the producer's, the task that just
    ran."* The configuration is `TaskAttempt.environment`, a read-only mapping
    carried from `_deploy` onwards, and `attempt_of(task.id)` is the handle.

    **Asserted through the phase runner**, for the reason the test above states:
    a unit test of `choose_configuration` already passed while three of four rows
    were dead, so coverage of the chain says nothing about whether the caller can
    reach it.
    """
    from tests.validator.conftest import bind_kind, validator_record, write_body
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    class _Attempt:
        environment = MappingProxyType({"MODEL": "llama", "PATH": "/usr/bin"})

    runner = registry.get("runner")
    runner.attempt_of = lambda task_id: _Attempt() if task_id == dispatched.id else None

    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])
    outcome = PhaseRunner(
        StrictLevel.DEFAULT,
        zone_root=tmp_path / "zones",
        package_root=write_body(tmp_path / "pkg"),
    ).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.ran[0].verdict.environment["source"] == ConfigSource.PRODUCER.value


def test_a_non_leafs_empty_configuration_falls_through_to_global(
    registry, dispatched, tmp_path: Path
) -> None:
    """**Empty reads as absent, and it is a decision rather than a coincidence.**

    `agent` is explicit that `TaskAttempt.environment` is `{}` until `_deploy`,
    the same shape as `executor is None` — and for a **non-leaf** it stays `{}`,
    because the scheduler runs a non-leaf's main phase by unfolding and nothing
    ever calls `prepare`. §8.2's row is *the configuration already resolved*, so
    a task that resolved none must fall through.

    Passing `{}` on would select `PRODUCER` and hand the validation an **empty
    environment** — a value that is wrong but not type-wrong, which is the family
    `interfaces.md` §4.11 names and the one nothing raises on.
    """
    from tests.validator.conftest import bind_kind, validator_record, write_body
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    class _NeverDeployed:
        environment = MappingProxyType({})

    registry.get("runner").attempt_of = lambda task_id: _NeverDeployed()

    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])
    outcome = PhaseRunner(
        StrictLevel.DEFAULT,
        zone_root=tmp_path / "zones",
        package_root=write_body(tmp_path / "pkg"),
    ).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    assert outcome.ran[0].verdict.environment["source"] == ConfigSource.GLOBAL.value


def test_the_chain_reads_a_bare_string_kind_too() -> None:
    """`choose_configuration` is reachable from outside the phase runner, and
    `PhaseKind` is a `(str, Enum)`. A caller holding the *value* would otherwise
    fall through both `is` comparisons to the global row without complaint —
    picking the wrong environment configuration and reporting nothing."""
    assert choose_configuration("input_validation", consumer={}).source is ConfigSource.CONSUMER
    assert choose_configuration("output_validation", producer={}).source is ConfigSource.PRODUCER
    with pytest.raises(ValueError, match="not a valid PhaseKind"):
        choose_configuration("main")


def test_reusing_a_configuration_is_fine_and_inheriting_an_environment_is_not(
    tmp_path: Path,
) -> None:
    """Spec §8.2 in one line. Same configuration, different environment."""
    config = choose_configuration(PhaseKind.OUTPUT, producer={"MODEL": "llama"})
    a, b = env(tmp_path, config=config), env(tmp_path, config=config)
    assert a.config is b.config
    assert a.env["MODEL"] == b.env["MODEL"] == "llama"
    assert a.zone != b.zone


# --------------------------------------------------------------------------- #
# Criterion 10


def test_hook_denies_and_logs(tmp_path: Path) -> None:
    """Criterion 10. **The spy and the denier are the same object**, not two.

    Measured against the SDK: one *synchronous* `PreToolUse` callback logs every
    attempt before deciding and then denies. The async form cannot block — *"async
    outputs can't block, modify, or inject context into the operation since the
    agent has already moved on"* — so logging-only is not an available
    optimisation.
    """
    standard = tmp_path / "standard"
    standard.mkdir()
    (standard / "answers.yaml").write_text("threshold: 3")
    hook = ZoneBoundaryHook(standards=(standard,))

    denied = hook.on_tool_use(
        ToolUseEvent("Read", {"file_path": str(standard / "answers.yaml")}, agent_id="producer")
    )
    allowed = hook.on_tool_use(
        ToolUseEvent("Read", {"file_path": str(tmp_path / "notes.md")}, agent_id="producer")
    )
    assert denied is Decision.DENY
    assert allowed is Decision.ALLOW
    # Logged before deciding, and allowed attempts are logged too — that ordering
    # is what makes the log evidence rather than a summary of denials.
    assert [e.decision for e in hook.log()] == [Decision.DENY, Decision.ALLOW]
    assert "standard" in hook.log()[0].reason


def test_no_producer_frame_reads_the_standard(tmp_path: Path) -> None:
    """Criterion 10. Every read the producer attempted, bracketed by identity.

    The SDK has **no** field denoting a stack, caller, frame, origin or parent —
    the only identity fields are `agent_id` and `agent_type`, both optional. So
    "who called this" is *recorded* rather than inferred from the call stack,
    which is the same device `tests/task_graph/test_authority.py` justifies with
    the observation that inference here *"would be unimplementable in any honest
    way."*
    """
    standard = tmp_path / "standard"
    standard.mkdir()
    (standard / "answers.yaml").write_text("x")
    hook = ZoneBoundaryHook(standards=(standard,))

    for who in ("producer", "validator"):
        hook.on_tool_use(
            ToolUseEvent("Read", {"file_path": str(standard / "answers.yaml")}, agent_id=who)
        )

    reached = [e for e in hook.log() if e.decision is Decision.ALLOW]
    assert not [e for e in reached if e.agent_id == "producer"]
    assert all(e.decision is Decision.DENY for e in hook.log())


def test_the_hook_cannot_see_an_indirect_read(tmp_path: Path) -> None:
    """The honest ceiling, asserted so nobody upgrades this layer quietly.

    Measured: `Bash{'command': 'python3 reader.py'}` returns ALLOW, because there
    is no path in the payload and therefore nothing to match. Anthropic documents
    the same for declarative rules — deny rules *"don't apply to arbitrary
    subprocesses that read or write files indirectly."*
    """
    standard = tmp_path / "standard"
    standard.mkdir()
    hook = ZoneBoundaryHook(standards=(standard,))
    assert (
        hook.on_tool_use(ToolUseEvent("Bash", {"command": "python3 reader.py"})) is Decision.ALLOW
    )


def test_a_phase_without_an_agent_id_is_refused() -> None:
    """§8.2. `agent_id` is *"present only when the hook fires from inside a
    Task-spawned sub-agent; absent on the main thread"*, so a main-thread phase is
    unattributable and criterion 10 is not testable for it.

    **That** a phase must be attributable is this module's requirement; **how** a
    backend delivers it is `agent` design O6 and is open. So this fails loudly
    rather than assuming a mechanism.
    """
    assert assert_attributable("a:input_validation") == "a:input_validation"
    with pytest.raises(ValidatorInvalid, match="attributable"):
        assert_attributable(None)


def test_the_standard_is_asserted_unreachable_not_arranged(tmp_path: Path) -> None:
    """**Absence is a property to assert, not merely to arrange.**

    SWE-bench's answer key was physically absent for two years and still leaked:
    `git remote remove origin` leaves the fix commit reachable through
    `git cat-file --batch-all-objects`. Their fix is the lesson — the clone now
    ends with a count that must be zero or it exits nonzero.
    """
    zone = tmp_path / "zone"
    zone.mkdir()
    assert_standard_unreachable(zone, [tmp_path / "elsewhere"])
    inside = zone / "standard"
    inside.mkdir()
    with pytest.raises(ValidatorInvalid, match="reachable"):
        assert_standard_unreachable(zone, [inside])


# --------------------------------------------------------------------------- #
# Criterion 21


def test_producer_leavings_absent(tmp_path: Path) -> None:
    """Criterion 21, named after Bazel's `test_sandbox_undeclared_deps` and
    Concourse's `It("doesn't mount its file system into the next task")`.

    It enumerates **channels**, not a directory. `CHANNELS` is the measured list
    and each entry is checked, so adding a channel to the list without closing it
    fails here.
    """
    root = tmp_path / "zones"
    producer_env = env(root)
    litter = {
        "zone": producer_env.zone / "litter.txt",
        "tmp": Path(producer_env.env["TMPDIR"]) / "litter.txt",
        "home": Path(producer_env.env["HOME"]) / "litter.txt",
    }
    for path in litter.values():
        path.write_text("the producer was here")
    producer_env.env["SECRET"] = "the producer's"

    validation = env(root)
    checked = set()

    assert not (validation.zone / "litter.txt").exists()
    checked.add("zone")
    assert not (Path(validation.env["TMPDIR"]) / "litter.txt").exists()
    checked.add("tmp")
    assert not (Path(validation.env["HOME"]) / "litter.txt").exists()
    checked.add("home")
    assert validation.cwd == validation.zone  # never inherited
    checked.add("cwd")
    assert "SECRET" not in validation.env
    assert set(validation.env) & set(os.environ) <= {"TMPDIR", "HOME", "PWD"}
    checked.add("environ")

    assert checked == {name for name, _ in CHANNELS}


def test_rebuild_not_reuse_across_consecutive_runs(tmp_path: Path) -> None:
    """Criterion 21, named after Bazel's
    `test_sandbox_old_contents_not_reused_in_consecutive_builds`.

    **Path identity is the property, not emptiness.** Strategy B in the survey —
    `rmtree` then recreate, which is tox's `-r` — genuinely removes the litter and
    fails only this probe. It matters because if the validation's zone has the
    producer's absolute path, any absolute path the producer baked into an
    artefact still resolves, which is exactly the locality dependence the handoff
    module exists to catch.
    """
    root = tmp_path / "zones"
    seen = {env(root).zone for _ in range(5)}
    assert len(seen) == 5


def test_the_zone_comes_from_env_mgr_when_it_is_wired(registry, dispatched, tmp_path) -> None:
    """`interfaces.md` §4.3 rev. 5 added `env_mgr` to this module's resolve row,
    for `prepare_validation(task, execution, phase)`.

    It places the zone as a **sibling of the producing task's zone, never a
    descendant** — their D5, and their criterion 13 is untrue without it. Nothing
    on this side changed: `build_environment` allocates a fresh directory *inside*
    whatever root it is handed, and a fresh directory inside a correctly-placed
    sibling is still a sibling.

    Asserted against a stub, because `validator` may not import `env_mgr` and this
    is about *which root is used*, not about their placement — which their own
    `test_validation_is_a_sibling_not_a_descendant` owns.
    """
    from tests.validator.conftest import bind_kind, validator_record, write_body
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    placed_root = tmp_path / "beside-the-producer"
    placed_root.mkdir()

    class StubEnvManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def prepare_validation(self, task, execution, phase):
            self.calls.append((str(task.id), str(getattr(phase, "value", phase))))
            return SimpleNamespace(root=str(placed_root), phase=str(phase), materials=())

    env_mgr = StubEnvManager()
    registry.register("env_mgr", env_mgr)
    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])

    outcome = PhaseRunner(StrictLevel.DEFAULT, package_root=write_body(tmp_path / "pkg")).run_phase(
        PhaseKind.OUTPUT, dispatched, registry
    )

    assert outcome.passed is True
    assert env_mgr.calls == [(str(dispatched.id), "output_validation")]
    # The zone is inside the placed root, and freshly allocated within it.
    zone = Path(outcome.ran[0].verdict.environment["zone"])
    assert zone.parent == placed_root
    assert zone != placed_root


def test_the_body_is_told_where_its_materials_are(registry, dispatched, tmp_path) -> None:
    """`demo` F-D5's residue, and it was one name wide.

    `env_mgr.prepare_validation` stages copies under `<placed.root>/materials/`
    and returns the paths; this module allocated its zone *inside* that root and
    discarded them, so a body sat in `<placed.root>/validation-XXXX/` with the
    copies at `../materials` — reachable, and named by nothing. A body reading
    `../materials` would rely on a relative path no document declares.

    `materials.json` is now beside `args.json` and `inputs.json` in the body's
    `cwd`, holding zone-relative paths. **Written even when nothing was staged**,
    because an absent file and an empty one are different records.
    """
    from tests.validator.conftest import bind_kind, validator_record, write_body
    from validator.phase import ZONE_FILES, PhaseRunner
    from validator.protocols import StrictLevel

    placed_root = tmp_path / "beside-the-producer"
    hid = dispatched.outputs[0]
    (placed_root / "materials" / str(hid) / "v0").mkdir(parents=True)

    class StubEnvManager:
        def prepare_validation(self, task, execution, phase):
            hid = next(iter(task.outputs))
            return SimpleNamespace(
                root=str(placed_root),
                phase=str(phase),
                # A **mapping**, matching `env_mgr.ValidationZone.materials`
                # since 789796d. It was a bare tuple, and `tuple()` over the new
                # shape yields the *keys* — so this stub is what would have kept
                # a real break green. `test_the_validation_zone_stub_matches_the_real_seam`
                # is the guard that stops that happening a second time.
                materials={hid: str(placed_root / "materials" / str(hid) / "v0")},
            )

    registry.register("env_mgr", StubEnvManager())
    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])

    outcome = PhaseRunner(StrictLevel.DEFAULT, package_root=write_body(tmp_path / "pkg")).run_phase(
        PhaseKind.OUTPUT, dispatched, registry
    )

    zone = Path(outcome.ran[0].verdict.environment["zone"])
    declared = json.loads((zone / "materials.json").read_text())

    # Keyed by handoff id, which is what criterion 4's multi-input case needs and
    # what a flat list could not give: `inputs.json` is sorted while `stage` walks
    # declaration order and skips absent versions, so the two cannot be zipped.
    assert list(declared) == [str(hid)]
    # Relative to the body's cwd, which is the zone — not `../materials`.
    assert (zone / declared[str(hid)]).resolve().is_dir()
    assert "materials.json" in ZONE_FILES


def test_materials_json_is_written_even_when_nothing_is_staged(
    registry, dispatched, tmp_path
) -> None:
    """An absent file and an empty one are different records — the same rule
    `handoff` follows by creating `validation.yaml` with an empty `verdicts:`
    list rather than not creating it. A body that cannot tell *nothing was
    staged* from *this system does not stage* is the JUnit failure one directory
    down."""
    from tests.validator.conftest import bind_kind, validator_record, write_body
    from validator.phase import PhaseRunner
    from validator.protocols import StrictLevel

    registry.get("validator_specs").add(
        "shape", validator_record("shape", inputs=["trace"]), origin="s.jsonnet"
    )
    bind_kind(registry, "trace", ["shape"])

    outcome = PhaseRunner(  # no env_mgr registered, so nothing is staged
        StrictLevel.DEFAULT,
        zone_root=tmp_path / "zones",
        package_root=write_body(tmp_path / "pkg"),
    ).run_phase(PhaseKind.OUTPUT, dispatched, registry)

    zone = Path(outcome.ran[0].verdict.environment["zone"])
    assert (zone / "materials.json").exists()
    assert json.loads((zone / "materials.json").read_text()) == {}


def test_a_script_body_gets_a_shell_default_path_not_an_empty_one(tmp_path: Path) -> None:
    """`demo` F-D5 claims a script body starts with an empty `PATH`. Measured, it
    does not — POSIX `sh` substitutes a built-in default when none is inherited.

    Pinned because the residue is real but is the *opposite* of the claim: the
    value comes from the shell rather than from the configuration, so it is not
    something `CHANNELS` records or a `config` can reason about, and it will
    differ on another platform. If a future change starts passing an explicit
    `PATH`, this test is where that decision becomes visible.
    """
    import subprocess

    env = build_environment(
        tmp_path / "zones",
        config=choose_configuration(PhaseKind.OUTPUT),
        agent_id="a:output_validation",
    )
    assert "PATH" not in env.env  # we supply none

    probe = env.zone / "probe.sh"
    probe.write_text('#!/bin/sh\nprintf "%s" "${PATH-<<unset>>}"\n')
    got = subprocess.run(
        ["/bin/sh", str(probe)], cwd=env.cwd, env=dict(env.env), capture_output=True, text=True
    ).stdout

    assert got not in ("", "<<unset>>")  # not empty, and not unset
    assert "/bin" in got


def test_freshness_does_not_depend_on_a_teardown(tmp_path: Path) -> None:
    """No cleanup runs, and the guarantee still holds. nox's staleness check is
    the counter-example: `if not os.environ.get("NOX_ENABLE_STALENESS_CHECK", ""):
    return True` — disabled for a Python 2.7 bug and never re-enabled. **A
    staleness check with a hidden off-switch is worse than none.**

    Walks the **AST**, not the source text: both names appear in this module's
    prose, and `"scheduler" in runner.py` being `True` from two docstring mentions
    is the measured reason a substring check fails for the wrong reason.
    """
    import ast
    from pathlib import Path as P

    tree = ast.parse(
        P(__file__).resolve().parents[2].joinpath("validator/environment.py").read_text()
    )
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    assert "rmtree" not in names  # freshness is allocation, never cleanup
    assert "environ" not in names  # no inherited block, and no hidden off-switch
    assert "getenv" not in names
