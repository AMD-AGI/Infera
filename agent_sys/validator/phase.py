"""The two validation phases, and the seam `agent.Runner` calls.

`TaskRunner` runs all three phases for the one task the scheduler dispatched, and
the scheduler sees phase 2 and nothing else. Nothing here schedules, cascades,
invalidates or notifies: the phase returns a `PhaseOutcome`, the runner turns a
failing one into a task failure, and the scheduler's response is to stop
scheduling. The temptation at this exact seam is to add a retry or an escalation,
and either would give the runner an opinion about *what*.

**This is the only module in the package that touches a manager**, and it
resolves `handoff_mgr`, `validator_specs`, `handoff_store` and `closures` from
the component `Registry` **at call time, never by import** — the rule
`task_graph` design §2 establishes and this package does not get to break.

`closures` is the fourth, added after `demo`'s first assembly found that a
closure's phase validators were read by nothing. It is asked for the whole set
rather than for the parts to join, which is why `handoff_specs` is no longer
resolved here at all: that join is `closure`'s, and doing it twice was the
defect.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from handoff.protocols import Verdict
from task_graph.ids import HandoffId
from validator import history
from validator.composite import Composite
from validator.environment import (
    ValidationEnvironment,
    build_environment,
    choose_configuration,
)
from validator.history import Target
from validator.protocols import (
    PhaseKind,
    SkipRecord,
    Strength,
    StrictLevel,
    ValidatorInvalid,
    VerdictRecord,
)
from validator.reducers import get_reducer
from validator.report import PhaseOutcome
from validator.spec import ValidatorSpec

__all__ = [
    "AgentBodyRunner",
    "BodyRunner",
    "LeafValidator",
    "PhaseRunner",
    "ScriptBodyRunner",
    "read_verdict_file",
]


# --------------------------------------------------------------------------- #
# Running a body


class BodyRunner(Protocol):
    """Runs a validator's body so that it leaves `env.verdict_file` behind.

    One seam, two shapes, and the verdict file is what makes them substitutable:
    the phase runner reads one thing either way, and `Validator.__call__` is what
    it looks like once read. That is the property the withdrawn Python callable
    could not have — a callable cannot express a validator an agent is
    responsible for without a wrapper whose whole job is to run an agent.
    """

    def run(
        self,
        spec: ValidatorSpec,
        env: ValidationEnvironment,
        handoffs: Mapping[HandoffId, Any],
        registry: Any,
    ) -> Any:
        """Returns the `AgentId` that ran the body, or `None` if no agent did.

        The return is how the real identity reaches a verdict: `env` is frozen and
        built before the body runs, so there is no earlier moment at which the id
        exists. `agent`'s executor mints a fresh **unbound** `Agent` per phase —
        different from the producer's, and different between the two phases of one
        run — which is what criterion 10 needs and what a derived string could not
        give.
        """
        ...


#: How much of a crashed body's `stderr` reaches the exception message, and
#: **from which end**. The end is the defect that was here: `[:200]` kept the
#: *head* of a Python traceback, which is `Traceback (most recent call last):`
#: followed by the outermost frames — so the file, the line and the exception
#: type, all of which live at the tail, were the part discarded. Measured by
#: `demo` against a real body: the recorded message was cut mid-path inside the
#: zone, before the first frame, and named none of the three.
#:
#: A tail rather than a head, and a budget large enough for a few frames. Not
#: unbounded: this lands in a `monitor` event, and a body that dies inside a
#: loop can produce megabytes.
STDERR_TAIL = 2000


def _stderr_tail(stderr: str) -> str:
    text = (stderr or "").strip()
    if len(text) <= STDERR_TAIL:
        return text
    return "…" + text[-STDERR_TAIL:]


class ScriptBodyRunner:
    """`entry` present — the script, in the rebuilt environment.

    `entry.sh` carries the assembly command and the run command; for a code-shaped
    check both go against the shipped pytest harness. Nothing here frameworks the
    test code: spec §7 draws that line and this module must not cross it.
    """

    def __init__(self, package_root: Path) -> None:
        self._root = package_root

    def run(
        self,
        spec: ValidatorSpec,
        env: ValidationEnvironment,
        handoffs: Mapping[HandoffId, Any],
        registry: Any,
    ) -> Any:
        """A script has no agent, so this reports none. Criterion 10's
        attribution is unavailable for a programmatic body and the caller says so
        rather than substituting one."""
        entry = spec.body.get("entry")
        if entry is None:  # pragma: no cover - selected on this very field
            raise ValidatorInvalid(f"{spec.name}: no entry to run")
        env.args_file.write_text(json.dumps(dict(spec.args), indent=2, sort_keys=True))
        (env.zone / "inputs.json").write_text(
            json.dumps(sorted(str(h) for h in handoffs), indent=2)
        )
        completed = subprocess.run(  # noqa: S603 - the entry is a spec-declared path
            ["/bin/sh", str(self._root / entry)],
            cwd=env.cwd,
            env=dict(env.env),
            capture_output=True,
            text=True,
        )
        if not env.verdict_file.exists():
            # **A body that reports nothing must not pass — and must not fail
            # either.** The first half was here from the start; the second was
            # not, and the asymmetry was the defect. A nonzero exit with no
            # verdict file used to have `{hid: False}` fabricated for it, and a
            # fabricated `False` is byte-identical to a considered one
            # (`scratch/impl-2026-08/validator/p6_crashed_body_verdict.py`). So a
            # segfaulting validator reported *the validator worked and the answer
            # is no*, which is the flattening `monitor` spec §2.1 exists to
            # prevent: `VALIDATION_FAILED` says a branch is **judged** dead,
            # `VALIDATION_UNREACHED` says it is **undetermined**, and the
            # analysing dispatcher's whole job is telling those apart.
            #
            # A crash is not a judgement. Both halves now raise, and the exit
            # status is in the message because it is the only thing that
            # distinguishes them for a human.
            raise ValidatorInvalid(
                f"{spec.name}: exited {completed.returncode} and wrote no "
                f"{env.verdict_file.name}; nothing was decided. "
                f"stderr: {_stderr_tail(completed.stderr) or '<empty>'}"
            )


class AgentBodyRunner:
    """`entry` absent — an agent, given `readme.md` as its instruction.

    The agent writes the same verdict file. **The mechanism is not chosen here**:
    a phase must be separately attributable (`environment.assert_attributable`
    enforces that), and *how* a backend delivers it — subagent, `fork_session`,
    `resume`, a second client — is `agent` design O6 and is open. So this
    resolves the executor by name at call time and fails loudly rather than
    assuming one.
    """

    #: The component that runs an agent-bodied validator. Resolved, never
    #: imported: `validator` may not import `agent`.
    COMPONENT = "validator_executor"

    def run(
        self,
        spec: ValidatorSpec,
        env: ValidationEnvironment,
        handoffs: Mapping[HandoffId, Any],
        registry: Any,
    ) -> Any:
        env.args_file.write_text(json.dumps(dict(spec.args), indent=2, sort_keys=True))
        if self.COMPONENT not in registry:
            raise ValidatorInvalid(
                f"{spec.name} is agent-bodied and no {self.COMPONENT!r} is registered; "
                f"the backend mechanism is agent design O6 and is open"
            )
        return registry.get(self.COMPONENT).run_body(spec, env, handoffs, registry)


def runner_for(spec: ValidatorSpec, package_root: Path | None) -> BodyRunner:
    """A leaf's body is required, so `admit` has already rejected a `None` here.

    Stated rather than assumed: a composite has no body and never reaches this —
    `_validator` builds a `Composite` over `LeafValidator` members instead.

    A **script** body needs a package root to resolve against and there is no
    safe default: the working directory is a plausible path that is nobody's
    package. An **agent** body needs none, because the executor is handed the
    spec rather than a resolved path, so the absence is refused here rather than
    at construction.
    """
    if not spec.body:  # pragma: no cover - admit rejects a bodiless leaf
        raise ValidatorInvalid(f"{spec.name}: no body to run")
    if not spec.body.get("entry"):
        return AgentBodyRunner()
    if package_root is None:
        raise ValidatorInvalid(
            f"{spec.name} has a script body and no package root to resolve "
            f"{spec.body['entry']!r} against; pass PhaseRunner(..., package_root=...). "
            f"The working directory is not a default — it is wherever the process "
            f"started, and resolving a package-relative path against it finds "
            f"either nothing or the wrong file"
        )
    return ScriptBodyRunner(package_root)


#: The body-facing contract: every file a validation body may rely on, all of
#: them in its `cwd`. This module writes the first three and reads the last.
ZONE_FILES = ("args.json", "inputs.json", "materials.json", "verdict.json")


def _declare_materials(env: ValidationEnvironment, staged: Mapping[Any, str]) -> None:
    """Name what the body is validating, in its own working directory.

    **This closes `demo`'s F-D5 residue**, and it was one name wide.
    `env_mgr.prepare_validation` stages copies of the artefacts under
    `<placed.root>/materials/` — *"so a body handed handoff ids has somewhere to
    read them from"*, their words — and returns the paths. This module then
    allocated its fresh zone *inside* that root and **discarded them**, so a body
    sat in `<placed.root>/validation-XXXX/` with the copies at `../materials`:
    reachable, and named by nothing. A body reading `../materials` would be
    relying on a relative path no document declares.

    So the staged paths are written where the other three already are, **relative
    to `cwd`**, and the answer to *by what declared name does a body find what it
    is validating* is `materials.json`.

    **A map from handoff id to path**, which is what a multi-input validator
    needs and could not have derived: spec §4.1 makes the binding many-to-many
    and criterion 4 requires `dict[HandoffId, bool]`, so a body checking three
    handoffs has to know which copy is which. It shipped as a bare list first,
    because deriving the id would have meant reading `env_mgr`'s
    `<materials>/<hid>/v<N>` layout — `env_mgr` then changed
    `ValidationZone.materials` to carry the association, which is
    `engineer_principle.md` §4.4's second smell resolved at the right end:
    `layout.stage` had the id in hand and was throwing it away.

    Written **unconditionally**, empty list included: an absent file and an empty
    one are different records, and a body that cannot tell "nothing was staged"
    from "this system does not stage" is the JUnit failure one directory down.
    """
    env.zone.joinpath("materials.json").write_text(
        json.dumps(
            {str(hid): _relative_to(path, env.zone) for hid, path in staged.items()},
            indent=2,
            sort_keys=True,
        )
    )


#: Where each row of spec §8.2's configuration chain comes from, and whether it
#: has a source today. **Two of the four still do not**, and this table exists so
#: that is visible rather than silent.
#:
#: The call site used to be `getattr(spec, "environment", None)` and
#: `getattr(task, "environment", None)` — which *look* like field accesses and are
#: not: neither `ValidatorSpec` nor `task_graph.Task` has an `environment` field,
#: and `ValidatorSpec` sets `extra="forbid"`, so no document can add one. Both
#: yielded `None` on every call that has ever been made, so **every validation
#: takes the GLOBAL row** and the chain is dead in the phase runner.
#:
#: `test_configuration_chain_order` did not catch it because it calls
#: `choose_configuration` **directly** with the arguments — a correct unit test of
#: a pure function whose real caller can never supply three of its four inputs.
#: `test_only_the_global_row_is_reachable_today` is the one that asserts what the
#: caller can actually do.
#:
#: Found by applying `env_mgr`'s `stubs.Task.repos` finding to this package:
#: a `getattr` with a default is not a field access, and dead code reads as live.
CONFIGURATION_SOURCES: tuple[tuple[str, str], ...] = (
    (
        "bound",
        "the `env` of the agent spec named by `ValidatorSpec.agent` — reachable since "
        "fe9fd55 and this package's step 3; three packages to land one row",
    ),
    (
        "consumer",
        "the consuming task's resolved configuration — NO SOURCE, and unreachable "
        "in principle rather than unbuilt: it is env_mgr's `Prepared.environment`, "
        "and `prepare` has one call site (agent/runner.py:668, from `_deploy`) "
        "which `_one_phase` reaches only in RUNNING — strictly after "
        "INPUT_VALIDATING. Spec 8.2's phrase for this row is `the task about to "
        "run`, and about-to-run is exactly before `prepare`",
    ),
    (
        "producer",
        "the producing task's resolved configuration — LIVE since agent 3155ca2: "
        "`attempt_of(task.id).environment`, a read-only mapping carried on the "
        "TaskAttempt from `_deploy` onwards. Empty reads as absent, because `{}` "
        "means the task never deployed — every non-leaf, whose main phase the "
        "scheduler runs by unfolding",
    ),
    ("global_", "the `validation_env` component, when the composition root registers one"),
)


def _configuration_sources(task: Any, spec: ValidatorSpec, registry: Any) -> dict[str, Any]:
    """The arguments `choose_configuration` can actually be given today.

    **Three of spec §8.2's four rows: `bound`, `producer`, `global_`.** Only
    `consumer` has no source, and `CONFIGURATION_SOURCES` above is the row-by-row
    account — that table is the one to read, and it is kept current because a
    reader trusts it.

    **This paragraph used to say "only `global_`", and it was three commits out
    of date** — `ec5fbba` landed `bound` and `0b64554` landed `producer`, and
    neither updated the prose beside them. It was not harmless: `demo` asked
    which row their validator body had been given, and this docstring is where
    the wrong answer came from. They had to spend a probe to find that
    `ConfigSource.PRODUCER` had fired all along. A stale comment is a claim, and
    a claim next to code outranks the code for anyone reading rather than
    executing.

    A row that is still absent needs a route, not a `getattr`: `consumer` needs
    the consuming task's resolved configuration, which is `env_mgr`'s
    `Prepared.environment` rather than anything on `Task` — confirmed by
    `env_mgr`, who also confirmed that `EnvManager` keeps no per-task state.

    **The two rows are not one question, and reading them as one was mine to
    correct.** They are the same *task* — both phases run inside `TaskRunner` for
    one task — but not the same *moment*, and the configuration exists at one and
    not the other. Measured: `env.prepare` has exactly one call site,
    `agent/runner.py:668`, reached from `_deploy` inside `_main`, and
    `_one_phase` reaches `_main` only in `RUNNING`. So at `INPUT_VALIDATING`
    there is no `Prepared` for this task at all.
    """
    return {
        "bound": _bound_environment(spec, registry),
        "producer": _producer_environment(task, registry),
        "global_": registry.get("validation_env") if "validation_env" in registry else None,
    }


def _producer_environment(task: Any, registry: Any) -> Mapping[str, str] | None:
    """§8.2 row 3 — the configuration the task that just ran resolved.

    Built by `agent` on request (`3155ca2`): `TaskAttempt.environment` is a
    read-only `Mapping[str, str]`, carried from `_deploy` onwards, and
    `attempt_of(task.id)` is the handle. **`choose_configuration` uses it on the
    output phase only**, which is where the chain's own `kind` test puts it —
    this function reports what exists and does not decide the row.

    **A configuration is not an environment**, and `agent` wrote the same
    sentence beside their property because the two words collide here. Spec §8.2
    is *"reusing a configuration is fine; inheriting an environment or a
    conversation is not"*, and criterion 21 tests that a validation environment
    is a rebuild. This returns a mapping of strings and nothing that grants a
    zone, a handle or a conversation.

    **Empty reads as absent, and that is a decision rather than a coincidence.**
    `agent` is explicit that `{}` means *has not deployed* — the same shape as
    `executor is None`, and for a **non-leaf** it stays that way, because the
    scheduler runs a non-leaf's main phase by unfolding and nothing ever calls
    `prepare`. §8.2's row is *the configuration already resolved*, so a task that
    resolved none must fall through to the global row. Passing `{}` on would
    select `PRODUCER` and hand the validation an empty environment — a value that
    is wrong but not type-wrong, which is the family `interfaces.md` §4.11 names.

    **The `getattr` is guarded rather than defaulted, and it is the kind this
    package removed everywhere else.** The difference is that this default *can*
    fire: the component registered as `runner` is not one protocol. `task_graph`
    registers the shipped `FakeRunner`, which has `start` and `stop` and no
    attempts at all, while `agent.Runner` declares `attempt_of`
    (`agent/protocols.py:313`). A runner with no attempts has no resolved
    configuration to report, which is an answer and not a missing field.
    """
    if "runner" not in registry:
        return None
    attempt_of = getattr(registry.get("runner"), "attempt_of", None)
    if attempt_of is None:
        return None
    attempt = attempt_of(task.id)
    found = getattr(attempt, "environment", None) if attempt is not None else None
    return found or None


def _bound_environment(spec: ValidatorSpec, registry: Any) -> Mapping[str, str] | None:
    """§8.2 row 1 — the `env` of the agent spec this validator names.

    **Absent and unresolvable are different questions and do not share an
    answer**, which is `closure`'s correction of a conflation of mine. Absent is
    the declared way to take the global row and returns `None` quietly.
    Unresolvable **raises**: `closure`'s pass makes such a name fatal at load, so
    reaching one here means that check did not run, and falling back would give
    the author a *working* environment that is not the one they configured — the
    failure the fatal check exists to prevent, arriving at run time instead.

    The symptom of the silent version is the bad one: a validator that **runs**,
    in the wrong environment, producing a verdict somebody trusts.
    """
    name = spec.agent
    if name is None:
        return None
    if "agent_specs" not in registry:
        raise ValidatorInvalid(
            f"{spec.name} names agent spec {name!r} and no `agent_specs` is "
            f"registered; §8.2 row 1 cannot be resolved"
        )
    specs = registry.get("agent_specs")
    if name not in specs:
        raise ValidatorInvalid(
            f"{spec.name} names agent spec {name!r}, which does not resolve; "
            f"known: {specs.names()}. `closure`'s pass should have rejected this "
            f"at load — reaching it here means the catalogue check did not run"
        )
    env = specs.get(name).get("env")
    return dict(env) if env else None


def _relative_to(path: str, zone: Path) -> str:
    """A zone-relative path where possible, absolute where not.

    `os.path.relpath` would happily produce `../../elsewhere` for a staged path
    outside the zone's tree, and a body resolving that is reaching out of its own
    working directory on the strength of a computed string. An absolute path at
    least says what it is.
    """
    try:
        return str(Path(path).relative_to(zone))
    except ValueError:
        return str(Path(path))


def read_verdict_file(
    env: ValidationEnvironment, declared: Iterable[HandoffId]
) -> dict[HandoffId, bool]:
    """One verdict per declared handoff. A missing entry raises.

    `dict.get` would yield `None`, and `None` folded as falsy is indistinguishable
    from a genuine `False` — DeepEval's unreached DAG node reports identically to
    a real zero, with no signal that the graph terminated early.

    **Every way a body can fail to produce a usable verdict raises
    `ValidatorInvalid`**, and that is a promise `monitor` depends on: their
    `VALIDATION_UNREACHED` is *"its `entry.sh` crashed, its agent died — nothing
    was decided"*, and `agent.Runner` catches **any** exception out of
    `run_phase` to report it — the broad form, ruled on by `monitor` after this
    docstring first claimed a single type. Do not narrow it to an inventory of
    what this module raises: theirs would then exclude *"its own inputs were
    missing"*, which is a `KeyError` from `handoff_mgr`, and an inventory goes
    stale — this one went stale within the hour of being written. Measured
    (`scratch/impl-2026-08/validator/p5_what_escapes_run_phase.py`), two ways out
    of five escaped as something else: malformed JSON left as a
    `json.JSONDecodeError`, and a body writing `null` as a `TypeError` from
    `"x" in None`. Both are a body producing garbage — the exact case — and both
    would have been reported as *the monitor's own handler raised*, which routes
    to `GiveUp` instead of escalating. **A crashed validator was the quietest dead
    branch in the system.**
    """
    if not env.verdict_file.exists():
        raise ValidatorInvalid(f"no verdict at {env.verdict_file}")
    try:
        raw = json.loads(env.verdict_file.read_text())
    except ValueError as exc:  # JSONDecodeError, and anything else json raises
        raise ValidatorInvalid(f"{env.verdict_file} is not readable as JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValidatorInvalid(
            f"{env.verdict_file} holds {type(raw).__name__}, not an object of handoff id to boolean"
        )
    out: dict[HandoffId, bool] = {}
    for hid in declared:
        if str(hid) not in raw:
            raise ValidatorInvalid(f"{env.verdict_file}: no verdict for {hid}")
        out[hid] = bool(raw[str(hid)])
    return out


def _attributed_agent(validator: Any) -> Any:
    """The `AgentId` that actually ran the check, or `None` if none did.

    Criterion 10 is about the verdict being attributable to somebody **other than
    the producer**, and until `agent`'s `validator_executor` existed this module
    wrote `task.current.agent_id` — *the producing agent* — into every verdict. A
    record saying the producer's own agent validated the artefact is the exact
    claim §8.1 forbids, so this reads the id the executor minted instead.

    A composite is one verdict over several bodies, so its members may report
    several ids; the first is taken and that is a simplification, not a decision —
    `handoff.Verdict.agent_id` is a single required field and there is nowhere to
    put the rest.

    **`None` for a script body**, which has no agent at all, and `None` is what
    the verdict now carries. `handoff` widened `Verdict.agent_id` to
    `AgentId | None` rather than take a sentinel — a sentinel is a UUID, so a
    reader who does not know it takes it for a real agent and one who looks it up
    finds nothing, which is a plausible value flowing on undetected in the field
    whose entire purpose is attribution.
    """
    own = getattr(validator, "ran_as", None)
    if own is not None:
        return own
    for member in getattr(validator, "members", ()):
        reported = getattr(member, "ran_as", None)
        if reported is not None:
            return reported
    return None


class LeafValidator:
    """A spec plus a rebuilt environment, in the shape `Validator` declares.

    Satisfies the Protocol structurally so that a composite may hold one and
    `run_phase` cannot tell a leaf from a composite.
    """

    def __init__(
        self,
        spec: ValidatorSpec,
        env: ValidationEnvironment,
        registry: Any,
        *,
        package_root: Path | None,
    ) -> None:
        self.name = spec.name
        self.brief = spec.brief
        self.inputs = spec.inputs
        self.dimension = spec.dimension
        self.strength = spec.strength
        self._spec = spec
        self._env = env
        self._registry = registry
        self._body = runner_for(spec, package_root)
        #: The agent that actually ran this body, reported by the executor.
        #: `None` for a script body, which has no agent — see `_attributed_agent`.
        self.ran_as: Any = None

    def __call__(self, handoffs: Mapping[HandoffId, Any]) -> dict[HandoffId, bool]:
        # An agent-bodied runner returns the `AgentId` it minted; a script body
        # returns `None`, having no agent. Declared `-> None` and still declared
        # so — the return is additive, the caller may ignore it, and it is the
        # only route by which the real identity can reach a verdict, because
        # `env` is frozen and built before the body runs.
        self.ran_as = self._body.run(self._spec, self._env, handoffs, self._registry)
        return read_verdict_file(self._env, handoffs)


# --------------------------------------------------------------------------- #
# The phase runner


class PhaseRunner:
    """The two validation phases. Registered once as `phase_runner`; called twice
    per dispatch, by `agent.Runner`, around the main phase.

    `strict_level` is bound at construction because it is a run-wide policy — it
    arrives from `--validation-strict-level` and cannot change mid-run without
    changing what criterion 20 means. `registry` is passed per call because that
    is how the phase reaches the managers at the moment it needs them; binding it
    instead would make this object hold a collaborator handle across dispatches.

    **One method, and it stays one.** A second — `configure`, `set_level` — would
    put the knob back inside the object the knob is not allowed to reach.

    `zone_root` is keyword-only and optional, so the declared
    `PhaseRunner(strict_level)` call in `docs/interfaces.md` §4.3 is unchanged. It
    exists because a validation zone has to be allocated somewhere and no
    component in the composition root owns a place to put one.
    """

    def __init__(
        self,
        strict_level: StrictLevel,
        *,
        zone_root: Path | None = None,
        package_root: Path | None = None,
    ) -> None:
        self.strict_level = strict_level
        self._zone_root = zone_root
        # **Not `or Path.cwd()`.** A body path is package-relative, so falling
        # back to the working directory resolves it against wherever the process
        # happened to start — a plausible path that is nobody's package. It
        # either fails as a puzzling "does not resolve" or, worse, finds a
        # different file of the same name. `interfaces.md` §4.11's family again:
        # a value that is wrong but not type-wrong, so nothing raises.
        #
        # It stays optional because §4.3 declares `PhaseRunner(strict_level)`, so
        # the absence is refused at the point a body actually needs resolving —
        # loud, and naming what to supply.
        self._package_root = package_root

    def run_phase(self, kind: PhaseKind, task: Any, registry: Any) -> PhaseOutcome:
        # Coerced at the boundary, and this is load-bearing rather than tidy.
        # `agent.Runner` is the only caller and it **cannot name `PhaseKind`**:
        # `interfaces.md` §4.4 gives `agent` only spec_loader / task_graph /
        # monitor, so there is no import through which the enum member could
        # reach it, and it passes the value `"input_validation"` instead.
        # Everything below compares with `is`, which a bare string fails — and
        # the first thing it would fail is `_handoffs`, silently validating the
        # task's *outputs* during the input phase. One coercion here rather than
        # five `==` comparisons is the smaller surface: a value that is not a
        # phase raises `ValueError` naming it, instead of taking a wrong branch.
        kind = PhaseKind(kind)
        selected = self._select(kind, task, registry)

        # **The two "nothing was asked of this phase" cases come first**, and the
        # order is load-bearing since `interfaces.md` §4.15: everything that
        # reaches `fold` below is a phase verdicts were expected of, so an empty
        # output fold is a fault. Getting the order wrong makes the fault arm
        # swallow both of these — and the second one is the demo's **root** task.
        if history.phase_is_switched_off(self.strict_level):
            return PhaseOutcome.nothing_expected(
                kind,
                skipped=[
                    SkipRecord(s.name, "phase switched off by --validation-strict-level", None)
                    for s in selected
                ],
            )

        # `_handoffs`, not `_targets`: this is a read of `task.inputs`/`.outputs`
        # and asks no manager for a version. A task with nothing in this position
        # produced nothing here, so there is nothing unchecked — and it must be
        # answered independently of `selected`, because `consume` declares a
        # validator *and* `outputs: []`.
        if not self._handoffs(kind, task):
            return PhaseOutcome.nothing_expected(kind)

        if not selected:
            # NOT a pass, and since §4.15 **not merely empty either**: this task
            # has a handoff in this position and nothing is bound to check it.
            # On the output phase that is the fault, and it blocks.
            return PhaseOutcome.fold(kind)

        targets = self._targets(kind, task, registry)
        store = registry.get("handoff_store")
        ran: list[VerdictRecord] = []
        reused: list[VerdictRecord] = []
        skipped: list[SkipRecord] = []

        for spec in selected:
            mine = [t for t in targets if self._kind_of(t, registry) in spec.inputs]
            if not mine:
                continue
            prior = history.priors(store, mine, spec.name)
            if history.may_skip(prior, self.strict_level):
                assert prior is not None
                reused.extend(prior)
                skipped.append(
                    SkipRecord(spec.name, "already validated against this version", prior[0])
                )
                continue
            ran.extend(self._invoke(kind, spec, task, mine, registry))

        return PhaseOutcome.fold(kind, ran=ran, reused=reused, skipped=skipped)

    # ----------------------------------------------------------------- select

    def _select(self, kind: PhaseKind, task: Any, registry: Any) -> list[ValidatorSpec]:
        """The bound validators for this phase, **cheap-first**.

        Ordering by a declared `cost` tag has no prior art in anything surveyed —
        every system that consumes a cost signal does admission control (Bazel's
        `size` never appears in the ordering path), measured bin-packing
        (pytest-split), or a human-declared dependency graph (CI `needs`). So the
        design owes the failure mode rather than a citation: **the tag can be
        wrong and nothing here detects it.**
        """
        specs = registry.get("validator_specs")
        selected = [specs.spec(n) for n in self._bound(task, registry) if n in specs]
        return sorted(selected, key=lambda s: (s.cost.rank, s.name))

    @staticmethod
    def _bound(task: Any, registry: Any) -> Sequence[str]:
        """Every validator this task's closure says will run. **Asked, not derived.**

        This module used to build the set itself, from each handoff kind's own
        `validators` list — and the closure's `validators`, which
        `closure.schema.json` calls *"the PHASE validators… a property of the task
        rather than of any one handoff kind, **which is why the handoff specs
        cannot carry them**"*, was read by nothing in the tree. A closure
        declaring `validators: ['check_grounded']` therefore ran nothing. `demo`
        found it on the first assembly of all eight.

        Reading **both** lists here was the narrow fix and it is not the one
        taken: `closure` already computes the union — *"every validator that will
        run, the phase validators plus the per-handoff ones joined through the
        handoff registry"* — so doing it here would make this module a second
        computer of somebody else's fact. `engineer_principle.md` §4.4: when you
        seem to need another module's property, ask what you intend to compute
        with it and request **that** instead.

        A task with no `closure` has no declared set, and the phase folds to
        `empty` — which is reported and is **not** a pass, so nothing is silently
        admitted. That is the third state doing its job rather than a gap.
        """
        # `task.closure`, not `getattr(task, "closure", None)`. The field is
        # guaranteed on `task_graph.Task`, so the default could never fire — and
        # a default that cannot fire is how `environment` hid: if the field is
        # ever removed, a `getattr` turns a loud `AttributeError` into a silent
        # `None` and this phase quietly runs nothing. `closure` may *be* `None`,
        # which is a value the field carries rather than an absence of it.
        name = task.closure
        if not name:
            return ()
        return tuple(registry.get("closures").validators_for(name))

    @staticmethod
    def _handoffs(kind: PhaseKind, task: Any) -> Sequence[HandoffId]:
        return list(task.inputs if kind is PhaseKind.INPUT else task.outputs)

    @staticmethod
    def _kind_name(hid: HandoffId, registry: Any) -> str:
        return registry.get("handoff_mgr").get(hid).type

    def _targets(self, kind: PhaseKind, task: Any, registry: Any) -> list[Target]:
        """Which version of each handoff this phase checks.

        **Every number here is a `handoff` *store* version — the directory the
        bytes are in — and never `HandoffMgr`'s *slot* version.** They are two
        counters with no owned reference between them (`interfaces.md` §5.12)
        and everything downstream of a `Target` wants the first:
        `store.read_verdicts`, `store.record_verdict` and `history.priors` all
        resolve `<store>/<hid>/v<N>/`.

        An input's is the one the execution **pinned** at dispatch, not whatever
        is latest now.

        An output's used to be `handoff_mgr.get(hid).latest.version`, and that
        was the slot — a bug that could not show until a package existed whose
        **non-leaf declares an output**. The slot advances on every agent write,
        the store on every dispatch (`task_graph/scheduler.py::_pin_outputs`), so
        with one dispatch that does not write between them the two disagree and
        the phase reads a directory that is not the artefact. Measured on
        `scratch/demo2-2026-08/bringup/n1`: parent `main` pinned store v0, its
        end entry `directions` pinned store v1 and published there, the slot read
        0, and the output phase died on *"cannot read verdicts of … v0: it is not
        published (published: [1])"*.

        So an output's is **the store version this attempt published**, which
        `Execution.output_versions` carries — the symmetric field to the input's,
        and the same one `env_mgr` builds the write grant from.

        **A non-leaf pinned a version and did not write it, so its own pin is a
        hole.** `models.py::_instantiate` gives the end entry *the parent's own
        handoff id* (`mine.get(kind) if entry.is_end`), so one artefact has two
        tasks declaring it as an output and both get pinned; only the end entry
        writes. The parent holds no reference to the number its subgraph
        published — §5.12's gap, one level up — so the question it can ask is
        `handoff`'s: **which version is published?** `store.latest` answers it,
        filtered on the manifest, so an unpublished pin is invisible and a
        concurrent retry's hole cannot be selected.

        That is `store.latest`'s **first production caller**, and its docstring
        asked the first one to shape the contract with one warning: *"a caller
        that wants the version some earlier decision was about must carry that
        number itself."* This one does — that is the pinned branch — and falls
        back only when the number it carries names no published version, which
        is exactly the case where it did not write and has nothing to carry.
        """
        handoff_mgr = registry.get("handoff_mgr")
        # Direct, for the same reason as `task.closure` above: `input_versions`
        # is a guaranteed field on `Execution`, so the default was unreachable.
        pinned: Mapping[HandoffId, int] = task.current.input_versions or {}
        out: list[Target] = []
        for hid in self._handoffs(kind, task):
            if kind is PhaseKind.INPUT:
                version = pinned.get(hid)
            else:
                version = self._published(hid, task, registry)
            if version is None:
                version = handoff_mgr.get(hid).latest.version
            out.append(Target(hid, version))
        return out

    @staticmethod
    def _published(hid: HandoffId, task: Any, registry: Any) -> int | None:
        """The store version of `hid` this output phase is about, or `None`.

        `None` means no version of this handoff is published *and* this attempt
        pinned none — an output that was never written. `_targets` then keeps the
        pre-existing fallback rather than inventing an answer here, because the
        absence is already the gate's to report and this phase's job is only to
        say which directory it would have checked.

        The store is resolved here and not in `_targets` so that an input phase
        never needs it; `run_phase` requires it one line later either way.
        """
        mine = (task.current.output_versions or {}).get(hid)
        store = registry.get("handoff_store")
        if mine is not None and mine in store.list_versions(hid):
            return mine
        published = store.latest(hid)
        return published if published is not None else mine

    def _kind_of(self, target: Target, registry: Any) -> str:
        return self._kind_name(target.handoff_id, registry)

    # ----------------------------------------------------------------- invoke

    def _invoke(
        self,
        kind: PhaseKind,
        spec: ValidatorSpec,
        task: Any,
        targets: Sequence[Target],
        registry: Any,
    ) -> list[VerdictRecord]:
        """Build a fresh environment, run the body, persist one verdict per target.

        The environment is built **per validator, inside the loop** — rebuilt,
        never inherited (spec §8.2). There is no cost argument against it.
        """
        env = self._build_environment(kind, task, spec, registry)
        handoff_mgr = registry.get("handoff_mgr")
        handoffs = {t.handoff_id: handoff_mgr.get(t.handoff_id) for t in targets}

        validator = self._validator(spec, env, registry)
        results = validator(handoffs)

        store = registry.get("handoff_store")
        at = datetime.now(timezone.utc)
        checker = _attributed_agent(validator)
        records: list[VerdictRecord] = []
        for target in targets:
            verdict = Verdict(
                validator=spec.name,
                result=bool(results[target.handoff_id]),
                strength=Strength(spec.strength).value,
                dimension=spec.dimension.value,
                task_id=task.id,
                # `None` when no agent ran, which `handoff` made representable
                # (f9142aa) after this module had nowhere to put the truth. The
                # producer's id used to go here, and a record saying the producer
                # validated its own artefact is the claim §8.1 forbids. There is
                # no fallback and no `attributed` side-channel: the field a reader
                # consults says it directly.
                agent_id=checker,
                environment={"source": env.config.source.value, "zone": str(env.zone)},
                at=at,
            )
            store.record_verdict(target.handoff_id, target.version, verdict)
            records.append(
                VerdictRecord(verdict=verdict, handoff_id=target.handoff_id, version=target.version)
            )
        registry.get("validator_specs").record_run(spec.name, at=at)
        return records

    def _validator(self, spec: ValidatorSpec, env: ValidationEnvironment, registry: Any) -> Any:
        if not spec.is_composite:
            return LeafValidator(spec, env, registry, package_root=self._package_root)
        specs = registry.get("validator_specs")
        members = [
            LeafValidator(specs.spec(m), env, registry, package_root=self._package_root)
            for m in spec.members
        ]
        return Composite(
            spec.name,
            brief=spec.brief,
            dimension=spec.dimension,
            strength=spec.strength,
            members=members,
            reduce=get_reducer(spec.reduce or ""),
        )

    def _build_environment(
        self, kind: PhaseKind, task: Any, spec: ValidatorSpec, registry: Any
    ) -> ValidationEnvironment:
        config = choose_configuration(kind, **_configuration_sources(task, spec, registry))
        # Each phase is separately attributable. The requirement is this module's;
        # the mechanism is `agent` design O6's and is open.
        #
        # **This derivation is weaker than §8.1 needs, and saying so is the point.**
        # It is the *producing* task's agent id with the phase appended, so it is
        # distinct per phase and is NOT a distinct agent. Criterion 10 wants the
        # checking context to be one the producer cannot reach, and a suffix does
        # not buy that. What does is a fresh executor per phase, each bound to its
        # own `Agent` record and therefore its own system `AgentId` — which is what
        # `validator_executor` will supply when it exists (`agent` O6). Until then
        # this keeps the assertion live rather than letting a phase run
        # unattributed, and `test_no_producer_frame_reads_the_standard` asserts the
        # hook half, which does not depend on this string.
        agent_id = f"{task.current.agent_id}:{kind.value}"
        root, materials = self._placed_root(kind, task, registry)
        env = build_environment(root, config=config, agent_id=agent_id, standards=())
        _declare_materials(env, materials)
        return env

    def _placed_root(
        self, kind: PhaseKind, task: Any, registry: Any
    ) -> tuple[Path, tuple[str, ...]]:
        """Where this validation's zone goes, asked of `env_mgr` when it is wired.

        `prepare_validation` places the zone as a **sibling of the producing
        task's zone, never a descendant** — `env_mgr` design D5, and their
        criterion 13 is untrue without it, because anything under the producing
        task's directory is inside its subtree and reachable. It also stages
        **copies** of what this phase validates, which is what finally gives
        §5.8's body somewhere to point.

        Nothing on this module's side changes: `build_environment` allocates a
        fresh directory *inside* whatever root it is given, and a fresh directory
        inside a correctly-placed sibling is still a sibling. Freshness by
        allocation, `CHANNELS` and `assert_standard_unreachable` all hold as
        written.

        **`prepare_validation` confines nothing**, and that is a boundary rather
        than an omission: `prepare` applies Landlock to its own process, and a
        phase runner calling this is the supervisor. Who confines a validation
        *body* is `interfaces.md` §5.15, open with three candidates and no owner.
        So placement makes the standard unreachable by *where it is*, and nothing
        yet makes it unreachable by *what the kernel permits*.

        The fallback is the documented stand-in, not a second answer: without an
        `env_mgr` the zone lands under `zone_root` or a `mkdtemp`, which is
        outside the granted set and so separates by accident of location. That is
        why the module docstring calls it a stand-in.
        """
        if "env_mgr" in registry:
            placed = registry.get("env_mgr").prepare_validation(task, task.current, kind)
            return Path(placed.root), dict(placed.materials)
        return (self._zone_root or self._default_root()), {}

    def _default_root(self) -> Path:
        if self._zone_root is None:
            self._zone_root = Path(tempfile.mkdtemp(prefix="agent-sys-validation-"))
        return self._zone_root
