"""Run a handoff kind's validators offline, with agent_sys's validator code.

`run_one` builds each validator through `validator.spec.admit`, chooses its
environment with `validator.environment.choose_configuration` (the producer's
environment for an output check, as the runtime does) and allocates the zone
with `build_environment`, stages COPIES of the contents, declares them with
`validator.phase.declare_materials`, and runs the body with `LeafValidator` /
`Composite`, which also read `verdict.json`. A body that writes no verdict —
or an agent-bodied validator, for which production registers no executor — is
UNREACHED ("nothing was decided"), with the runtime's own message.

What is this module's: grouping a leaf's outputs so a multi-input validator
sees them together (`check_targets`, as `PhaseRunner` does), the phase-wide
fold over them (`validator/report.py`), the sealing pre-check, and the human
report.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

from env_mgr.fs.layout import copy_out
from env_mgr.harness import harness_env
from handoff.protocols import Verdict
from skillkit import hstore
from skillkit.pkg import UNSET, Package
from task_graph.ids import HandoffId, TaskId
from validator.composite import Composite
from validator.environment import build_environment, choose_configuration
from validator.phase import BODY_OUTPUT, LeafValidator, declare_materials
from validator.protocols import PhaseKind
from validator.reducers import get_reducer
from validator.spec import admit


@dataclass
class Outcome:
    validator: str
    strength: str
    dimension: str
    cost: str
    result: bool | None = None  # None = UNREACHED
    zone: str | None = None
    seconds: float = 0.0
    report: str = ""
    output_tail: str = ""
    why_unreached: str = ""

    @property
    def verdict(self) -> str:
        return {True: "PASS", False: "FAIL", None: "UNREACHED"}[self.result]


@dataclass
class KindCheck:
    kind: str
    content: str
    precheck: list[str]
    outcomes: list[Outcome]
    #: Another output of the same phase ran a validator, so this kind having
    #: none of its own does not make the phase "unchecked" (`validator/report.py`
    #: folds per phase, not per output).
    covered: bool = False
    #: Why it was not sealed although its own checks passed: the phase it
    #: belongs to did not pass as a whole.
    held: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Sealable, every verdict on it a pass, and something ran in its phase.

        No validator at all is a block, not a pass (`validator/report.py`:
        an output phase with nothing to run is "unchecked")."""
        return (
            not self.precheck
            and not self.held
            and (bool(self.outcomes) or self.covered)
            and all(o.result is True for o in self.outcomes)
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        for o, od in zip(self.outcomes, d["outcomes"]):
            od["verdict"] = o.verdict
        return d


def run_one(
    pkg: Package,
    name: str,
    targets: Mapping[str, tuple[str, Path]],
    work: Path,
    *,
    source_env: dict[str, str] | None = None,
    store: Path | None = None,
    phase: str = "output_validation",
) -> dict[str, Outcome]:
    """Run validator `name` ONCE over every target `{kind: (hid, content dir)}`
    with agent_sys's own machinery (module docstring), and return one `Outcome`
    per hid. The environment row is `bound` if the validator names an agent,
    else `source_env` — the producer's environment for an output check, as the
    runtime passes it. A composite hands each member the handoffs whose kind it
    lists, read from the slot's `.type` as the runtime's are. Like the runtime,
    there is no per-validator timeout.
    """
    hids = [hid for hid, _content in targets.values()]
    record = pkg.validators.get(name, {})
    base = dict(
        validator=name,
        strength=str(record.get("strength", "?")),
        dimension=str(record.get("dimension", "?")),
        cost=str((record.get("tags") or {}).get("cost", "?")),
    )

    def unreached(why: str) -> dict[str, Outcome]:
        return {h: Outcome(**base, why_unreached=why) for h in hids}

    if not record:
        return unreached(f"no validator {name!r}")
    if UNSET in json.dumps(record.get("args") or {}):
        return unreached(
            f"its args use a --var that was not given (filled with {UNSET!r}); "
            "pass the real value to judge with this validator"
        )
    try:
        spec = admit(record, origin=f"package validator {name}")
        members = [
            admit(pkg.validators[m], origin=f"package validator {m}") for m in spec.members or ()
        ]
    except Exception as exc:  # noqa: BLE001 — an inadmissible spec decides nothing
        return unreached(f"inadmissible: {exc}")

    try:
        operator = harness_env()
    except ValueError:  # a settings file that does not parse must not stop a check
        operator = {}
    facts = {
        **operator,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "AGENT_SYS_DEMO_PACKAGE": str(pkg.root),
        "AGENT_SYS_DEMO_PYTHON": sys.executable,
        **({"AGENT_SYS_DEMO_STORE": str(store)} if store is not None else {}),
    }
    # `validator/phase.py::_bound_environment`: no agent, or an agent with no
    # `env`, takes the next row; an agent that does not resolve is refused, never
    # run in an environment nobody configured.
    bound = None
    if spec.agent:
        if spec.agent not in pkg.agents:
            return unreached(
                f"{name} names agent spec {spec.agent!r}, which does not resolve; known: "
                f"{sorted(pkg.agents)}. agent_sys refuses this package at load (`astk show`)"
            )
        agent_env = (pkg.agents[spec.agent] or {}).get("env")
        bound = {str(k): str(v) for k, v in agent_env.items()} if agent_env else None
    kind = PhaseKind(phase)
    config = choose_configuration(
        kind,
        bound=bound,
        producer={**facts, "AGENT_SYS_TASK_PACKAGE": str(pkg.root), **(source_env or {})}
        if kind is PhaseKind.OUTPUT
        else None,
        consumer={**facts, **(source_env or {})}
        if kind is PhaseKind.INPUT and source_env
        else None,
        global_=facts,
    )
    root = work / name
    env = build_environment(root, config=config, agent_id=f"skillkit:{name}:{kind.value}")
    staged = {
        HandoffId(hid): copy_out(str(content), str(env.zone / "materials" / str(n)))
        for n, (hid, content) in enumerate(targets.values())
    }
    declare_materials(env, staged)
    slots = {HandoffId(hid): SimpleNamespace(type=kind) for kind, (hid, _c) in targets.items()}
    if spec.is_composite:
        validator = Composite(
            spec.name,
            brief=spec.brief,
            dimension=spec.dimension,
            strength=spec.strength,
            members=[LeafValidator(m, env, {}, package_root=pkg.root) for m in members],
            reduce=get_reducer(spec.reduce or ""),
        )
    else:
        validator = LeafValidator(spec, env, {}, package_root=pkg.root)

    t0 = time.monotonic()
    result: dict | None = None
    why = ""
    try:
        result = validator(slots)
    except Exception as exc:  # noqa: BLE001 — ValidatorInvalid and friends: nothing was decided
        why = str(exc)
    seconds = round(time.monotonic() - t0, 2)
    report_path = env.zone / "validator_report.txt"
    report = report_path.read_text(errors="replace")[-8000:] if report_path.is_file() else ""
    printed = env.zone / BODY_OUTPUT
    tail = printed.read_text(errors="replace")[-4000:] if printed.is_file() else ""
    out = {}
    for hid in hids:
        value = (result or {}).get(HandoffId(hid))
        out[hid] = Outcome(
            **base,
            result=None if value is None else bool(value),
            zone=str(env.zone),
            seconds=seconds,
            report=report,
            output_tail=tail,
            why_unreached=why,
        )
    return out


def check_targets(
    pkg: Package,
    targets: dict[str, tuple[str, Path]],
    work: Path,
    *,
    closure: str | None = None,
    source_env: dict[str, str] | None = None,
    store: Path | None = None,
    precheck_only: bool = False,
    bound: list[str] | None = None,
) -> dict[str, KindCheck]:
    """Judge several kinds at once: `{kind: (hid, content dir)}`.

    Seal pre-check per kind; then every validator bound to any of them runs
    once, over every sealable target whose kind it lists in `inputs`. `bound`
    is the phase's validator set (`Package.bound_validators`, as the runtime's
    closure index computes it); without it, each kind's own plus `closure`'s.

    When more than one kind is judged, they are one phase: a kind no validator
    lists is `covered` if something ran on another one (`validator/report.py`
    folds per phase)."""
    checks: dict[str, KindCheck] = {}
    for kind, (_hid, content) in targets.items():
        spec = pkg.kind(kind)
        problems = hstore.precheck(content, str(spec["content_type"]), spec.get("items_schema"))
        checks[kind] = KindCheck(kind=kind, content=str(content), precheck=problems, outcomes=[])
    if precheck_only:
        return checks
    ready = {k: t for k, t in targets.items() if not checks[k].precheck}
    if bound is None:
        bound = [n for kind in ready for n in pkg.validators_for(closure, kind)]
    for name in pkg.cheap_first(bound):
        wants = set(pkg.validators.get(name, {}).get("inputs") or [])
        mine = {k: t for k, t in ready.items() if k in wants}
        if not mine:
            continue
        res = run_one(pkg, name, mine, work, source_env=source_env, store=store)
        for kind, (hid, _content) in mine.items():
            checks[kind].outcomes.append(res[hid])
    if len(checks) > 1 and any(c.outcomes for c in checks.values()):
        for c in checks.values():
            c.covered = True
    return checks


def phase_passed(checks: Mapping[str, KindCheck]) -> bool:
    """`PhaseOutcome.passed` over one leaf's outputs: something ran, nothing
    failed or went unreached, and every output is sealable."""
    return bool(checks) and all(c.passed for c in checks.values())


def check_kind(
    pkg: Package, kind: str, content_dir: Path, work: Path, *, hid: str | None = None, **kw
) -> KindCheck:
    """`check_targets` on one kind; `hid` defaults to a fresh UUID (agent_sys ids are UUIDs)."""
    return check_targets(pkg, {kind: (hid or str(uuid.uuid4()), content_dir)}, work, **kw)[kind]


def verdicts(outcomes: list[Outcome], label: str) -> list[Verdict]:
    """The `validation.yaml` records of every decided outcome, as `label`'s."""
    now = dt.datetime.now(dt.timezone.utc)
    return [
        Verdict(
            validator=o.validator,
            result=o.result,
            strength=o.strength,
            dimension=o.dimension,
            task_id=TaskId(hstore.task_id(label)),
            agent_id=None,
            environment={
                "source": "skillkit (Claude Code, not the agent_sys runtime)",
                "task_label": label,
                "zone": o.zone,
            },
            at=now,
        )
        for o in outcomes
        if o.result is not None
    ]


def render(check: KindCheck) -> str:
    """The human report: every verdict, verbatim reasons, what to read next."""
    lines = [f"== {check.kind}  ({check.content})"]
    if check.precheck:
        lines.append("  SEAL REFUSED — fix these first (validators were not run):")
        lines += [f"    - {p}" for p in check.precheck]
    elif not check.outcomes and check.covered:
        lines.append("  no validator lists this kind; the leaf's other verdicts decide its phase.")
    elif not check.outcomes:
        lines.append("  UNCHECKED — no validator bound to this kind; agent_sys blocks this.")
    for o in check.outcomes:
        lines.append(
            f"  [{o.verdict}] {o.validator}  ({o.dimension}/{o.strength}/{o.cost}, {o.seconds}s)"
        )
        if o.why_unreached:
            lines.append(f"      unreached: {o.why_unreached}")
        if o.result is not True:
            body = o.report.strip() or o.output_tail.strip()
            for ln in body.splitlines()[-40:]:
                lines.append(f"      | {ln}")
        if o.zone:
            lines.append(f"      zone: {o.zone}")
    lines += [f"  NOT SEALED — {h}" for h in check.held]
    lines.append(f"  => {'PASSED' if check.passed else 'NOT PASSED'}")
    return "\n".join(lines)
