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
sees them together (`check_targets`, as `PhaseRunner` does), the sealing
pre-check, and the human report.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from env_mgr.fs.layout import copy_out
from skillkit import hstore
from skillkit.pkg import UNSET, Package
from task_graph.ids import HandoffId
from validator.composite import Composite
from validator.environment import build_environment, choose_configuration
from validator.phase import BODY_OUTPUT, LeafValidator, declare_materials
from validator.protocols import PhaseKind
from validator.reducers import get_reducer
from validator.spec import admit


@dataclass
class Outcome:
    validator: str
    result: bool | None  # None = UNREACHED
    strength: str
    dimension: str
    cost: str
    zone: str | None
    seconds: float
    report: str = ""
    output_tail: str = ""
    why_unreached: str = ""
    members: list[dict] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return {True: "PASS", False: "FAIL", None: "UNREACHED"}[self.result]


@dataclass
class KindCheck:
    kind: str
    content: str
    precheck: list[str]
    outcomes: list[Outcome]

    @property
    def passed(self) -> bool:
        """Sealable, at least one validator ran, and every one passed.

        No validator at all is a block, not a pass (`validator/report.py`:
        an output phase with nothing to run is "unchecked")."""
        return (
            not self.precheck
            and bool(self.outcomes)
            and all(o.result is True for o in self.outcomes)
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        for o, od in zip(self.outcomes, d["outcomes"]):
            od["verdict"] = o.verdict
        return d


def harness_env() -> dict[str, str]:
    try:
        from env_mgr.harness import harness_env as _h  # noqa: PLC0415

        return _h()
    except Exception:  # noqa: BLE001 — a missing/odd settings file must not stop a check
        return {}


def run_one(
    pkg: Package,
    name: str,
    targets: dict[str, Path],
    work: Path,
    *,
    source_env: dict[str, str] | None = None,
    store: Path | None = None,
    phase: str = "output_validation",
) -> dict[str, Outcome]:
    """Run validator `name` ONCE over every target `{hid: content dir}` with
    agent_sys's own machinery, and return one `Outcome` per hid.

    `validator.spec.admit` builds the spec; `validator.environment` chooses the
    configuration row (`bound` if the validator names an agent, else
    `source_env` — the producer's environment for an output check, as the
    runtime passes it) and allocates the zone; the contents are staged as
    COPIES (`env_mgr.fs.layout.copy_out`) and declared with
    `validator.phase.declare_materials`; `LeafValidator` (or `Composite`) runs
    the body and reads the verdict. Like the runtime, there is no per-validator
    timeout.
    """
    hids = list(targets)

    def every(**kw) -> dict[str, Outcome]:
        return {h: Outcome(**kw) for h in hids}

    record = pkg.validators.get(name)
    if record is None:
        return every(
            validator=name,
            result=None,
            strength="?",
            dimension="?",
            cost="?",
            zone=None,
            seconds=0.0,
            why_unreached=f"no validator {name!r}",
        )
    tags = record.get("tags") or {}
    base = dict(
        validator=name,
        strength=str(record.get("strength", "?")),
        dimension=str(record.get("dimension", "?")),
        cost=str(tags.get("cost", "?")),
    )
    if UNSET in json.dumps(record.get("args") or {}):
        return every(
            **base,
            result=None,
            zone=None,
            seconds=0.0,
            why_unreached="its args use a --var that was not given (filled with "
            f"{UNSET!r}); pass the real value to judge with this validator",
        )
    try:
        spec = admit(record, origin=f"package validator {name}")
        members = [
            admit(pkg.validators[m], origin=f"package validator {m}") for m in spec.members or ()
        ]
    except Exception as exc:  # noqa: BLE001 — an inadmissible spec decides nothing
        return every(
            **base, result=None, zone=None, seconds=0.0, why_unreached=f"inadmissible: {exc}"
        )

    facts = {
        **harness_env(),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "AGENT_SYS_DEMO_PACKAGE": str(pkg.root),
        "AGENT_SYS_DEMO_PYTHON": sys.executable,
        **({"AGENT_SYS_DEMO_STORE": str(store)} if store is not None else {}),
    }
    bound = None
    if spec.agent:
        bound = {
            str(k): str(v) for k, v in (pkg.agents.get(spec.agent, {}).get("env") or {}).items()
        }
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
    staged = {}
    for n, (hid, content) in enumerate(targets.items()):
        staged[HandoffId(hid)] = copy_out(str(content), str(env.zone / "materials" / str(n)))
    declare_materials(env, staged)
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
        result = validator({h: None for h in staged})
    except Exception as exc:  # noqa: BLE001 — ValidatorInvalid and friends: nothing was decided
        why = str(exc)
    seconds = round(time.monotonic() - t0, 2)
    report_path = env.zone / "validator_report.txt"
    report = report_path.read_text(errors="replace")[-8000:] if report_path.is_file() else ""
    printed = env.zone / BODY_OUTPUT
    tail = printed.read_text(errors="replace")[-4000:] if printed.is_file() else ""
    out = {}
    for hid in hids:
        value = None if result is None else result.get(HandoffId(hid))
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
) -> dict[str, KindCheck]:
    """Judge several kinds at once: `{kind: (hid, content dir)}`.

    Seal pre-check per kind; then every validator bound to any of them runs
    once, over every sealable target whose kind it lists in `inputs`."""
    checks: dict[str, KindCheck] = {}
    for kind, (_hid, content) in targets.items():
        spec = pkg.kind(kind)
        problems = hstore.precheck(content, str(spec["content_type"]), spec.get("items_schema"))
        checks[kind] = KindCheck(kind=kind, content=str(content), precheck=problems, outcomes=[])
    if precheck_only:
        return checks
    ready = {k: t for k, t in targets.items() if not checks[k].precheck}
    names: list[str] = []
    for kind in ready:
        names += [n for n in pkg.validators_for(closure, kind) if n not in names]
    names.sort(key=lambda n: (pkg.cost_rank(n), n))
    for name in names:
        wants = set(pkg.validators.get(name, {}).get("inputs") or [])
        mine = {k: t for k, t in ready.items() if k in wants}
        if not mine:
            continue
        res = run_one(
            pkg,
            name,
            {hid: content for hid, content in mine.values()},
            work,
            source_env=source_env,
            store=store,
        )
        for kind, (hid, _content) in mine.items():
            checks[kind].outcomes.append(res[hid])
    return checks


def check_kind(
    pkg: Package,
    kind: str,
    content_dir: Path,
    work: Path,
    *,
    closure: str | None = None,
    hid: str | None = None,
    source_env: dict[str, str] | None = None,
    store: Path | None = None,
    precheck_only: bool = False,
) -> KindCheck:
    hid = hid or str(uuid.uuid4())  # agent_sys ids are UUIDs
    return check_targets(
        pkg,
        {kind: (hid, content_dir)},
        work,
        closure=closure,
        source_env=source_env,
        store=store,
        precheck_only=precheck_only,
    )[kind]


def rows_for_store(check: KindCheck, task_id: str) -> list[dict]:
    """`validation.yaml` rows (agent_sys's 8 fields) for every decided verdict."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [
        {
            "validator": o.validator,
            "result": o.result,
            "strength": o.strength,
            "dimension": o.dimension,
            "task_id": hstore.task_id(task_id),
            "agent_id": None,
            "environment": {
                "source": "skillkit (Claude Code, not the agent_sys runtime)",
                "task_label": task_id,
                "zone": o.zone,
            },
            "at": now,
        }
        for o in check.outcomes
        if o.result is not None
    ]


def render(check: KindCheck) -> str:
    """The human report: every verdict, verbatim reasons, what to read next."""
    lines = [f"== {check.kind}  ({check.content})"]
    if check.precheck:
        lines.append("  SEAL REFUSED — fix these first (validators were not run):")
        lines += [f"    - {p}" for p in check.precheck]
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
        for m in o.members:
            lines.append(f"      member {m['validator']}: {m['result']}")
    lines.append(f"  => {'PASSED' if check.passed else 'NOT PASSED'}")
    return "\n".join(lines)
