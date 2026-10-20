"""Run a handoff kind's validators offline, as `validator/phase.py` would.

Each validator gets a fresh zone as its cwd holding `inputs.json`,
`args.json` and `materials.json`, an environment that is NOT inherited
(`PATH`, `HOME`, `TMPDIR`, `PWD`, the `AGENT_SYS_DEMO_*` trio, the producer's
`AGENT_SYS_TASK_PACKAGE`, and the operator's harness credentials), and must
write `verdict.json`. Its stdout is kept here (the runtime discards it) because
a human reading a refusal needs it; `validator_report.txt` is read when present.

A body that writes no verdict is UNREACHED — "nothing was decided" — which is
neither a pass nor a fail, exactly as the runtime reports it. An agent-bodied
validator (no `entry`) is UNREACHED too: production registers no executor
for one (`validator/phase.py`), so pretending to run it would be a lie.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from skillkit import hstore
from skillkit.pkg import UNSET, Package

DEFAULT_TIMEOUT = 4 * 3600.0


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
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Outcome]:
    """Run validator `name` ONCE over every target `{hid: content dir}` — as
    `validator/phase.py` does for a validator whose `inputs` span several of a
    task's kinds — and return one `Outcome` per hid.

    Each zone holds COPIES of the contents (`materials/<n>/`), never the store's
    own `content/`: a validator must not be able to change what gets published.
    `source_env` is the configuration row agent_sys would choose: the producing
    leaf's environment for an output check; a validator naming an `agent:` gets
    that agent's `env:` instead (the BOUND row)."""
    spec = pkg.validators.get(name)
    hids = list(targets)

    def every(**kw) -> dict[str, Outcome]:
        return {h: Outcome(**kw) for h in hids}

    if spec is None:
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
    tags = spec.get("tags") or {}
    base = dict(
        validator=name,
        strength=str(spec.get("strength", "?")),
        dimension=str(spec.get("dimension", "?")),
        cost=str(tags.get("cost", "?")),
    )
    members = spec.get("members") or []
    if members:
        subs = [
            run_one(pkg, m, targets, work, source_env=source_env, store=store, timeout=timeout)
            for m in members
        ]
        out = {}
        for h in hids:
            results = [sub[h].result for sub in subs]
            result = None if None in results else all(results)
            out[h] = Outcome(
                **base,
                result=result,
                zone=None,
                seconds=sum(sub[h].seconds for sub in subs),
                members=[asdict(sub[h]) for sub in subs],
                why_unreached="a member was unreached" if result is None else "",
            )
        return out
    if UNSET in json.dumps(spec.get("args") or {}):
        return every(
            **base,
            result=None,
            zone=None,
            seconds=0.0,
            why_unreached="its args use a --var that was not given (filled with "
            f"{UNSET!r}); pass the real value to judge with this validator",
        )
    entry = (spec.get("body") or {}).get("entry")
    if not entry:
        return every(
            **base,
            result=None,
            zone=None,
            seconds=0.0,
            why_unreached=(
                "agent-bodied validator (no entry.sh): agent_sys production registers no "
                "executor for it, so the runtime reports it unreached too. Judge it by "
                "reading its readme, and label that judgement as Claude's, not agent_sys's."
            ),
        )

    zone = _fresh(work / name)
    (zone / "tmp").mkdir()
    (zone / "home").mkdir()
    materials = {}
    for n, (hid, content) in enumerate(targets.items()):
        staged = zone / "materials" / str(n)
        shutil.copytree(content, staged, symlinks=True)
        materials[hid] = f"materials/{n}"
    (zone / "inputs.json").write_text(json.dumps(sorted(hids)))
    (zone / "args.json").write_text(json.dumps(spec.get("args") or {}, indent=2))
    (zone / "materials.json").write_text(json.dumps(materials))
    agent = spec.get("agent")
    row = dict((pkg.agents.get(str(agent), {}).get("env") or {}) if agent else (source_env or {}))
    env = {
        **harness_env(),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "AGENT_SYS_DEMO_PACKAGE": str(pkg.root),
        "AGENT_SYS_DEMO_PYTHON": sys.executable,
        "AGENT_SYS_TASK_PACKAGE": str(pkg.root),
        **({"AGENT_SYS_DEMO_STORE": str(store)} if store is not None else {}),
        **{str(k): str(v) for k, v in row.items()},
        "HOME": str(zone / "home"),
        "TMPDIR": str(zone / "tmp"),
        "PWD": str(zone),
    }
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            ["/bin/sh", str(pkg.root / entry)],
            cwd=str(zone),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        tail = (proc.stdout + proc.stderr)[-4000:]
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        tail = f"timed out after {timeout:.0f}s: {exc}"
        rc = None
    seconds = round(time.monotonic() - t0, 2)
    (zone / "validator_output.txt").write_text(tail)
    report_path = zone / "validator_report.txt"
    report = report_path.read_text(errors="replace")[-8000:] if report_path.is_file() else ""
    out = {}
    for hid in hids:
        result, why = _verdict(zone / "verdict.json", hid, rc)
        out[hid] = Outcome(
            **base,
            result=result,
            zone=str(zone),
            seconds=seconds,
            report=report,
            output_tail=tail,
            why_unreached=why,
        )
    return out


def _verdict(path: Path, hid: str, rc: int | None) -> tuple[bool | None, str]:
    """`validator/phase.py` `read_verdict_file`: the file decides, not the exit code."""
    if not path.is_file():
        return None, f"exited {rc} and wrote no verdict.json; nothing was decided"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"verdict.json is not JSON: {exc}"
    if not isinstance(data, dict) or hid not in data:
        return None, f"verdict.json does not decide {hid!r}: {data!r}"[:500]
    return bool(data[hid]), ""


def _fresh(prefix: Path) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        p = prefix.with_name(f"{prefix.name}.{n}")
        try:
            p.mkdir()
            return p
        except FileExistsError:
            n += 1


def check_targets(
    pkg: Package,
    targets: dict[str, tuple[str, Path]],
    work: Path,
    *,
    closure: str | None = None,
    source_env: dict[str, str] | None = None,
    store: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
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
            timeout=timeout,
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
    hid: str = "skillkit-handoff",
    source_env: dict[str, str] | None = None,
    store: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    precheck_only: bool = False,
) -> KindCheck:
    return check_targets(
        pkg,
        {kind: (hid, content_dir)},
        work,
        closure=closure,
        source_env=source_env,
        store=store,
        timeout=timeout,
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
