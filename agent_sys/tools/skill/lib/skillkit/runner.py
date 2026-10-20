"""Execute a task package's leaves from Claude Code, without the agent_sys runtime.

A run directory mirrors what `agent-sys run` leaves behind, at the scale a
person reads:

    <run>/state.json                              package, --var set, per-leaf status
    <run>/handoffs/handoff.<kind>.<hid>/v<N>/     the store (see `hstore`)
    <run>/zones/task.<leaf>.<attempt>/            one zone per attempt:
        env.sh   brief.md   handoffs/  workspace/  playground/  logs/  tmp/  package/
    <run>/validation/<leaf>.<attempt>/<validator>.<n>/   validator zones, kept

The life of a leaf is `prepare` -> (program: `run-program` | ai: Claude does
the brief) -> `finish`. `finish` is the quality gate: the store's seal test,
the gate's +x test, then every validator bound to each output; only a full
pass writes `manifest.yaml`, and only a published version is visible to the
leaves that consume it — the scheduler's readiness rule.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from skillkit import AGENT_SYS, hstore
from skillkit import validate as vmod
from skillkit.pkg import Leaf, Package, PackageError, env_name, load

STATE = "state.json"


class NotReady(Exception):
    """A leaf cannot start: an input is unpublished or failed input validation.
    A refusal (exit 1), not a usage error."""


#: A package bigger than this is linked, not copied, into each zone.
COPY_LIMIT = 200 << 20
ASTK = AGENT_SYS / "tools" / "skill" / "lib" / "astk.py"


def cmd(verb: str, run: Path, leaf: str) -> str:
    """An astk command line, quoted for any shell."""
    return " ".join(
        shlex.quote(x) for x in ("python3", str(ASTK), verb, "--run", str(run), "--leaf", leaf)
    )


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# -- state ------------------------------------------------------------------ #


def init(run: Path, package: Path, variables: dict[str, str]) -> dict:
    run = run.resolve()
    state_path = run / STATE
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if Path(state["package"]) != package.resolve():
            raise PackageError(f"{run} already belongs to package {state['package']}")
        if variables and variables != state["variables"]:
            state["variables"] = {**state["variables"], **variables}
            state.setdefault("notes", []).append(f"{_now()} --var set updated: {sorted(variables)}")
    else:
        state = {
            "package": str(package.resolve()),
            "variables": variables,
            "created": _now(),
            "executor": "Claude Code via agent_sys/tools/skill (not the agent_sys runtime)",
            "leaves": {},
            "seeded": {},
            "notes": [],
        }
    pkg = load(package, state["variables"])
    for d in ("handoffs", "zones", "validation"):
        (run / d).mkdir(parents=True, exist_ok=True)
    for leaf in pkg.leaves():
        state["leaves"].setdefault(leaf.name, {"status": "pending", "attempts": 0})
    save(run, state)
    return state


def save(run: Path, state: dict) -> None:
    tmp = run / f"{STATE}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(run / STATE)


@contextmanager
def locked(run: Path) -> Iterator[dict]:
    """Read-modify-write `state.json` under an exclusive lock.

    The generated Workflow runs independent leaves at once, each driving
    `astk` in its own process; without the lock, two `finish` calls would each
    write back a stale copy and lose the other's update."""
    run = run.resolve()
    with (run / "state.lock").open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            state = json.loads((run / STATE).read_text())
            yield state
            save(run, state)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def open_run(run: Path) -> tuple[dict, Package]:
    run = run.resolve()
    path = run / STATE
    if not path.is_file():
        raise PackageError(f"{run} is not a run directory (no {STATE}); run `astk run-init` first")
    state = json.loads(path.read_text())
    return state, load(state["package"], state["variables"])


def leaf_of(pkg: Package, name: str) -> Leaf:
    for leaf in pkg.leaves():
        if leaf.name == name:
            return leaf
    raise PackageError(f"no leaf {name!r} (leaves: {[x.name for x in pkg.leaves()]})")


# -- seeding an external input ---------------------------------------------- #


def seed(run: Path, kind: str, src: Path, *, timeout: float) -> vmod.KindCheck:
    """Publish an existing content dir as `kind`, validated like a producer's.

    This is what a publisher task does in agent_sys (`cp -a` into the output,
    then the kind's validators): an input is trusted because it passed, not
    because somebody pointed at it.
    """
    _state, pkg = open_run(run)
    pkg.kind(kind)
    slot = hstore.import_content(run.resolve() / "handoffs", kind, src.resolve())
    work = _fresh_dir(run.resolve() / "validation" / f"seed.{kind}")
    check = vmod.check_kind(
        pkg, kind, slot.content, work, hid=slot.hid, store=run / "handoffs", timeout=timeout
    )
    hstore.record_verdicts(slot, vmod.rows_for_store(check, task_id=f"seed:{kind}"))
    if check.passed:
        hstore.seal(slot, producer=f"seed:{src}")
    with locked(run) as st:
        st["seeded"][kind] = {
            "from": str(src.resolve()),
            "content": str(slot.content),
            "passed": check.passed,
        }
    return check


# -- prepare ---------------------------------------------------------------- #


def prepare(run: Path, name: str) -> dict:
    run = run.resolve()
    _state, pkg = open_run(run)
    leaf = leaf_of(pkg, name)
    store = run / "handoffs"

    missing = []
    inputs: dict[str, hstore.Slot] = {}
    for kind in leaf.inputs:
        slot = hstore.latest_published(store, kind)
        if slot is None:
            producer = pkg.producer_of(kind)
            how = (
                f"run leaf {producer!r} first"
                if producer and producer != name
                else f"seed it: astk seed --run {run} --kind {kind} --content <dir>"
            )
            missing.append(f"{kind}: no published version — {how}")
        else:
            inputs[kind] = slot
    if missing:
        raise NotReady(f"leaf {name!r} is not ready:\n  " + "\n  ".join(missing))
    _input_phase(run, pkg, leaf, inputs)

    with locked(run) as st:
        rec = st["leaves"].setdefault(name, {"status": "pending", "attempts": 0})
        attempt = int(rec.get("attempts", 0))
        rec["attempts"] = attempt + 1
        rec["status"] = "preparing"
    zone = _fresh_dir(run / "zones" / f"task.{name}.{attempt}")
    for d in ("handoffs", "workspace", "playground", "logs", "tmp"):
        (zone / d).mkdir(parents=True, exist_ok=True)
    package_copy = _stage_package(pkg.root, zone / "package")

    staged: dict[str, Path] = {}
    for kind, slot in inputs.items():
        dst = zone / "handoffs" / slot.root.name / f"v{slot.version}"
        shutil.copytree(slot.content, dst, symlinks=True)
        staged[kind] = dst
    outputs = {kind: hstore.allocate(store, kind) for kind in leaf.outputs}

    env = _environment(pkg, leaf, zone, package_copy, staged, outputs, store)
    (zone / "env.json").write_text(json.dumps(env, indent=2))
    (zone / "env.sh").write_text(
        "# Source this before every command: `. "
        + str(zone / "env.sh")
        + "`\n"
        + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
    )
    (zone / "brief.md").write_text(_brief(pkg, leaf, zone, staged, outputs, run))

    with locked(run) as st:
        rec = st["leaves"][name]
        rec.update(
            status="prepared",
            zone=str(zone),
            prepared_at=_now(),
            inputs={k: str(v) for k, v in staged.items()},
            outputs={
                k: {"hid": s.hid, "version": s.version, "content": str(s.content)}
                for k, s in outputs.items()
            },
        )
    return {
        "leaf": name,
        "agent_kind": leaf.agent_kind,
        "zone": str(zone),
        "env_sh": str(zone / "env.sh"),
        "brief": str(zone / "brief.md"),
        "entry": str(package_copy / leaf.entry) if leaf.entry else None,
        "outputs": rec["outputs"],
        "inputs": rec["inputs"],
    }


def _input_phase(run: Path, pkg: Package, leaf: Leaf, inputs: dict[str, hstore.Slot]) -> None:
    """The consumer's input validation. Kind validators already judged each
    input when it was published, and agent_sys's default strict level reuses
    those verdicts; what is left is the leaf's own phase validators."""
    phase = [
        v
        for v in pkg.closures[leaf.name].get("validators") or []
        if set(pkg.validators.get(v, {}).get("inputs") or []) & set(inputs)
    ]
    if not phase:
        return
    targets = {k: (s.hid, s.content) for k, s in inputs.items()}
    work = _fresh_dir(run / "validation" / f"{leaf.name}.input")
    refused = []
    for v in phase:
        wants = set(pkg.validators[v].get("inputs") or [])
        mine = {hid: content for k, (hid, content) in targets.items() if k in wants}
        for hid, o in vmod.run_one(pkg, v, mine, work, store=run / "handoffs").items():
            if o.result is not True:
                refused.append(f"{v} on {hid}: {o.verdict} {o.why_unreached or o.report[-300:]}")
    if refused:
        raise NotReady(f"leaf {leaf.name!r}: input validation refused:\n  " + "\n  ".join(refused))


def _stage_package(root: Path, dst: Path) -> Path:
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    if size > COPY_LIMIT:
        dst.symlink_to(root, target_is_directory=True)
    else:
        # Dereference: a package may link files from a sibling package
        # (02a/02b/03 link into e2e-flow), and a relative link would dangle here.
        shutil.copytree(root, dst, symlinks=False, ignore_dangling_symlinks=True)
    return dst


def _environment(
    pkg: Package,
    leaf: Leaf,
    zone: Path,
    package_copy: Path,
    staged: dict[str, Path],
    outputs: dict[str, hstore.Slot],
    store: Path,
) -> dict[str, str]:
    """`env_mgr.prepare`'s exports, in its order; the agent's `env` wins."""
    env: dict[str, str] = {
        "AGENT_SYS_MY_ZONE": str(zone),
        "AGENT_SYS_MY_HANDOFFS": str(zone / "handoffs"),
        "AGENT_SYS_MY_WORKSPACE": str(zone / "workspace"),
        "AGENT_SYS_MY_PLAYGROUND": str(zone / "playground"),
        "AGENT_SYS_MY_LOGS": str(zone / "logs"),
        "AGENT_SYS_TASK_PACKAGE": str(package_copy),
        "AGENT_SYS_DEMO_PACKAGE": str(pkg.root),
        "AGENT_SYS_DEMO_STORE": str(store),
        "AGENT_SYS_DEMO_PYTHON": sys.executable,
        "TMPDIR": str(zone / "tmp"),
    }
    for prefix, paths in (
        ("AGENT_SYS_OUTPUT_", {k: s.content for k, s in outputs.items()}),
        ("AGENT_SYS_INPUT_", staged),
    ):
        names: dict[str, list[Path]] = {}
        for kind, path in paths.items():
            names.setdefault(prefix + env_name(kind), []).append(path)
        # Two kinds on one name: the runtime exports neither (env_mgr/grants.py).
        env.update({n: str(p[0]) for n, p in names.items() if len(p) == 1})
    agent = pkg.agents.get(leaf.agent, {})
    if agent.get("assets"):
        env["AGENT_SYS_AGENT_ASSETS"] = str(package_copy / str(agent["assets"]))
    for k, v in (agent.get("env") or {}).items():
        env[str(k)] = str(v)
    return env


def _brief(
    pkg: Package,
    leaf: Leaf,
    zone: Path,
    staged: dict[str, Path],
    outputs: dict[str, hstore.Slot],
    run: Path,
) -> str:
    from handoff import content as content_mod  # noqa: PLC0415

    out = [
        f"# Leaf `{leaf.name}` — executed by Claude Code, not the agent_sys runtime",
        "",
        f"- package: `{pkg.root}`",
        f"- agent: `{leaf.agent}` (kind `{leaf.agent_kind}`)",
        f"- goal: {leaf.goal}",
        f"- zone (your working directory): `{zone}`",
        "",
        "## Environment",
        "",
        f'Every shell is fresh: start every command with `. {shlex.quote(str(zone / "env.sh"))} && cd "$AGENT_SYS_MY_ZONE"`.',
        "It exports what agent_sys would: `AGENT_SYS_INPUT_<KIND>`, `AGENT_SYS_OUTPUT_<KIND>`,",
        "`AGENT_SYS_TASK_PACKAGE`, `AGENT_SYS_MY_*`, and the agent's `env:` block.",
        (
            "`AGENT_SYS_TASK_PACKAGE` is a SYMLINK to the original package (too large to copy): "
            "treat it as read-only."
            if (zone / "package").is_symlink()
            else "`AGENT_SYS_TASK_PACKAGE` is a private copy of the package."
        ),
        "",
        "## Inputs (staged copies of each input's content: `README.md` + `items/` directly under the path; read, do not edit)",
        "",
    ]
    out += [f"- **{k}**: `{p}`" for k, p in staged.items()] or ["- (none)"]
    out += ["", "## What this task must deliver", ""]
    for kind, slot in outputs.items():
        spec = pkg.kind(kind)
        ctype = content_mod.content_type(str(spec["content_type"]))
        out.append(f"- **{kind}** (`{slot.hid}`): write it into `{slot.content}`")
        out.append(
            f"  - content_type `{ctype.name}`: `README.md` with document-root sections "
            + ", ".join(f"`## {s}`" for s in ctype.readme_sections)
            + " (exact names, real text under each)"
        )
        req = sorted(
            ctype.required_items | set((spec.get("items_schema") or {}).get("required") or [])
        )
        out.append(
            f"  - items under `items/`: required {req}"
            + (
                "; one of `script`/`command` (chmod +x)"
                if ctype.name == "reproducible"
                else "; one of `text.json`/`text.yaml`/`text.xml`"
                if ctype.name == "structured_text"
                else ""
            )
        )
        out.append(f"  - {spec.get('description', '')}")
        for v in pkg.validators_for(leaf.name, kind):
            vs = pkg.validators.get(v, {})
            out.append(
                f"  - judged by `{v}`: {vs.get('brief', '')} ({vs.get('dimension')}/{vs.get('strength')})"
            )
    readme = (pkg.root / leaf.readme).read_text() if leaf.readme else "(no readme)"
    out += [
        "",
        "## Task instructions (the task's readme.md — agent_sys's system prompt for this leaf)",
        "",
    ]
    if leaf.agent_kind == "program":
        out += [
            "This is a PROGRAM leaf: do not do its work by hand. Run",
            f"`{cmd('run-program', run, leaf.name)}`",
            "and, if it fails, diagnose from `logs/program.log`. The readme below is context.",
            "",
        ]
    out += [
        readme,
        "",
        "## Done means",
        "",
        f"`{cmd('finish', run, leaf.name)}` prints PASSED for every output.",
        (
            "If it refuses, diagnose from `logs/program.log` and report; a program leaf's output is the "
            "program's — do not hand-edit it."
            if leaf.agent_kind == "program"
            else "If it refuses, read the reasons it prints, fix the handoff (never the validators), "
            "and run it again."
        ),
    ]
    return "\n".join(out) + "\n"


# -- execute and finish ----------------------------------------------------- #


def run_program(run: Path, name: str, *, timeout: float | None = None) -> int:
    run = run.resolve()
    state, pkg = open_run(run)
    leaf = leaf_of(pkg, name)
    rec = state["leaves"][name]
    if rec.get("status") != "prepared":
        raise PackageError(f"leaf {name!r} is {rec.get('status')}; run `astk prepare` first")
    if not leaf.entry:
        raise PackageError(f"leaf {name!r} has no entry.sh (it is an {leaf.agent_kind} leaf)")
    zone = Path(rec["zone"])
    env = {**os.environ, **json.loads((zone / "env.json").read_text())}
    log = zone / "logs" / "program.log"
    with log.open("w") as fh:
        try:
            rc = subprocess.run(
                ["/bin/sh", str(zone / "package" / leaf.entry)],
                cwd=str(zone),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    with locked(run) as st:
        st["leaves"][name].update(program_exit=rc, program_log=str(log))
    return rc


def finish(run: Path, name: str, *, timeout: float) -> list[vmod.KindCheck]:
    """The gate. All unpublished outputs are judged together, so a validator
    spanning two of them (e2e-flow's `check_measurement_order`) sees both."""
    run = run.resolve()
    state, pkg = open_run(run)
    leaf = leaf_of(pkg, name)
    rec = state["leaves"][name]
    if "outputs" not in rec:
        raise PackageError(f"leaf {name!r} was never prepared")
    label = f"{name}#{rec['attempts']}"
    slots = {
        kind: hstore.Slot(
            kind=kind, hid=o["hid"], version=o["version"], root=Path(o["content"]).parents[1]
        )
        for kind, o in rec["outputs"].items()
    }
    pending = {k: s for k, s in slots.items() if not s.published}
    before = {k: hstore.digest(s.content) for k, s in pending.items() if s.content.is_dir()}
    zone_env = Path(rec["zone"]) / "env.json"
    source_env = json.loads(zone_env.read_text()) if zone_env.is_file() else None
    work = _fresh_dir(run / "validation" / f"{name}.{rec['attempts'] - 1}")
    checks = vmod.check_targets(
        pkg,
        {k: (s.hid, s.content) for k, s in pending.items()},
        work,
        closure=leaf.name,
        source_env=source_env,
        store=run / "handoffs",
        timeout=timeout,
    )
    for kind, check in checks.items():
        slot = pending[kind]
        hstore.record_verdicts(slot, vmod.rows_for_store(check, task_id=label))
        if check.passed:
            if hstore.digest(slot.content) != before.get(kind):
                check.precheck.append("content changed while it was being validated; not sealed")
                continue
            hstore.seal(slot, producer=label)
    done = all(s.published for s in slots.values())
    with locked(run) as st:
        r = st["leaves"][name]
        r["status"] = "succeeded" if done else "output_invalid"
        r.setdefault("finish_log", []).append(
            {
                "at": _now(),
                "passed": [k for k, c in checks.items() if c.passed],
                "failed": [k for k, c in checks.items() if not c.passed],
            }
        )
    return list(checks.values())


def leaf_status(state: dict, leaf: Leaf, store: Path) -> str:
    """The leaf's recorded status — or `seeded` when it never ran but every
    kind it produces was published by `seed` (the run consumed it as input)."""
    status = state["leaves"].get(leaf.name, {}).get("status", "pending")
    if (
        status in ("pending",)
        and leaf.outputs
        and all(
            k in state.get("seeded", {}) and state["seeded"][k].get("passed") for k in leaf.outputs
        )
    ):
        return "seeded"
    return status


DONE = ("succeeded", "seeded")


def closure_report(run: Path, *, timeout: float = vmod.DEFAULT_TIMEOUT) -> dict:
    """The end-of-run view, and the non-leaves' own output phase.

    Every leaf succeeded (or was seeded); every non-leaf's declared outputs are
    published; and every non-leaf phase validator (`validators:` on a task with a
    subgraph) passes on the latest published version of the kinds it lists."""
    run = run.resolve()
    state, pkg = open_run(run)
    store = run / "handoffs"
    leaves = {leaf.name: leaf_status(state, leaf, store) for leaf in pkg.leaves()}
    nonleaf = {}
    for name in pkg.closures:
        if pkg.is_leaf(name):
            continue
        outs = {}
        for kind in pkg.task(name).get("outputs") or []:
            slot = hstore.latest_published(store, kind)
            outs[kind] = str(slot.content) if slot else None
        verdicts = {}
        for v in pkg.closures[name].get("validators") or []:
            mine = {}
            for kind in pkg.validators.get(v, {}).get("inputs") or []:
                slot = hstore.latest_published(store, kind)
                if slot is not None:
                    mine[slot.hid] = slot.content
            if not mine:
                verdicts[v] = "UNREACHED (no published input)"
                continue
            work = _fresh_dir(run / "validation" / f"{name}.closure")
            res = vmod.run_one(pkg, v, mine, work, store=store, timeout=timeout)
            verdicts[v] = (
                "PASS"
                if all(o.result is True for o in res.values())
                else (
                    "FAIL: "
                    + "; ".join(
                        o.why_unreached or o.report[-200:]
                        for o in res.values()
                        if o.result is not True
                    )
                )
            )
        nonleaf[name] = {"outputs": outs, "phase_validators": verdicts}
    ok = (
        all(s in DONE for s in leaves.values())
        and all(p for n in nonleaf.values() for p in n["outputs"].values())
        and all(v == "PASS" for n in nonleaf.values() for v in n["phase_validators"].values())
    )
    return {
        "run": str(run),
        "package": state["package"],
        "complete": ok,
        "leaves": leaves,
        "non_leaves": nonleaf,
    }


def _fresh_dir(prefix: Path) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        p = prefix.with_name(f"{prefix.name}.{n}") if n else prefix
        try:
            p.mkdir()
            return p
        except FileExistsError:
            n += 1
