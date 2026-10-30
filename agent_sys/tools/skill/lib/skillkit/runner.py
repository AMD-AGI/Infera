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
the gate's +x test, then every validator the leaf binds, over all its outputs
as one output phase; only a pass of the whole phase writes `manifest.yaml`
(for every output, or for none), and only a published version is visible to
the leaves that consume it — the scheduler's readiness rule.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from agent.gate import EXECUTABLE_ITEMS
from agent.runner import outputs_brief
from env_mgr.fs.domain import DomainKind, subdir_for
from env_mgr.fs.layout import LOGS, stage, stage_package
from env_mgr.fs.zone import Zone
from env_mgr.grants import input_env, output_env
from env_mgr.paths import zone_env
from handoff import content as content_mod
from skillkit import AGENT_SYS, hstore
from skillkit import validate as vmod
from skillkit.pkg import Leaf, Package, PackageError, load, require_show
from task_graph.ids import HandoffId

STATE = "state.json"


class NotReady(Exception):
    """A leaf cannot start: an input is unpublished or failed input validation.
    A refusal (exit 1), not a usage error."""


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
    require_show(pkg.root, state["variables"])
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


def seed(run: Path, kind: str, src: Path) -> vmod.KindCheck:
    """Publish an existing content dir as `kind`, validated like a producer's.

    This is what a publisher task does in agent_sys (`cp -a` into the output,
    then the kind's validators): an input is trusted because it passed, not
    because somebody pointed at it.
    """
    _state, pkg = open_run(run)
    pkg.kind(kind)
    work = _fresh_dir(run.resolve() / "validation" / f"seed.{kind}")
    slot, check = admit(
        pkg, run.resolve() / "handoffs", kind, src.resolve(), work, label=f"seed:{kind}"
    )
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
    state, pkg = open_run(run)
    leaf = leaf_of(pkg, name)
    store = run / "handoffs"
    # `load` reads the documents; agent_sys's closure and catalogue checks
    # (`show`) are what refuse e.g. a validator agent that does not resolve.
    require_show(pkg.root, state["variables"])

    # The scheduler dispatches a leaf only after the leaves it waits for
    # succeeded: one that has started and not succeeded (running, re-prepared,
    # failed) holds its dependents, whatever an executor says about it.
    by_name = {x.name: x for x in pkg.leaves()}
    held = [
        f"{d}: {s}"
        for d in leaf.deps
        for s in [leaf_status(state, by_name[d])]
        if s not in DONE and s != "pending"
    ]
    if held:
        raise NotReady(
            f"leaf {name!r} is not ready: upstream has not succeeded:\n  " + "\n  ".join(held)
        )

    inputs = {kind: hstore.latest_published(store, kind) for kind in leaf.inputs}
    missing = []
    for kind in (k for k, slot in inputs.items() if slot is None):
        producer = pkg.producer_of(kind)
        how = (
            f"run leaf {producer!r} first"
            if producer and producer != name
            else f"seed it: astk seed --run {run} --kind {kind} --content <dir>"
        )
        missing.append(f"{kind}: no published version — {how}")
    if missing:
        raise NotReady(f"leaf {name!r} is not ready:\n  " + "\n  ".join(missing))
    attempts = int(state["leaves"].get(name, {}).get("attempts", 0))
    _input_phase(run, pkg, leaf, inputs, label=f"{name}#{attempts + 1}")

    with locked(run) as st:
        rec = st["leaves"].setdefault(name, {"status": "pending", "attempts": 0})
        attempt = int(rec.get("attempts", 0))
        rec["attempts"] = attempt + 1
        rec["status"] = "preparing"
        # The previous attempt's slots are not this attempt's: `finish` must
        # never judge (or publish) them under the new attempt's label.
        for key in ("outputs", "inputs", "zone", "program_exit", "program_log"):
            rec.pop(key, None)
    try:
        info = _stage(run, pkg, leaf, store, inputs, attempt)
    except BaseException as exc:
        with locked(run) as st:
            st["leaves"][name].update(status="prepare_failed", prepare_error=str(exc)[-2000:])
        raise
    return info


def _stage(
    run: Path, pkg: Package, leaf: Leaf, store: Path, inputs: dict[str, hstore.Slot], attempt: int
) -> dict:
    """`prepare` after the attempt is counted: zone, staging, slots, env, brief."""
    name = leaf.name
    zone = _fresh_dir(run / "zones" / f"task.{name}.{attempt}")
    ezone = Zone(task_id=name, attempt=attempt, root=str(zone))
    for d in [subdir_for(k) for k in DomainKind] + [LOGS, "tmp"]:
        (zone / d).mkdir(parents=True, exist_ok=True)
    # env_mgr's own staging: the package copied (outward links materialised),
    # each input's content/ copied under handoffs/ in the store's naming.
    package_copy = Path(stage_package(str(pkg.root), ezone) or "")
    kind_of = {HandoffId(s.hid): kind for kind, s in inputs.items()}
    staged_by_hid = stage(
        list(kind_of),
        {HandoffId(s.hid): s.version for s in inputs.values()},
        str(zone / subdir_for(DomainKind.HANDOFF_STORAGE)),
        str(store),
    )
    staged = {kind_of[h]: Path(p) for h, p in staged_by_hid.items()}
    outputs = {kind: hstore.allocate(store, kind, pkg.kinds) for kind in leaf.outputs}

    # What a body sees, from env_mgr's own functions in `prepare`'s order —
    # `paths.zone_env`, `grants.output_env`, `grants.input_env` — then the CLI's
    # `AGENT_SYS_DEMO_*` facts, the zone's tmp, and the agent's `env:` last.
    out_hids = {HandoffId(s.hid): kind for kind, s in outputs.items()}
    task = SimpleNamespace(outputs=tuple(out_hids), kinds={**out_hids, **kind_of})
    execution = SimpleNamespace(
        output_versions={HandoffId(s.hid): s.version for s in outputs.values()}
    )
    env: dict[str, str] = {
        **zone_env(ezone, staged_package=str(package_copy)),
        **output_env(task, execution, str(store)),
        **input_env(task, staged_by_hid),
        "AGENT_SYS_DEMO_PACKAGE": str(pkg.root),
        "AGENT_SYS_DEMO_STORE": str(store),
        "AGENT_SYS_DEMO_PYTHON": sys.executable,
        "TMPDIR": str(zone / "tmp"),
    }
    agent = pkg.agents.get(leaf.agent, {})
    if agent.get("assets"):
        env["AGENT_SYS_AGENT_ASSETS"] = str(package_copy / str(agent["assets"]))
    env.update({str(k): str(v) for k, v in (agent.get("env") or {}).items()})
    (zone / "env.json").write_text(json.dumps(env, indent=2))
    (zone / "env.sh").write_text(
        f"# Source this before every command: `. {zone / 'env.sh'}`\n"
        + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
    )
    (zone / "brief.md").write_text(_brief(pkg, leaf, zone, staged, outputs, run))

    with locked(run) as st:
        rec = st["leaves"][name]
        rec.pop("prepare_error", None)
        rec.update(
            status="prepared",
            zone=str(zone),
            prepared_at=_now(),
            # An overlapping `prepare` may have counted a later attempt by now.
            prepared_attempt=attempt + 1,
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


def _input_phase(
    run: Path, pkg: Package, leaf: Leaf, inputs: dict[str, hstore.Slot], *, label: str
) -> None:
    """The consumer's input validation. Kind validators already judged each
    input when it was published, and agent_sys's default strict level reuses
    those verdicts; what is left is the leaf's own phase validators — run as
    the input phase (its environment row) and recorded on each input's
    `validation.yaml`, as `PhaseRunner._invoke` does, pass or fail."""
    work = None
    refused = []
    for v in pkg.closures[leaf.name].get("validators") or []:
        wants = pkg.validators.get(v, {}).get("inputs") or ()
        mine = {k: s for k, s in inputs.items() if k in wants}
        if not mine:
            continue
        work = work or _fresh_dir(run / "validation" / f"{leaf.name}.input")
        res = vmod.run_one(
            pkg,
            v,
            {k: (s.hid, s.content) for k, s in mine.items()},
            work,
            store=run / "handoffs",
            phase="input_validation",
        )
        for slot in mine.values():
            hstore.record_verdicts(slot, vmod.verdicts([res[slot.hid]], label))
        refused += [
            f"{v} on {hid}: {o.verdict} {o.why_unreached or o.report[-300:]}"
            for hid, o in res.items()
            if o.result is not True
        ]
    if refused:
        raise NotReady(f"leaf {leaf.name!r}: input validation refused:\n  " + "\n  ".join(refused))


def _brief(
    pkg: Package,
    leaf: Leaf,
    zone: Path,
    staged: dict[str, Path],
    outputs: dict[str, hstore.Slot],
    run: Path,
) -> str:
    """The leaf's prompt: agent_sys's readme + outputs brief, and how to work here."""
    program = leaf.agent_kind == "program"
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
        "`AGENT_SYS_TASK_PACKAGE` (a private copy of the package), `AGENT_SYS_MY_*`, and the agent's `env:` block.",
        "",
        "## Inputs (staged copies of each input's content: `README.md` + `items/` directly under the path; read, do not edit)",
        "",
        *([f"- **{k}**: `{p}`" for k, p in staged.items()] or ["- (none)"]),
        "",
        # The runtime's own wording for the paths, then, per output, what the
        # seal and the validators will hold it to.
        outputs_brief([(k, s.hid, s.content) for k, s in outputs.items()]),
        "",
    ]
    for kind in outputs:
        spec = pkg.kind(kind)
        ctype = content_mod.content_type(str(spec["content_type"]))
        sections = ", ".join(f"`## {s}`" for s in ctype.readme_sections)
        required = ctype.required_items | set(
            (spec.get("items_schema") or {}).get("required") or []
        )
        out += [
            f"- **{kind}**:",
            f"  - content_type `{ctype.name}`: `README.md` with document-root sections {sections}"
            " (exact names, real text under each)",
            f"  - items under `items/`: required {sorted(required)}{_one_of(ctype.name)}",
            f"  - {spec.get('description', '')}",
        ]
        for v in pkg.validators_for(leaf.name, kind):
            vs = pkg.validators.get(v, {})
            out.append(
                f"  - judged by `{v}`: {vs.get('brief', '')} ({vs.get('dimension')}/{vs.get('strength')})"
            )
    out += [
        "",
        "## Task instructions (the task's readme.md — agent_sys's system prompt for this leaf)",
        "",
    ]
    if program:
        out += [
            "This is a PROGRAM leaf: do not do its work by hand. Run",
            f"`{cmd('run-program', run, leaf.name)}`",
            "and, if it fails, diagnose from `logs/program.log`. The readme below is context.",
            "",
        ]
    out += [
        (pkg.root / leaf.readme).read_text() if leaf.readme else "(no readme)",
        "",
        "## Done means",
        "",
        f"`{cmd('finish', run, leaf.name)}` prints PASSED for every output.",
        "If it refuses, diagnose from `logs/program.log` and report; a program leaf's output is the "
        "program's — do not hand-edit it."
        if program
        else "If it refuses, read the reasons it prints, fix the handoff (never the validators), "
        "and run it again.",
    ]
    return "\n".join(out) + "\n"


def _one_of(content_type: str) -> str:
    """`handoff.content`'s one-of item groups, as the brief states them."""
    out = ""
    for group in content_mod._ALTERNATIVES.get(content_type, ()):
        names = "/".join(f"`{n}`" for n in sorted(group))
        out += f"; one of {names}" + (" (chmod +x)" if group & EXECUTABLE_ITEMS else "")
    return out


# -- execute and finish ----------------------------------------------------- #


class _Terminated(BaseException):
    """A termination signal delivered to astk while it waits on a program."""


@contextmanager
def _signals_raise(*signums: int) -> Iterator[None]:
    """Raise `_Terminated` on `signums` (main thread only; elsewhere a no-op)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum, _frame):
        raise _Terminated(signum)

    saved = {s: signal.signal(s, _raise) for s in signums}
    try:
        yield
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


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
        # Its own session, so a timeout stops everything entry.sh started — a
        # background child left running would keep writing into an output slot
        # that `finish` then seals.
        proc = subprocess.Popen(
            ["/bin/sh", str(zone / "package" / leaf.entry)],
            cwd=str(zone),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # The group is killed however the wait ends — timeout, Ctrl-C, or a
        # SIGTERM/SIGHUP to astk from a supervisor (turned into an exception
        # here so the cleanup runs) — not only on --timeout.
        try:
            with _signals_raise(signal.SIGTERM, signal.SIGHUP):
                rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = 124
        except _Terminated as e:
            # Die of the same signal once the group is gone (handler restored).
            _kill_group(proc)
            signal.signal(e.args[0], signal.SIG_DFL)
            os.kill(os.getpid(), e.args[0])
            raise
        finally:
            if proc.returncode is None:
                _kill_group(proc)
    with locked(run) as st:
        st["leaves"][name].update(program_exit=rc, program_log=str(log))
    return rc


def finish(run: Path, name: str) -> list[vmod.KindCheck]:
    """The gate: the leaf's output phase, all or nothing.

    Every validator the leaf binds runs on every output of the current attempt,
    every time, so a validator spanning two kinds (e2e-flow's
    `check_measurement_order`) always sees both. The verdicts fold over the
    whole phase (`validator/report.py`), and the outputs are sealed only if it
    passed — then all of them; otherwise none, so no consumer can stage an
    output of an attempt that failed (the runtime seals the attempt's model
    versions with the phase-wide verdict)."""
    run = run.resolve()
    state, pkg = open_run(run)
    leaf = leaf_of(pkg, name)
    rec = state["leaves"][name]
    # Judged: the current attempt, prepared (and, for a program leaf, run), or
    # refused by an earlier `finish` of this same attempt.
    if "outputs" not in rec or rec.get("status") not in ("prepared", "output_invalid"):
        raise PackageError(
            f"leaf {name!r} is {rec.get('status', 'pending')}, not prepared; "
            "run `astk prepare` (a new attempt) first"
        )
    if rec.get("prepared_attempt", rec["attempts"]) != rec["attempts"]:
        raise PackageError(f"leaf {name!r}: its output record is not the current attempt's")
    label = f"{name}#{rec['attempts']}"
    slots = {
        kind: hstore.Slot(
            kind=kind, hid=o["hid"], version=o["version"], root=Path(o["content"]).parents[1]
        )
        for kind, o in rec["outputs"].items()
    }
    before = {k: hstore.digest(s.content) for k, s in slots.items() if s.content.is_dir()}
    env_file = Path(rec["zone"]) / "env.json"
    source_env = json.loads(env_file.read_text()) if env_file.is_file() else None
    work = _fresh_dir(run / "validation" / f"{name}.{rec['attempts'] - 1}")
    checks = vmod.check_targets(
        pkg,
        {k: (s.hid, s.content) for k, s in slots.items()},
        work,
        source_env=source_env,
        store=run / "handoffs",
        bound=pkg.bound_validators(leaf.name),
    )
    changed = [
        k
        for k, s in slots.items()
        if not s.published and s.content.is_dir() and hstore.digest(s.content) != before.get(k)
    ]
    for kind in changed:
        checks[kind].precheck.append("content changed while it was being validated; not sealed")
    if not vmod.phase_passed(checks):
        failed = sorted(k for k, c in checks.items() if not c.passed)
        for kind, check in checks.items():
            if check.passed:
                check.held.append(f"the leaf's output phase did not pass ({', '.join(failed)})")
    else:
        for kind, check in checks.items():
            if not slots[kind].published:
                _publish(slots[kind], check, pkg, label=label)
    done = vmod.phase_passed(checks) and all(s.published for s in slots.values())
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


def admit(
    pkg: Package,
    store: Path,
    kind: str,
    src: Path,
    work: Path,
    *,
    label: str,
    closure: str | None = None,
) -> tuple[hstore.Slot, vmod.KindCheck]:
    """Copy `src` into a fresh version of `kind` in `store`, validate it there (with
    `closure`'s phase validators too, if given) and publish it on a full pass."""
    slot = hstore.import_content(store, kind, pkg.kinds, src)
    check = vmod.check_kind(
        pkg, kind, slot.content, work, closure=closure, hid=slot.hid, store=store
    )
    _publish(slot, check, pkg, label=label)
    return slot, check


def _publish(slot: hstore.Slot, check: vmod.KindCheck, pkg: Package, *, label: str) -> None:
    """On a full pass: the store's seal, then the verdicts appended to the
    `validation.yaml` the seal created — the runtime's order. A refused seal
    (it re-runs the admission checks) is reported on the check."""
    if not check.passed:
        return
    reason = hstore.seal(slot, pkg.kinds, producer=label)
    if reason is not None:
        check.precheck.append(f"the store refused to seal it: {reason}")
        return
    hstore.record_verdicts(slot, vmod.verdicts(check.outcomes, label))


def leaf_status(state: dict, leaf: Leaf) -> str:
    """The leaf's recorded status — or `seeded` when it has not succeeded but
    every kind it produces was published by `seed` (the run consumed it as
    input). That includes a leaf that ran and failed: seeding over a failed
    producer is the documented skip-ahead, and a failed attempt published
    nothing, so the seed is what its consumers stage."""
    status = state["leaves"].get(leaf.name, {}).get("status", "pending")
    if (
        status != "succeeded"
        and leaf.outputs
        and all(
            k in state.get("seeded", {}) and state["seeded"][k].get("passed") for k in leaf.outputs
        )
    ):
        return "seeded"
    return status


DONE = ("succeeded", "seeded")


def closure_report(run: Path) -> dict:
    """The end-of-run view, and the non-leaves' own output phase.

    Every leaf succeeded (or was seeded); every non-leaf's declared outputs are
    published; and every non-leaf phase validator (`validators:` on a task with a
    subgraph) passes on the latest published version of the kinds it lists."""
    run = run.resolve()
    state, pkg = open_run(run)
    store = run / "handoffs"
    leaves = {leaf.name: leaf_status(state, leaf) for leaf in pkg.leaves()}
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
            # A validator the package no longer defines (edited after run-init)
            # has no inputs, so it reports UNREACHED instead of raising.
            wants = pkg.validators.get(v, {}).get("inputs") or ()
            latest = {k: hstore.latest_published(store, k) for k in wants}
            mine = {k: (s.hid, s.content) for k, s in latest.items() if s is not None}
            if not mine:
                verdicts[v] = "UNREACHED (no published input)"
                continue
            work = _fresh_dir(run / "validation" / f"{name}.closure")
            res = vmod.run_one(pkg, v, mine, work, store=store)
            bad = [o.why_unreached or o.report[-200:] for o in res.values() if o.result is not True]
            verdicts[v] = "FAIL: " + "; ".join(bad) if bad else "PASS"
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
