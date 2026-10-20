"""`astk` — command line for the agent_sys skills. Run `astk <command> -h`.

Exit codes: 0 ok / passed, 1 refused / not passed, 2 usage or load error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from handoff.errors import Malformed
from skillkit import audit as audit_mod
from skillkit import envinfo, history, hstore, runner, scaffold, wf
from skillkit import pkg as pkg_mod
from skillkit import validate as vmod


def _vars(raw: list[str] | None) -> dict[str, str]:
    out = {}
    for item in raw or []:
        k, sep, v = item.partition("=")
        if not sep or not k:
            print(f"astk: --var wants K=V, got {item!r}", file=sys.stderr)
            raise SystemExit(2)
        out[k] = v
    return out


def _add_unset(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--allow-unset-vars",
        action="store_true",
        help="fill missing required --vars with a sentinel (judging only; a validator whose args need one is UNREACHED)",
    )


def _add_var(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--var",
        action="append",
        metavar="K=V",
        help="package variable, repeatable (as agent-sys --var)",
    )


def _emit(obj: object, as_json: bool, text: str) -> None:
    print(json.dumps(obj, indent=2, default=str) if as_json else text)


# -- history ------------------------------------------------------------------ #


def cmd_history_list(a: argparse.Namespace) -> int:
    project = Path(a.project) if a.project else history.project_dir_for(a.cwd or os.getcwd())
    sessions = [
        s
        for s in history.list_sessions(project)
        if (not a.since or s.end[: len(a.since)] >= a.since)
        and (not a.until or s.start[: len(a.until)] <= a.until)
    ]
    if a.json:
        print(json.dumps([{**s.__dict__, "path": str(s.path)} for s in sessions], indent=2))
        return 0
    print(f"{project}  ({len(sessions)} sessions, by last activity, most recent last)")
    for s in sessions:
        print(
            f"- {s.session_id[:8]}  {s.start[:16]} .. {s.end[:16]}  {s.size / 2**20:6.1f} MB  "
            f"prompts={s.human_prompts:<3} compactions={s.compactions:<2} {s.title or ''}"
        )
        if s.first_prompt:
            print(f"    first: {s.first_prompt[:150]!r}")
    return 0


def cmd_history_digest(a: argparse.Namespace) -> int:
    project = Path(a.project) if a.project else history.project_dir_for(a.cwd or os.getcwd())
    opts = history.DigestOptions(
        since=a.since, until=a.until, grep=a.grep, max_text=a.max_text, tools=not a.no_tools
    )
    parts, scripts = [], []
    for path in history.resolve_sessions(project, a.session or ["latest"]):
        info = history.session_info(path)
        dg = history.digest(path, opts)
        parts.append(history.render(info, dg))
        scripts += [(info.session_id[:8], i, w) for i, w in enumerate(dg.workflows)]
        sub = path.with_suffix("") / "subagents"
        if sub.is_dir():
            parts[-1] += (
                f"\n(Subagent and workflow transcripts of this session: {sub} —"
                " their results are not in this digest.)\n"
            )
    text = "\n\n".join(parts)
    if a.out and scripts:
        for sid, i, body in scripts:
            side = Path(a.out).with_name(f"{Path(a.out).stem}.{sid}.workflow-{i}.js")
            side.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(history.redact(body))
        print(f"{len(scripts)} Workflow script(s) written beside {a.out} as *.workflow-N.js")
    if not a.out:
        print(text)
        return 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.split_chars and len(text) > a.split_chars:
        chunks = history.split(text, a.split_chars)
        for i, c in enumerate(chunks):
            out.with_name(f"{out.stem}.part{i:02d}{out.suffix}").write_text(c)
        print(
            f"{len(text)} chars in {len(chunks)} parts: {out.stem}.partNN{out.suffix} in {out.parent}"
        )
    else:
        out.write_text(text)
        print(f"{len(text)} chars -> {out}")
    return 0


# -- packages ----------------------------------------------------------------- #


def cmd_show(a: argparse.Namespace) -> int:
    rc, out = pkg_mod.show(Path(a.package).resolve(), _vars(a.var))
    print(out)
    return 0 if rc == 0 else 1


def cmd_inspect(a: argparse.Namespace) -> int:
    p = pkg_mod.load(a.package, _vars(a.var), allow_unset=a.allow_unset_vars)
    leaves = p.leaves()
    data = {
        "package": str(p.root),
        "leaves": [leaf.__dict__ for leaf in leaves],
        "non_leaves": [n for n in p.closures if not p.is_leaf(n)],
        "kinds": {
            k: {
                "content_type": v.get("content_type"),
                "validators": p.validators_for(p.producer_of(k), k),
                "producer": p.producer_of(k),
                "items_schema": v.get("items_schema"),
            }
            for k, v in p.kinds.items()
        },
        "warnings": p.warnings,
    }
    lines = [f"package {p.root}", "", "leaves (execution order):"]
    for leaf in leaves:
        lines.append(f"  {leaf.name:<28} {leaf.agent_kind:<8} in={leaf.inputs} out={leaf.outputs}")
        lines.append(f"  {'':<28} waits for {leaf.deps or '—'}; under {' > '.join(leaf.ancestors)}")
        if leaf.external_inputs:
            lines.append(f"  {'':<28} EXTERNAL inputs (seed them): {leaf.external_inputs}")
    lines += ["", "kinds (validators: cost/strength/logic_source):"]
    for k, v in data["kinds"].items():
        vs = ", ".join(
            f"{n} ({(p.validators.get(n, {}).get('tags') or {}).get('cost')}/"
            f"{p.validators.get(n, {}).get('strength')}/"
            f"{(p.validators.get(n, {}).get('tags') or {}).get('logic_source')})"
            for n in v["validators"]
        )
        lines.append(f"  {k:<36} {v['content_type']:<16} by {v['producer'] or 'EXTERNAL'}")
        lines.append(f"  {'':<36} {vs}")
    lines += [f"warning: {w}" for w in p.warnings]
    _emit(data, a.json, "\n".join(lines))
    return 0


def cmd_scaffold(a: argparse.Namespace) -> int:
    plan = scaffold.load_plan(Path(a.plan))
    dest = Path(a.dest).resolve()
    written = scaffold.scaffold(plan, dest, force=a.force)
    print(f"wrote {len(written)} files under {dest}")
    rc, out = pkg_mod.show(dest, _vars(a.var))
    print(out)
    print("agent_sys `show`: " + ("OK" if rc == 0 else f"FAILED (exit {rc})"))
    if rc != 0:
        print(f"The package was written to {dest} anyway; fix the plan and re-run with --force.")
    return 0 if rc == 0 else 1


# -- handoffs ----------------------------------------------------------------- #


def cmd_template(a: argparse.Namespace) -> int:
    p = pkg_mod.load(a.package, _vars(a.var), allow_unset=a.allow_unset_vars)
    spec = p.kind(a.kind)
    made = hstore.template(Path(a.out), str(spec["content_type"]), spec.get("items_schema"))
    print(f"{spec['content_type']} skeleton for {a.kind} at {a.out}:")
    for m in made:
        print(f"  + {m}")
    print(
        "Every README body is the placeholder the seal rejects — replace each one with real text."
    )
    producer = p.producer_of(a.kind)
    leaf = next((x for x in p.leaves() if x.name == producer), None)
    if leaf and leaf.readme:
        hint = _section(
            (p.root / leaf.readme).read_text(errors="replace"), ("write", "deliver", "output")
        )
        if hint:
            print(f"\nWhat the producing task `{producer}` says to write (from its readme):\n")
            print(hint)
    return 0


def _section(markdown: str, words: tuple[str, ...]) -> str:
    """The first heading section whose title contains one of `words`."""
    lines, out, depth = markdown.splitlines(), [], 0
    for ln in lines:
        stripped = ln.lstrip("#")
        level = len(ln) - len(stripped)
        if level and ln.startswith("#") and stripped.startswith(" "):
            if out and level <= depth:
                break
            if not out and any(w in stripped.lower() for w in words):
                depth = level
                out.append(ln)
                continue
        if out:
            out.append(ln)
    return "\n".join(out).strip()


def _scratch() -> Path:
    """Validator zones go to a private dir under the temp dir by default, never
    beside the content (which may be inside a repo). The dir must be ours:
    owned by this uid, mode 0700, not a symlink — a shared /tmp is not trusted."""
    base = Path(tempfile.gettempdir()) / f"astk-{os.getuid()}"
    base.mkdir(mode=0o700, exist_ok=True)
    st = base.lstat()
    if base.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise pkg_mod.PackageError(f"{base} is not a private dir of this user; pass --work <dir>")
    return base / "validation"


def _work(a: argparse.Namespace, label: str) -> Path:
    base = Path(a.work) if getattr(a, "work", None) else _scratch()
    base.mkdir(parents=True, exist_ok=True)
    return runner._fresh_dir(base / label)


def cmd_check(a: argparse.Namespace) -> int:
    p = pkg_mod.load(a.package, _vars(a.var), allow_unset=a.allow_unset_vars)
    content = Path(a.content).resolve()
    work = _work(a, f"check.{a.kind}")
    check = vmod.check_kind(
        p,
        a.kind,
        content,
        work,
        closure=a.closure or p.producer_of(a.kind),
        timeout=a.timeout,
        precheck_only=a.precheck_only,
    )
    _emit(check.to_dict(), a.json, vmod.render(check) + f"\n(validator zones under {work})")
    return 0 if (check.passed or (a.precheck_only and not check.precheck)) else 1


def cmd_seal(a: argparse.Namespace) -> int:
    """Copy a content dir into an agent_sys-shaped store, validate, publish on pass."""
    p = pkg_mod.load(a.package, _vars(a.var), allow_unset=a.allow_unset_vars)
    store = Path(a.store).resolve()
    content = Path(a.content).resolve()
    same = hstore.find_digest(store, a.kind, hstore.digest(content))
    if same is not None:
        print(f"already published, identical content: {same.vdir} (nothing written)")
        return 0
    slot = hstore.import_content(store, a.kind, content)
    work = _work(a, f"seal.{a.kind}")
    check = vmod.check_kind(
        p,
        a.kind,
        slot.content,
        work,
        closure=p.producer_of(a.kind),
        hid=slot.hid,
        store=store,
        timeout=a.timeout,
    )
    hstore.record_verdicts(slot, vmod.rows_for_store(check, task_id=a.task_id))
    print(vmod.render(check))
    if check.passed:
        m = hstore.seal(slot, producer=a.task_id)
        print(f"published {slot.vdir}  sha256={m['digest']['sha256']}")
        return 0
    print(f"NOT published: {slot.vdir} keeps content + validation.yaml, no manifest.yaml")
    return 1


def cmd_audit(a: argparse.Namespace) -> int:
    r = audit_mod.audit(Path(a.dir))
    _emit(r.__dict__ | {"blocking": r.blocking, "ask_user": r.ask_user}, a.json, r.render())
    return 1 if r.blocking else 0


def cmd_env(a: argparse.Namespace) -> int:
    info = envinfo.capture(a.repo, verbose=a.verbose)
    print(json.dumps(info, indent=2) if a.json else envinfo.render_md(info))
    return 0


def cmd_digest(a: argparse.Namespace) -> int:
    print(hstore.digest(Path(a.content)))
    return 0


# -- runs ---------------------------------------------------------------------- #


def cmd_run_init(a: argparse.Namespace) -> int:
    state = runner.init(Path(a.run), Path(a.package), _vars(a.var))
    print(f"run {Path(a.run).resolve()} for {state['package']}")
    return cmd_status(argparse.Namespace(run=a.run, json=False))


def cmd_seed(a: argparse.Namespace) -> int:
    check = runner.seed(Path(a.run), a.kind, Path(a.content), timeout=a.timeout)
    print(vmod.render(check))
    return 0 if check.passed else 1


def cmd_prepare(a: argparse.Namespace) -> int:
    info = runner.prepare(Path(a.run), a.leaf)
    print(json.dumps(info, indent=2))
    return 0


def cmd_run_program(a: argparse.Namespace) -> int:
    rc = runner.run_program(Path(a.run), a.leaf, timeout=a.timeout)
    state, _ = runner.open_run(Path(a.run))
    log = state["leaves"][a.leaf]["program_log"]
    tail = Path(log).read_text(errors="replace").splitlines()[-30:]
    print(f"exit {rc}; log {log}\n" + "\n".join(f"  | {ln}" for ln in tail))
    return 0 if rc == 0 else 1


def cmd_finish(a: argparse.Namespace) -> int:
    checks = runner.finish(Path(a.run), a.leaf, timeout=a.timeout)
    for c in checks:
        print(vmod.render(c))
    state, _ = runner.open_run(Path(a.run))
    status = state["leaves"][a.leaf]["status"]
    print(f"leaf {a.leaf}: {status}")
    return 0 if status == "succeeded" else 1


def cmd_status(a: argparse.Namespace) -> int:
    state, p = runner.open_run(Path(a.run))
    if a.json:
        print(json.dumps(state, indent=2))
        return 0
    print(f"package {state['package']}  vars {state['variables']}")
    for leaf in p.leaves():
        rec = state["leaves"].get(leaf.name, {})
        status = runner.leaf_status(state, leaf, Path(a.run).resolve() / "handoffs")
        print(
            f"  {leaf.name:<28} {leaf.agent_kind:<8} {status:<15} "
            f"prepares={rec.get('attempts', 0)} finishes={len(rec.get('finish_log', []))}  "
            f"waits for {leaf.deps or '—'}"
        )
    for k, s in state.get("seeded", {}).items():
        print(f"  seeded {k}: passed={s['passed']} from {s['from']}")
    return 0


def cmd_closure(a: argparse.Namespace) -> int:
    rep = runner.closure_report(Path(a.run))
    print(json.dumps(rep, indent=2))
    return 0 if rep["complete"] else 1


def cmd_workflow(a: argparse.Namespace) -> int:
    state, p = runner.open_run(Path(a.run))
    store = Path(a.run).resolve() / "handoffs"
    done = frozenset(
        x.name for x in p.leaves() if runner.leaf_status(state, x, store) in runner.DONE
    )
    text = wf.emit(p, Path(a.run).resolve(), max_finish=a.max_finish, done=done)
    if done:
        print(f"already done, not re-run: {sorted(done)}", file=sys.stderr)
    if a.out:
        Path(a.out).write_text(text)
        print(f"workflow script -> {a.out}")
    else:
        print(text)
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    """Copy the latest published content of each kind to `<to>/<kind>/` (fixture layout)."""
    store = Path(a.run).resolve() / "handoffs" if a.run else Path(a.store).resolve()
    p = runner.open_run(Path(a.run))[1] if a.run else None
    to = Path(a.to).resolve()
    kinds = a.kind or sorted({s.kind for s in hstore.slots(store) if s.published})
    for k in kinds:
        slot = hstore.latest_published(store, k)
        if slot is None:
            print(f"  skip {k}: nothing published")
            continue
        dst = to / k
        if dst.exists():
            print(f"  skip {k}: {dst} exists (never overwritten)")
            continue
        shutil.copytree(slot.content, dst, symlinks=True)
        print(f"  {k}: {slot.vdir} -> {dst}")
        if p is not None:
            dynamic = [
                n
                for n in p.validators_for(p.producer_of(k), k)
                if (p.validators.get(n, {}).get("tags") or {}).get("logic_source")
                == "external_dynamic"
            ]
            if dynamic:
                print(
                    f"    note: {', '.join(dynamic)} re-measure the world; this fixture may only pass where it was made"
                )
    print("\nTo feed these to agent_sys, a program leaf publishes each (validated on publication):")
    for k in kinds:
        print(
            f'  cp -a "$AGENT_SYS_TASK_PACKAGE/assets/fixture/{k}/." "$AGENT_SYS_OUTPUT_{pkg_mod.env_name(k)}/"'
        )
    return 0


# -- parser ------------------------------------------------------------------- #


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="astk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = top.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("history", help="Claude Code session history").add_subparsers(
        dest="hcmd", required=True
    )
    s = h.add_parser("list", help="list sessions of a project")
    s.add_argument("--project", help="~/.claude/projects/<slug> dir (default: from --cwd)")
    s.add_argument("--cwd", help="the working dir the sessions ran in (default: $PWD)")
    s.add_argument("--since", help="only sessions active at/after this ISO time prefix")
    s.add_argument("--until", help="only sessions that started at/before this ISO time prefix")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_history_list)
    s = h.add_parser("digest", help="digest sessions into reviewable markdown (secrets redacted)")
    s.add_argument("--project")
    s.add_argument("--cwd")
    s.add_argument(
        "--session", action="append", help="id prefix | latest | all (repeatable; default latest)"
    )
    s.add_argument("--since", help="ISO time prefix, e.g. 2026-10-14T00")
    s.add_argument("--until")
    s.add_argument("--grep", help="keep only entries matching this regex")
    s.add_argument("--max-text", type=int, default=1500)
    s.add_argument("--no-tools", action="store_true", help="omit tool-call lines")
    s.add_argument("--out")
    s.add_argument(
        "--split-chars", type=int, default=0, help="split --out into parts of at most N chars"
    )
    s.set_defaults(fn=cmd_history_digest)

    s = sub.add_parser("show", help="agent_sys's own static check of a package")
    s.add_argument("package")
    _add_var(s)
    s.set_defaults(fn=cmd_show)
    s = sub.add_parser("inspect", help="leaves in order, dependencies, kinds, validators")
    s.add_argument("package")
    _add_var(s)
    s.add_argument("--json", action="store_true")
    _add_unset(s)
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("scaffold", help="plan.yaml -> package, then `show` it")
    s.add_argument("plan")
    s.add_argument("dest")
    s.add_argument("--force", action="store_true")
    _add_var(s)
    s.set_defaults(fn=cmd_scaffold)

    s = sub.add_parser("template", help="README + required items skeleton for a kind")
    s.add_argument("--package", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--out", required=True)
    _add_var(s)
    _add_unset(s)
    s.set_defaults(fn=cmd_template)
    s = sub.add_parser("check", help="seal pre-check + every validator of a kind, on a content dir")
    s.add_argument("--package", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--content", required=True)
    s.add_argument(
        "--closure", help="producing task (adds its phase validators); default: the kind's producer"
    )
    s.add_argument(
        "--work", help="where validator zones go (default: $TMPDIR/astk-<uid>/validation)"
    )
    s.add_argument("--timeout", type=float, default=vmod.DEFAULT_TIMEOUT)
    s.add_argument(
        "--precheck-only", action="store_true", help="seal + gate test only, no validator"
    )
    s.add_argument("--json", action="store_true")
    _add_var(s)
    _add_unset(s)
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser(
        "seal", help="copy content into an agent_sys-shaped store; validate; publish on pass"
    )
    s.add_argument("--package", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--content", required=True)
    s.add_argument("--store", required=True)
    s.add_argument(
        "--task-id",
        default="claude-code",
        help="producer label, e.g. <task>#<n>; recorded in validation.yaml (manifest gets its UUID)",
    )
    s.add_argument(
        "--work", help="where validator zones go (default: $TMPDIR/astk-<uid>/validation)"
    )
    s.add_argument("--timeout", type=float, default=vmod.DEFAULT_TIMEOUT)
    _add_var(s)
    _add_unset(s)
    s.set_defaults(fn=cmd_seal)
    s = sub.add_parser("audit", help="size / secret / placeholder audit of a dir")
    s.add_argument("dir")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("env", help="capture the environment (markdown or json)")
    s.add_argument("--repo", help="a git checkout to record branch/commit of")
    s.add_argument("--verbose", action="store_true", help="full GPU tool output")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_env)
    s = sub.add_parser("digest", help="agent_sys tree digest of a content dir")
    s.add_argument("content")
    s.set_defaults(fn=cmd_digest)

    s = sub.add_parser("run-init", help="create (or reopen) a run dir for a package")
    s.add_argument("--run", required=True)
    s.add_argument("--package", required=True)
    _add_var(s)
    s.set_defaults(fn=cmd_run_init)
    s = sub.add_parser("seed", help="publish an external input into a run, validated")
    s.add_argument("--run", required=True)
    s.add_argument("--kind", required=True)
    s.add_argument("--content", required=True)
    s.add_argument("--timeout", type=float, default=vmod.DEFAULT_TIMEOUT)
    s.set_defaults(fn=cmd_seed)
    for name, fn, helptext in (
        ("prepare", cmd_prepare, "stage a leaf: zone, inputs, output slots, env.sh, brief.md"),
        ("run-program", cmd_run_program, "run a program leaf's entry.sh as the runtime would"),
        ("finish", cmd_finish, "the gate: seal test + validators on every output; publish on pass"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--run", required=True)
        s.add_argument("--leaf", required=True)
        if name != "prepare":
            s.add_argument(
                "--timeout",
                type=float,
                default=None if name == "run-program" else vmod.DEFAULT_TIMEOUT,
            )
        s.set_defaults(fn=fn)
    s = sub.add_parser("status", help="per-leaf status of a run")
    s.add_argument("--run", required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("closure", help="end-of-run report; exit 0 only if complete")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_closure)
    s = sub.add_parser("workflow", help="emit a Workflow script executing the run's leaf graph")
    s.add_argument("--run", required=True)
    s.add_argument("--out")
    s.add_argument("--max-finish", type=int, default=3)
    s.set_defaults(fn=cmd_workflow)
    s = sub.add_parser("export", help="copy published handoffs out as fixtures (<to>/<kind>/)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--store")
    s.add_argument("--to", required=True)
    s.add_argument("--kind", action="append")
    s.set_defaults(fn=cmd_export)
    return top


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        return int(a.fn(a))
    except runner.NotReady as exc:
        print(f"astk: {exc}", file=sys.stderr)
        return 1
    except (pkg_mod.PackageError, scaffold.PlanError, FileNotFoundError, Malformed) as exc:
        print(f"astk: {exc}", file=sys.stderr)
        return 2
