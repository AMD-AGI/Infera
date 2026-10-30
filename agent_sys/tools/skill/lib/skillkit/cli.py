"""`astk` — command line for the agent_sys skills. Run `astk <command> -h`.

Exit codes: 0 ok / passed, 1 refused / not passed, 2 usage or load error,
3 knowledge repo not configured, 4 astk itself failed (a crash, or agent_sys's
dependencies are missing) — never a verdict. `mode --prompt` answers 0 engage /
1 step aside; anything else from it is not an answer.

`mode` and `render` need only the standard library (the entry skills' gate must
work on any python3); every other command needs agent_sys's dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from skillkit import AGENT_SYS
from skillkit import settings as settings_mod

try:
    from handoff.errors import Malformed
    from skillkit import audit as audit_mod
    from skillkit import history, hstore, remote, runner, scaffold, wf
    from skillkit import pkg as pkg_mod
    from skillkit import search as search_mod
    from skillkit import validate as vmod
    from task_graph.ids import HandoffId
except ImportError as _exc:  # reported by main() for the commands that need them
    _MISSING: ImportError | None = _exc
else:
    _MISSING = None

SKILL_ROOT = AGENT_SYS / "tools" / "skill"


def _vars(raw: list[str] | None) -> dict[str, str]:
    out = {}
    for item in raw or []:
        k, sep, v = item.partition("=")
        if not sep or not k:
            print(f"astk: --var wants K=V, got {item!r}", file=sys.stderr)
            raise SystemExit(2)
        out[k] = v
    return out


def _emit(obj: object, as_json: bool, text: str) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False) if as_json else text)


def _load(a: argparse.Namespace) -> pkg_mod.Package:
    return pkg_mod.load(a.package, _vars(a.var), allow_unset=a.allow_unset_vars)


def _tags(p: pkg_mod.Package, validator: str) -> dict:
    return p.validators.get(validator, {}).get("tags") or {}


# -- history ------------------------------------------------------------------ #


def _project(a: argparse.Namespace) -> Path:
    return Path(a.project) if a.project else history.project_dir_for(a.cwd or os.getcwd())


def cmd_history_list(a: argparse.Namespace) -> int:
    project = _project(a)
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
    opts = history.DigestOptions(
        since=a.since, until=a.until, grep=a.grep, max_text=a.max_text, tools=not a.no_tools
    )
    parts, scripts = [], []
    for path in history.resolve_sessions(_project(a), a.session or ["latest"]):
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
    if not a.out:
        print(text)
        return 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for sid, i, body in scripts:
        out.with_name(f"{out.stem}.{sid}.workflow-{i}.js").write_text(history.redact(body))
    if scripts:
        print(f"{len(scripts)} Workflow script(s) written beside {out} as *.workflow-N.js")
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
    p = _load(a)
    leaves = p.leaves()
    kinds = {
        k: {
            "content_type": v.get("content_type"),
            "validators": p.validators_for(p.producer_of(k), k),
            "producer": p.producer_of(k),
            "items_schema": v.get("items_schema"),
        }
        for k, v in p.kinds.items()
    }
    lines = [f"package {p.root}", "", "leaves (execution order):"]
    for leaf in leaves:
        lines.append(f"  {leaf.name:<28} {leaf.agent_kind:<8} in={leaf.inputs} out={leaf.outputs}")
        lines.append(f"  {'':<28} waits for {leaf.deps or '—'}; under {' > '.join(leaf.ancestors)}")
        if leaf.external_inputs:
            lines.append(f"  {'':<28} EXTERNAL inputs (seed them): {leaf.external_inputs}")
    lines += ["", "kinds (validators: cost/strength/logic_source):"]
    for k, v in kinds.items():
        vs = ", ".join(
            f"{n} ({_tags(p, n).get('cost')}/{p.validators.get(n, {}).get('strength')}/"
            f"{_tags(p, n).get('logic_source')})"
            for n in v["validators"]
        )
        lines.append(f"  {k:<36} {v['content_type']:<16} by {v['producer'] or 'EXTERNAL'}")
        lines.append(f"  {'':<36} {vs}")
    lines += [f"warning: {w}" for w in p.warnings]
    data = {
        "package": str(p.root),
        "leaves": [leaf.__dict__ for leaf in leaves],
        "non_leaves": [n for n in p.closures if not p.is_leaf(n)],
        "kinds": kinds,
        "warnings": p.warnings,
    }
    _emit(data, a.json, "\n".join(lines))
    return 0


def cmd_scaffold(a: argparse.Namespace) -> int:
    dest = Path(a.dest).resolve()
    written = scaffold.scaffold(scaffold.load_plan(Path(a.plan)), dest, force=a.force)
    print(f"wrote {len(written)} files under {dest}")
    rc, out = pkg_mod.show(dest, _vars(a.var))
    print(out)
    print("agent_sys `show`: " + ("OK" if rc == 0 else f"FAILED (exit {rc})"))
    if rc != 0:
        print(f"The package was written to {dest} anyway; fix the plan and re-run with --force.")
    return 0 if rc == 0 else 1


# -- handoffs ----------------------------------------------------------------- #


def cmd_template(a: argparse.Namespace) -> int:
    p = _load(a)
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
        readme = (p.root / leaf.readme).read_text(errors="replace")
        hint = _section(readme, ("write", "deliver", "output"))
        if hint:
            print(f"\nWhat the producing task `{producer}` says to write (from its readme):\n")
            print(hint)
    return 0


def _section(markdown: str, words: tuple[str, ...]) -> str:
    """The first heading section whose title contains one of `words`."""
    out, depth = [], 0
    for ln in markdown.splitlines():
        stripped = ln.lstrip("#")
        level = len(ln) - len(stripped)
        if level and stripped.startswith(" "):
            if out and level <= depth:
                break
            if not out and any(w in stripped.lower() for w in words):
                depth = level
                out.append(ln)
                continue
        if out:
            out.append(ln)
    return "\n".join(out).strip()


def _work(a: argparse.Namespace, label: str) -> Path:
    """A fresh validator-zone dir. By default under a private dir in the temp dir,
    never beside the content (which may be inside a repo). That dir must be ours:
    owned by this uid, mode 0700, not a symlink — a shared /tmp is not trusted."""
    if a.work:
        base = Path(a.work)
    else:
        private = Path(tempfile.gettempdir()) / f"astk-{os.getuid()}"
        private.mkdir(mode=0o700, exist_ok=True)
        st = private.lstat()
        if private.is_symlink() or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise pkg_mod.PackageError(
                f"{private} is not a private dir of this user; pass --work <dir>"
            )
        base = private / "validation"
    base.mkdir(parents=True, exist_ok=True)
    return runner._fresh_dir(base / label)


def cmd_check(a: argparse.Namespace) -> int:
    p = _load(a)
    work = _work(a, f"check.{a.kind}")
    check = vmod.check_kind(
        p,
        a.kind,
        Path(a.content).resolve(),
        work,
        closure=a.closure or p.producer_of(a.kind),
        precheck_only=a.precheck_only,
    )
    _emit(check.to_dict(), a.json, vmod.render(check) + f"\n(validator zones under {work})")
    return 0 if (check.passed or (a.precheck_only and not check.precheck)) else 1


def cmd_seal(a: argparse.Namespace) -> int:
    """Copy a content dir into an agent_sys-shaped store, validate, publish on pass."""
    p = _load(a)
    store = Path(a.store).resolve()
    content = Path(a.content).resolve()
    same = hstore.find_digest(store, a.kind, hstore.digest(content))
    if same is not None:
        print(f"already published, identical content: {same.vdir} (nothing written)")
        return 0
    work = _work(a, f"seal.{a.kind}")
    slot, check = runner.admit(
        p, store, a.kind, content, work, label=a.task_id, closure=p.producer_of(a.kind)
    )
    print(vmod.render(check))
    if not slot.published:
        print(f"NOT published: {slot.vdir} holds the content, unsealed (no manifest.yaml)")
        return 1
    manifest = hstore.store(store)[0].get_manifest(HandoffId(slot.hid), slot.version)
    print(f"published {slot.vdir}  sha256={manifest.digest['sha256']}")
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    r = audit_mod.audit(Path(a.dir))
    _emit(r.__dict__ | {"blocking": r.blocking, "ask_user": r.ask_user}, a.json, r.render())
    return 1 if r.blocking else 0


def cmd_digest(a: argparse.Namespace) -> int:
    print(hstore.digest(Path(a.content)))
    return 0


# -- runs ---------------------------------------------------------------------- #


def cmd_run_init(a: argparse.Namespace) -> int:
    state = runner.init(Path(a.run), Path(a.package), _vars(a.var))
    print(f"run {Path(a.run).resolve()} for {state['package']}")
    return cmd_status(argparse.Namespace(run=a.run, json=False))


def cmd_seed(a: argparse.Namespace) -> int:
    check = runner.seed(Path(a.run), a.kind, Path(a.content))
    print(vmod.render(check))
    return 0 if check.passed else 1


def cmd_prepare(a: argparse.Namespace) -> int:
    print(json.dumps(runner.prepare(Path(a.run), a.leaf), indent=2))
    return 0


def cmd_run_program(a: argparse.Namespace) -> int:
    rc = runner.run_program(Path(a.run), a.leaf, timeout=a.timeout)
    log = runner.open_run(Path(a.run))[0]["leaves"][a.leaf]["program_log"]
    tail = Path(log).read_text(errors="replace").splitlines()[-30:]
    print(f"exit {rc}; log {log}\n" + "\n".join(f"  | {ln}" for ln in tail))
    return 0 if rc == 0 else 1


def cmd_finish(a: argparse.Namespace) -> int:
    for c in runner.finish(Path(a.run), a.leaf):
        print(vmod.render(c))
    status = runner.open_run(Path(a.run))[0]["leaves"][a.leaf]["status"]
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
        print(
            f"  {leaf.name:<28} {leaf.agent_kind:<8} {runner.leaf_status(state, leaf):<15} "
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
    done = frozenset(x.name for x in p.leaves() if runner.leaf_status(state, x) in runner.DONE)
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
        dst = to / k
        if slot is None or dst.exists():
            why = "nothing published" if slot is None else f"{dst} exists (never overwritten)"
            print(f"  skip {k}: {why}")
            continue
        shutil.copytree(slot.content, dst, symlinks=True)
        print(f"  {k}: {slot.vdir} -> {dst}")
        dynamic = [
            n
            for n in (p.validators_for(p.producer_of(k), k) if p else [])
            if _tags(p, n).get("logic_source") == "external_dynamic"
        ]
        if dynamic:
            print(
                f"    note: {', '.join(dynamic)} re-measure the world;"
                " this fixture may only pass where it was made"
            )
    print("\nTo feed these to agent_sys, a program leaf publishes each (validated on publication):")
    for k in kinds:
        out = f"$AGENT_SYS_OUTPUT_{pkg_mod.env_name(k)}"
        print(f'  cp -a "$AGENT_SYS_TASK_PACKAGE/assets/fixture/{k}/." "{out}/"')
    return 0


# -- the knowledge repo --------------------------------------------------------- #


def cmd_remote_init(a: argparse.Namespace) -> int:
    cfg = remote.init(a.url, user=a.user, clone=a.clone, branch=a.branch)
    print(f"configured {remote.config_path()}\n  url {cfg.url}\n  clone {cfg.clone}")
    ok, why = remote.can_push(cfg)
    push = "ok" if ok else f"DENIED — backups will fail until you have write access: {why}"
    print(f"  user {cfg.user}\n  push: {push}")
    return 0


def cmd_remote_status(a: argparse.Namespace) -> int:
    try:
        cfg = remote.load()
    except remote.NotConfigured as exc:
        print(str(exc))
        return 3
    st = remote.status(cfg, offline=a.offline)
    _emit(st, a.json, "\n".join(f"{k}: {v}" for k, v in st.items()))
    return 0


def cmd_remote_sync(a: argparse.Namespace) -> int:
    print(f"up to date at {remote.sync(remote.load())}")
    return 0


def cmd_remote_list(a: argparse.Namespace) -> int:
    rows = remote.items(remote.load())
    lines = [
        f"  {r.type:<8} {'release' if r.tier == 'release' else f'{r.user}/{r.tier}':<28} "
        f"{r.name:<40} {r.meta.get('summary', '')[:80]}"
        for r in rows
    ]
    _emit([r.__dict__ for r in rows], a.json, "\n".join([*lines, f"{len(rows)} items"]))
    return 0


def cmd_remote_backup(a: argparse.Namespace) -> int:
    out = remote.backup(
        remote.load(),
        Path(a.path),
        name=a.name,
        summary=a.summary or "",
        tags=a.tag or [],
        context=_vars(a.context),
        allow_big=a.allow_big,
        allow_unscanned=a.allow_unscanned,
        dry_run=a.dry_run,
    )
    if a.dry_run:
        print(f"dry run: would publish {out['files']} files as {out['relative']}; nothing pushed")
    else:
        print(f"backed up to {out['relative']} (pushed)")
    return 0


def cmd_remote_promote(a: argparse.Namespace) -> int:
    print(f"promoted to staging: {remote.promote(remote.load(), a.type, a.name)['path']}")
    return 0


def cmd_remote_release_pr(a: argparse.Namespace) -> int:
    print(json.dumps(remote.release_pr(remote.load(), a.type, a.name), indent=2))
    return 0


def cmd_remote_search(a: argparse.Namespace) -> int:
    try:
        cfg = remote.load()
    except remote.NotConfigured:
        cfg = None
        print("knowledge repo not configured: searching the local agent_sys examples only")
    if cfg is not None and not a.no_sync:
        try:
            print(f"synced {remote.sync(cfg)}")
        except remote.RemoteError as exc:
            print(f"sync failed, searching the local clone as it is: {str(exc).splitlines()[0]}")
    types = tuple(a.type or ("task", "handoff"))
    res = search_mod.search(
        cfg, " ".join(a.query), limit=a.limit, types=types, local=not a.no_local
    )
    if a.json:
        _emit({"searched": res.searched, "hits": [h.__dict__ for h in res.hits]}, True, "")
        return 0
    print(f"searched {res.searched} items; {len(res.hits)} shown")
    for h in res.hits:
        where = {"release": "release", "local": "local example"}.get(h.tier, f"{h.user}/{h.tier}")
        flags = ("weak " if h.weak else "") + ("unsealed" if h.sealed is False else "")
        print(f"[{h.score:6.2f}] {h.type:<8} {where:<24} {h.name}  {flags.strip()}")
        print(f"         {h.path}")
        if h.summary:
            print(f"         summary: {h.summary[:160]}")
        if h.context:
            print("         context: " + ", ".join(f"{k}={v}" for k, v in h.context.items()))
        created = f"   created {h.created[:10]}" if h.created else ""
        print(f"         matched: {', '.join(h.matched)}{created}")
        for sn in h.snippets:
            print(f"         | {sn}")
    return 0


def cmd_mode(a: argparse.Namespace) -> int:
    changes: dict = {}
    if a.trigger:
        changes["trigger"] = a.trigger
    if a.ask_remote:
        changes["ask_remote"] = a.ask_remote == "yes"
    path = settings_mod.path()
    existed = path.is_file()
    data = settings_mod.save(**changes) if changes else settings_mod.load()
    if "trigger" in changes:
        for p in settings_mod.rerender_all(data["trigger"]):
            print(f"re-rendered {p}")
    where = path if existed or changes else f"defaults; {path} not written"
    ask = "yes" if data["ask_remote"] else "no"
    print(f"trigger: {data['trigger']}  ask_remote: {ask}  ({where})")
    if a.prompt_file == "-":
        prompt = sys.stdin.read()
    else:
        prompt = Path(a.prompt_file).read_text() if a.prompt_file else a.prompt
    if prompt is None:
        return 0
    if settings_mod.engaged(prompt, data, prior_work=a.prior_work):
        print(f"engage: yes ({', '.join(settings_mod.matches(prompt)[:6]) or 'agent_sys named'})")
        return 0
    explicit = data["trigger"] == "explicit"
    why = "agent_sys is not named (explicit mode)" if explicit else "not in the domain"
    print(f"engage: no — {why}")
    return 1


def cmd_render(a: argparse.Namespace) -> int:
    for p in settings_mod.render_gated(SKILL_ROOT):
        print(p)
    return 0


# -- parser ------------------------------------------------------------------- #

#: Arguments several commands share, by placeholder name.
_SHARED = {
    "VAR": (
        "--var*",
        {"metavar": "K=V", "help": "package variable, repeatable (as agent-sys --var)"},
    ),
    "UNSET": (
        "--allow-unset-vars?",
        "fill missing required --vars with a sentinel (judging only;"
        " a validator whose args need one is UNREACHED)",
    ),
    "WORK": ("--work", "where validator zones go (default: $TMPDIR/astk-<uid>/validation)"),
}
_SUFFIX = {"!": {"required": True}, "?": {"action": "store_true"}, "*": {"action": "append"}}


def _cmd(sub, name: str, fn, helptext: str, *args) -> argparse.ArgumentParser:
    """Add subcommand `name` running `fn`. Each arg is a name or (name, help or
    add_argument kwargs); a trailing `!` makes it required, `?` a flag, `*`
    repeatable. VAR, UNSET and WORK stand for the shared arguments above."""
    p = sub.add_parser(name, help=helptext)
    for arg in args:
        flag, extra = arg if isinstance(arg, tuple) else _SHARED.get(arg, (arg, {}))
        kw = {**_SUFFIX.get(flag[-1], {}), **({"help": extra} if isinstance(extra, str) else extra)}
        p.add_argument(flag.rstrip("!?*"), **kw)
    p.set_defaults(fn=fn)
    return p


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="astk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = top.add_subparsers(dest="cmd", required=True)
    one_of = {"choices": ["task", "handoff"]}
    # One `_cmd` call per command, laid out as a table: name, function, help, arguments.
    # fmt: off
    h = sub.add_parser("history", help="Claude Code session history")
    h = h.add_subparsers(dest="hcmd", required=True)
    project = ("--project", "~/.claude/projects/<slug> dir (default: from --cwd)")
    cwd = ("--cwd", "the working dir the sessions ran in (default: $PWD)")
    _cmd(h, "list", cmd_history_list, "list sessions of a project",
         project, cwd,
         ("--since", "only sessions active at/after this ISO time prefix"),
         ("--until", "only sessions that started at/before this ISO time prefix"),
         "--json?")
    _cmd(h, "digest", cmd_history_digest,
         "digest sessions into reviewable markdown (secrets redacted)",
         project, cwd,
         ("--session*", "id prefix | latest | all (repeatable; default latest)"),
         ("--since", "ISO time prefix, e.g. 2026-10-14T00"), "--until",
         ("--grep", "keep only entries matching this regex"),
         ("--max-text", {"type": int, "default": 1500}),
         ("--no-tools?", "omit tool-call lines"), "--out",
         ("--split-chars", {"type": int, "default": 0,
                            "help": "split --out into parts of at most N chars"}))

    _cmd(sub, "show", cmd_show, "agent_sys's own static check of a package",
         "package", "VAR")
    _cmd(sub, "inspect", cmd_inspect, "leaves in order, dependencies, kinds, validators",
         "package", "VAR", "--json?", "UNSET")
    _cmd(sub, "scaffold", cmd_scaffold, "plan.yaml -> package, then `show` it",
         "plan", "dest", "--force?", "VAR")
    _cmd(sub, "template", cmd_template, "README + required items skeleton for a kind",
         "--package!", "--kind!", "--out!", "VAR", "UNSET")
    _cmd(sub, "check", cmd_check, "seal pre-check + every validator of a kind, on a content dir",
         "--package!", "--kind!", "--content!",
         ("--closure", "producing task (adds its phase validators); default: the kind's producer"),
         "WORK", ("--precheck-only?", "seal + gate test only, no validator"), "--json?",
         "VAR", "UNSET")
    _cmd(sub, "seal", cmd_seal,
         "copy content into an agent_sys-shaped store; validate; publish on pass",
         "--package!", "--kind!", "--content!", "--store!",
         ("--task-id", {"default": "claude-code",
                        "help": "producer label, e.g. <task>#<n>; recorded in validation.yaml"
                                " (manifest gets its UUID)"}),
         "WORK", "VAR", "UNSET")
    _cmd(sub, "audit", cmd_audit, "size / secret / placeholder audit of a dir",
         "dir", "--json?")
    _cmd(sub, "digest", cmd_digest, "agent_sys tree digest of a content dir",
         "content")

    _cmd(sub, "run-init", cmd_run_init, "create (or reopen) a run dir for a package",
         "--run!", "--package!", "VAR")
    _cmd(sub, "seed", cmd_seed, "publish an external input into a run, validated",
         "--run!", "--kind!", "--content!")
    _cmd(sub, "prepare", cmd_prepare, "stage a leaf: zone, inputs, output slots, env.sh, brief.md",
         "--run!", "--leaf!")
    _cmd(sub, "run-program", cmd_run_program, "run a program leaf's entry.sh as the runtime would",
         "--run!", "--leaf!",
         ("--timeout", {"type": float, "help": "seconds (the runtime has none)"}))
    _cmd(sub, "finish", cmd_finish,
         "the gate: seal test + validators on every output; publish on pass",
         "--run!", "--leaf!")
    _cmd(sub, "status", cmd_status, "per-leaf status of a run",
         "--run!", "--json?")
    _cmd(sub, "closure", cmd_closure, "end-of-run report; exit 0 only if complete",
         "--run!")
    _cmd(sub, "workflow", cmd_workflow, "emit a Workflow script executing the run's leaf graph",
         "--run!", "--out", ("--max-finish", {"type": int, "default": 3}))
    s = _cmd(sub, "export", cmd_export, "copy published handoffs out as fixtures (<to>/<kind>/)",
             "--to!", "--kind*")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--store")

    r = sub.add_parser("remote", help="the shared knowledge repo (tasks/handoffs backup and search)")
    r = r.add_subparsers(dest="rcmd", required=True)
    _cmd(r, "init", cmd_remote_init, "configure once: clone the repo, remember url/user/clone",
         "--url!",
         ("--user", "your directory in the repo (default: git user.email local part)"),
         ("--clone", "where the local clone lives (default: ~/.cache/infera-agent-sys/knowledge)"),
         ("--branch", {"default": "main"}))
    _cmd(r, "status", cmd_remote_status, "config + clone state; exit 3 = not configured",
         "--json?", ("--offline?", "do not fetch"))
    _cmd(r, "sync", cmd_remote_sync, "update the clone from the remote")
    _cmd(r, "list", cmd_remote_list, "every task/handoff in release and every user's staging/draft",
         "--json?")
    _cmd(r, "backup", cmd_remote_backup,
         "copy a task package or handoff into <user>/draft, commit, push",
         ("path", "a task package dir, a handoff content dir, or a store version dir"),
         "--name",
         ("--summary", "one line: what it is and what it achieved (shown in search)"),
         "--tag*",
         ("--context*", {"metavar": "K=V",
                         "help": "model=, engine=, engine_version=, gpu=, image=, ..."
                                 " (searchable; judged by prior-work)"}),
         ("--allow-big?", "the user agreed to files over 4 MB (still scanned)"),
         ("--allow-unscanned?", "the user checked by hand what the secret scan cannot read"
                                " (an archive it cannot open, a string in a binary)"),
         ("--dry-run?", "build the commit, push nothing"))
    _cmd(r, "promote", cmd_remote_promote, "draft -> staging after the user's own review",
         ("type", one_of), "name")
    _cmd(r, "release-pr", cmd_remote_release_pr,
         "propose a staging item for release/ as a pull request",
         ("type", one_of), "name")
    _cmd(r, "search", cmd_remote_search, "sync, then rank tasks/handoffs by relevance to a request",
         ("query", {"nargs": "+"}), ("--limit", {"type": int, "default": 10}),
         ("--type*", one_of), "--no-sync?",
         ("--no-local?", "skip this checkout's agent_sys examples"), "--json?")

    _cmd(sub, "mode", cmd_mode,
         "when the entry skill engages: auto | explicit (only if agent_sys is named)",
         ("trigger", {"nargs": "?", "choices": [*settings_mod.TRIGGERS, *settings_mod.ALIASES]}),
         ("--prompt", "exit 0 if the skill should engage on this prompt, 1 if not"),
         ("--prompt-file", "the prompt from a file, or - for stdin (no shell quoting issues)"),
         ("--prior-work?", "(the agent-sys-prior-work gate) a request for prior work engages"
                           " too, in auto mode"),
         ("--ask-remote", {"choices": ["yes", "no"],
                           "help": "offer to configure the knowledge repo when missing"}))
    _cmd(sub, "render", cmd_render, "(installer) write the gated entry skills for the current mode")
    # fmt: on
    return top


#: The commands that run on the standard library alone.
_STDLIB_ONLY = ("mode", "render")


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if a.cmd in _STDLIB_ONLY:
        answers = [(OSError, 2)]  # e.g. a --prompt-file that does not exist: a load error
    elif _MISSING is not None:
        print(
            f"astk: `{a.cmd}` needs agent_sys's dependencies ({_MISSING}); "
            f"pip install -e {AGENT_SYS}",
            file=sys.stderr,
        )
        return 4
    else:
        answers = [
            ((runner.NotReady, remote.Refused), 1),
            (remote.NotConfigured, 3),
            ((pkg_mod.PackageError, scaffold.PlanError, FileNotFoundError, Malformed), 2),
            (remote.RemoteError, 2),
        ]
    try:
        return int(a.fn(a))
    except Exception as exc:  # noqa: BLE001 — anything else is 4: a crash is never a verdict
        code = next((c for types, c in answers if isinstance(exc, types)), 4)
        if code == 4:
            traceback.print_exc()
        else:
            print(f"astk: {exc}", file=sys.stderr)
        return code
