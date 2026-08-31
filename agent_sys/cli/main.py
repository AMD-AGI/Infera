"""`show`, `run`, `run --dry-run`. The demo's entry point. `demo` design §8.

`argparse`, not `click` or `typer`: two verbs and seven flags, and the demo is
the one artefact whose install must be boring. A dependency bought to parse
`--dry-run` is a dependency a reviewer has to install before they can find out
whether anything works.

**This file is a stand-in for the whole-system CLI** (`docs/TODO.md` item 5),
and `demo` design §8.5 says which parts migrate: `run` and `--dry-run` are that
CLI's verbs wearing a demo's name, and when it exists this shrinks to *locate
the package, call the real CLI, keep `show`*. Recorded so the demo is not later
mistaken for a second CLI to maintain.

Exit codes are §8.3's, and the split between 1 and 2 is the one that carries
meaning: refusing to start without a sandbox *"is itself correct behaviour, not
a demo failure"* (spec §3.2), and a reviewer needs to see that without reading
prose.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

from cli import build, expectations, package
from cli.environment import (
    CredentialsMissing,
    Layout,
    LiveHandoffs,
    RepositoryNotPrepared,
    build_context,
    confinement,
    latest_run,
    layout_for,
    preflight_credentials,
    preflight_repository,
)
from cli.events import EventKind
from cli.render.human import HumanRenderer
from cli.render.machine import JsonLinesRenderer
from cli.stream import Stream
from env_mgr import meta
from env_mgr.prepare import EnvManager, permissions_enforced
from env_mgr.protocols import NoConfinement, PrepareRefused, UnresolvedGrant
from monitor import (
    NullUserSink,
    check_liveness,
    install_excepthook,
    reached_the_user,
    start_monitors,
)
from spec_loader import SpecInvalid, task_of
from task_graph import JsonFileStoreMgr, Task, TaskStatus, build_registry, resume_all
from validator import PhaseRunner, StrictLevel

__all__ = ["main"]

log = logging.getLogger("demo")

OK = 0
LOAD_ERROR = 1
PRECONDITION = 2
UNEXPECTED_SUCCESS = 3
UNEXPECTED_FAILURE = 4
#: **The run did not finish, and nothing else said so.** A package with no
#: expectations has declared no failure to look for, and this accounting used to
#: read that as nothing being wrong: a graph with one task stuck in
#: `output_validating`, another in `running` and its only handoff `invalid`
#: exited 0, because zero promises were missed and zero were left unreached.
#: Measured by `main` on `scratch/demo2-2026-08/bringup/n1`.
#:
#: **Its own code, and specifically not 3 or 4.** `UNEXPECTED_SUCCESS` says *a
#: promise stopped being kept*, which is a claim about the system under test and
#: is exactly the cry-wolf this file has twice been wrong about; `UNEXPECTED_FAILURE`
#: says *the promises are untested because something else broke*, which needs
#: promises to be untested. Neither describes a package that promised nothing and
#: a graph that did not finish, so the third thing gets a third number.
INCOMPLETE = 5


#: Variables the CLI supplies itself, and therefore refuses from `--var`.
#:
#: **`outside` is per-run and absolute and only the CLI knows it** — it is
#: `Layout.outside`, created by `_layout` moments before the registry is built.
#: So the CLI's value has to win. Given that, a user `--var outside=...` has two
#: possible behaviours and only one is defensible: silently dropping it leaves a
#: reviewer with a flag they typed, no error, and a run that ignored them, which
#: is the worst of the three outcomes this file spends its length distinguishing.
#: Refused by name instead, at parse time, for every verb — `show` does not pass
#: `outside` at all, but a variable being CLI-owned is a property of the CLI and
#: not of one verb.
RESERVED_VARIABLES = frozenset({"outside"})


def _parse_variables(top: argparse.ArgumentParser, raw: Sequence[str] | None) -> dict[str, str]:
    """`--var K=V`, repeatable, into the map `spec_loader.YamlPackage` expands.

    **What replaced a hardcoded keyword.** `_registry` used to call
    `package.task_package(root, outside=...)`, and that one word was the whole
    variable channel: a package could declare `${n_problems:-12}` and had no way
    to be told otherwise. `examples/demo2` declares three such knobs, and a
    cheap bring-up run of it is `--var n_problems=2`.

    Both faults are `parser.error`, which is argparse's own usage failure: it
    prints usage, names the offending token, and exits 2 — the code an unknown
    flag already gets, so the CLI does not grow a second convention for *you
    typed that wrong*.
    """
    variables: dict[str, str] = {}
    for token in raw or ():
        name, sep, value = token.partition("=")
        if not sep or not name:
            top.error(f"--var {token!r} is not K=V: a '=' is required, and K may not be empty")
        if name in RESERVED_VARIABLES:
            top.error(
                f"--var {name}=... is refused: {name!r} is supplied by the CLI itself and is "
                f"per-run, so a value given here could not be honoured. Accepting it silently "
                f"would leave you holding a flag that did nothing."
            )
        variables[name] = value
    return variables


# --------------------------------------------------------------------------- #
# The parser


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="agent-sys",
        description="The runnable proof that the agent_sys components compose.",
    )
    verbs = top.add_subparsers(dest="verb", required=True)

    show = verbs.add_parser("show", help="print the graph without running it")
    _common(show)

    run = verbs.add_parser("run", help="run the graph end to end")
    _common(run)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate everything, dispatch nothing",
    )
    run.add_argument(
        "--with-broken",
        action="store_true",
        help="also load examples/demo-broken/, a sibling package that is deliberately broken",
    )
    run.add_argument("--resume", action="store_true", help="continue the last run")
    run.add_argument(
        "--allow-repo-config",
        action="store_true",
        help="let the demo set extensions.preciousObjects on the repository it clones",
    )
    run.add_argument("--clean", action="store_true", help="remove every run and exit")
    run.add_argument(
        "--timeout",
        metavar="SECONDS",
        type=float,
        default=_SETTLE_TIMEOUT,
        help=(
            "absolute ceiling on the run, in seconds "
            f"(default {_SETTLE_TIMEOUT:.0f}). A run that stops making progress "
            "ends in seconds regardless; this only bounds one that never stops"
        ),
    )
    return top


def _common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--package", metavar="DIR", help="the task package directory")
    sub.add_argument("--demo-root", metavar="DIR", help="where runs are kept")
    sub.add_argument("--json", metavar="PATH", help="write the machine-readable stream here")
    sub.add_argument(
        "--var",
        metavar="K=V",
        action="append",
        default=[],
        help="set a package variable, repeatable; a package expands it as ${K} or ${K:-default}",
    )


# --------------------------------------------------------------------------- #
# The entry point


def main(argv: Sequence[str] | None = None) -> int:
    top = parser()
    args = top.parse_args(argv)
    # Parsed here rather than by a `type=` callback, because a `--var` naming a
    # reserved variable is a fault about the *set* of them and not about one
    # token, and because `parser.error` wants the parser.
    args.variables = _parse_variables(top, args.var)
    with ExitStack() as stack:
        stream = Stream()
        stream.attach(HumanRenderer())
        if args.json:
            handle: TextIO = stack.enter_context(open(args.json, "w", encoding="utf-8"))
            stream.attach(JsonLinesRenderer(handle))
        try:
            if args.verb == "show":
                return _show(args, stream)
            return _run(args, stream)
        except package.PackageNotFound as exc:
            return _fail(stream, PRECONDITION, str(exc))
        except SpecInvalid as exc:
            # The offending file path is in the message: every `Problem` carries
            # an `origin` and `format_problems` puts it at the head of the line.
            # Criterion 11's second half is that sentence being true.
            return _fail(stream, LOAD_ERROR, str(exc), kind=EventKind.SPEC_REJECTED)
        except (CredentialsMissing, NoConfinement, RepositoryNotPrepared) as exc:
            # The precondition failed, not the run. §8.3 separates 2 from 1 so a
            # reviewer sees which without reading prose.
            return _fail(stream, PRECONDITION, str(exc))
        except (PrepareRefused, UnresolvedGrant) as exc:
            return _fail(stream, PRECONDITION, f"the environment refused the task: {exc}")
    return UNEXPECTED_FAILURE  # pragma: no cover — ExitStack always returns above


def _fail(stream: Stream, code: int, message: str, *, kind: EventKind | None = None) -> int:
    stream.emit(kind or EventKind.RUN_COMPLETE, message, exit_code=code, ok=False)
    return code


# --------------------------------------------------------------------------- #
# show


def _show(args: argparse.Namespace, stream: Stream) -> int:
    """*What is this graph?* Loads, resolves, builds the root, unfolds, prints.

    Dispatches nothing, prepares no environment, needs no credentials and no
    sandbox — and one of `closure` D5's two named callers, which is why §6's
    builder had to exist before this verb could.
    """
    root, registry = _load(args, stream)
    tasks = _graph(registry, stream)
    _describe(registry, tasks, stream)
    stream.emit(
        EventKind.RUN_COMPLETE,
        f"{len(tasks)} tasks in the graph; nothing was dispatched",
        exit_code=OK,
        ok=True,
        package=str(root),
    )
    return OK


# --------------------------------------------------------------------------- #
# run


def _run(args: argparse.Namespace, stream: Stream) -> int:
    if args.clean:
        return _clean(args, stream)
    if args.dry_run:
        return _dry_run(args, stream)
    return _real_run(args, stream)


def _clean(args: argparse.Namespace, stream: Stream) -> int:
    layout = layout_for(Path(args.demo_root) if args.demo_root else None)
    shutil.rmtree(layout.root / "runs", ignore_errors=True)
    (layout.root / "latest").unlink(missing_ok=True)
    stream.emit(EventKind.RUN_COMPLETE, f"removed {layout.root / 'runs'}", exit_code=OK, ok=True)
    return OK


def _dry_run(args: argparse.Namespace, stream: Stream) -> int:
    """*Would this run?* Resolves every closure, validates every spec,
    dispatches nothing. Criterion 11, and the CI half of §11.

    The difference from `show` is not the amount of work; it is the audience.
    Everything here needs no credentials, no sandbox, no model and no network,
    so CI runs it on every commit — which is Airflow's arrangement and dissolves
    spec §5's apparent contradiction: **CI loads the example, a human runs it.**
    """
    root, registry = _load(args, stream)
    tasks = _graph(registry, stream)
    _describe(registry, tasks, stream)

    # Asserted rather than assumed, because "dispatched nothing" is exactly the
    # kind of claim that stays true by accident until it does not.
    agents = registry.get("agent_mgr").all()
    dispatched = [t for t in tasks if t.status is not TaskStatus.WAITING_HANDOFF]
    stream.emit(
        EventKind.RUN_COMPLETE,
        f"dry run: {len(tasks)} tasks resolved, {len(dispatched)} dispatched, "
        f"{len(agents)} agents instantiated",
        exit_code=OK,
        ok=True,
        tasks=len(tasks),
        dispatched=len(dispatched),
        agents_instantiated=len(agents),
        package=str(root),
    )
    return OK


def _layout(args: argparse.Namespace) -> Layout:
    """This run's directories, or the previous run's under `--resume`.

    `--resume` follows the `latest` symlink rather than taking a path: a
    reviewer who has just interrupted a run knows they want *that* one, and
    asking them for a timestamp they never saw is a worse interface than a
    refusal when there is nothing to resume.
    """
    root = Path(args.demo_root) if args.demo_root else None
    if args.resume:
        previous = latest_run(root)
        if previous is None:
            raise package.PackageNotFound(
                "there is no previous run to resume. Run `agent-sys run` first; "
                "`--resume` follows the `latest` symlink and nothing has pointed it yet."
            )
        return previous
    return layout_for(root).create()


def _real_run(args: argparse.Namespace, stream: Stream) -> int:
    """Everything. Needs credentials, a sandbox, and a model.

    The order of the two preconditions is measured rather than aesthetic: the
    credentials check is 0.6 s and the first zone is ~0.56 s of `git clone`, so
    a reviewer with no credentials waits under a second instead of watching a
    clone they are about to lose.
    """
    confinement()  # raises NoConfinement -> exit 2. Criterion 9.
    preflight = preflight_credentials()  # raises CredentialsMissing -> exit 2. Criterion 6.
    # The third precondition, and it is the one a reviewer cannot guess:
    # `workspace.cut` refuses without `extensions.preciousObjects`, so every
    # output-producing dispatch would die in `prepare`. Checked here rather
    # than fixed, because design O1 is open and mutating the reviewer's
    # repository is not a decision this artefact takes on their behalf.
    preflight_repository(
        _main_repo(package.locate(args.package)), allow_config=args.allow_repo_config
    )
    # **Looked up once, from the package the run was pointed at.** What a package
    # promises will go wrong is the package's fact and not the verb's, so every
    # reader below takes the same set rather than reaching for a module global —
    # which is how the demo-1 pair came to be hardcoded in the first place.
    promises = expectations.for_package(package.locate(args.package))

    layout = _layout(args)
    root = package.locate(args.package)
    # **Read once, at start-up, and it is the run's fact rather than a task's.**
    # `env_mgr.prepare.permissions_enforced()` is the single reader of the
    # variable and this calls it; the demo never learns the name. A function and
    # not a constant, on their advice: a module-level read is taken at import
    # and would answer with whatever the environment held then.
    #
    # `main` has no `Prepared` at all — a non-leaf never executes — so a banner
    # about the run cannot be assembled from per-task facts even where they
    # exist. `Prepared.permissions_enforced` stays a confirmation, not a second
    # source.
    enforced = permissions_enforced()
    stream.emit(
        EventKind.CONFINEMENT_APPLIED,
        f"confinement is available and the backend answered: {preflight!r}",
        mechanism=confinement(),
        enforced=enforced,
        run=str(layout.run),
    )
    if not enforced:
        # **§4.17a, and the reason it is its own kind.** The line above is still
        # true — the machine *has* Landlock and the probe finds it — so a run
        # with the switch on would otherwise print `confined` and run
        # unconfined. A run whose sandbox was switched off and a run on a
        # machine with no sandbox are different facts; the second already exits
        # 2 with a refusal (criterion 9), and this one must be as visible.
        stream.emit(
            EventKind.PERMISSIONS_DISABLED,
            "PERMISSION MANAGEMENT IS OFF for this run. Nothing is confined, no "
            "grant is enforced, and every isolation property this demo otherwise "
            "shows is NOT being demonstrated. A pass here is not evidence that "
            "the sandbox works.",
            enforced=False,
            mechanism_available=confinement(),
            run=str(layout.run),
        )
    report_dropped(stream, promises)

    registry = _registry(root, layout, stream, resume=args.resume, variables=dict(args.variables))
    # F-D7 was `demo`'s finding and `monitor`'s to own, and they took it: the
    # *decision* to spawn a thread is the entry point's, for `install_excepthook`'s
    # reason, but resolving `monitor:*`, taking a daemon each and knowing that
    # stopping is `stop()` **then** `join()` is four steps an entry point would
    # otherwise get right or wrong on its own — and this one got it wrong first,
    # which is how the bug was found. One call now.
    running = start_monitors(registry)
    try:
        if args.resume:
            resume_all(registry)
        else:
            _start(registry, stream)
        _settle(registry, stream, timeout=getattr(args, "timeout", None) or _SETTLE_TIMEOUT)
    finally:
        # Names that did **not** come back, rather than a hang or a silent pass.
        stragglers = running.stop(timeout=5.0)
        if stragglers:
            stream.emit(
                EventKind.RUN_COMPLETE,
                f"monitor loops that did not return: {sorted(stragglers)}",
                stragglers=sorted(stragglers),
                ok=False,
            )
    # Described AFTER the run, not before it: the subgraph does not exist
    # until the root's main phase unfolds, so a graph printed at submit time
    # would be one task long.
    tasks = registry.get("task_mgr").all()
    _emit_graph(stream, tasks, resumed=bool(args.resume))
    _describe(registry, tasks, stream)
    return _report(registry, stream, layout, promises)


# --------------------------------------------------------------------------- #
# Loading and building


def _load(args: argparse.Namespace, stream: Stream) -> tuple[Path, Any]:
    """The load-time half, with no store, no environment and no runner."""
    root = package.locate(args.package)
    # No `outside`: there is no `Layout` on this path and never was, so a
    # package's `${outside:-...}` renders its visibly-unfilled default. That is
    # the point of writing a default rather than leaving the reference bare —
    # and it is what lets `show --var K=V` be a cheap check that a value reaches
    # a spec without preparing a run.
    variables = dict(args.variables)
    packages = [package.task_package(root, **variables)]
    if getattr(args, "with_broken", False):
        packages.append(package.broken_package(root, **variables))
    registry = build_registry(
        packages=packages,
        handoff_root=str(root / ".unused-handoffs"),
        knowledge_root=str(root / ".unused-knowledge"),
    )
    _emit_loaded(root, registry, stream, packages=len(packages))
    return root, registry


def _registry(
    root: Path,
    layout: Layout,
    stream: Stream,
    *,
    resume: bool,
    variables: dict[str, str],
) -> Any:
    """The full wiring: a durable store, the real runner, and an `EnvManager`.

    `JsonFileStoreMgr` rather than `MemoryStoreMgr`, because criterion 12 needs
    a store that survives a process — and its records stay readable with `cat`,
    which is a demo virtue in its own right.
    """
    from agent import Runner  # noqa: PLC0415 — see below

    # The handoff view is built here and **bound after the root**, because
    # `EnvManager(ctx)` is an argument to `build_registry` and every handoff
    # in the run is declared inside it. See `LiveHandoffs`.
    handoffs = LiveHandoffs()
    # **The run path joins the configuration route that already shipped.**
    # `env_mgr.meta` has resolved `--meta` → `$ENV_MGR_META` →
    # `~/.config/env_mgr/meta.json` since the `domain` and `zone` sub-commands
    # existed, and `env-mgr` has read local↔remote mappings out of it — but
    # `build_context` was called with no `mapping` here, so `Context.mapping` was
    # `{}` in every production run this repository has ever made and
    # `prepare.py`'s `if ctx.mapping:` was never once true. `sync.sync`,
    # `sync.remote_root` and the `_REMOTE` half of `paths.zone_env` had no
    # production caller at all.
    #
    # `mapping_roots()` and not the `RemoteMapping`s: it keeps the **weak**
    # mappings, which are the ones with something to copy, and drops
    # `transport`/`target`, which nothing anywhere reads yet. A same-machine
    # mapping needs no transport, so this is enough to make the copy path run
    # and is deliberately not enough to reach another host — that needs a
    # `Connection` on `Context`, and it is not here.
    #
    # No `--meta` flag on `agent-sys`: `$ENV_MGR_META` and the config default are
    # the whole surface for now, and adding a third spelling of one setting to a
    # second CLI is a decision, not a convenience.
    mapping = meta.load(meta.configured_path()).mapping_roots()
    context = build_context(
        layout, main_repo=_main_repo(root), handoffs=handoffs, package=root, mapping=mapping
    )
    registry = build_registry(
        store=JsonFileStoreMgr(layout.store),
        # Loaded on a resume too, and that is not optional: the spec
        # registries are rebuilt empty, and without them `unfold` has no
        # catalogue and `PhaseRunner` selects no validator. That is the
        # intended shape rather than a defect — a spec table is configuration
        # and not state, which is `AgentMgr.resume_system`'s own argument one
        # level up. Measured; F-D2.
        packages=[
            package.task_package(
                root,
                # The user's `--var`s. **`outside=` below is a separate keyword,
                # and that is load-bearing rather than incidental:** Python
                # refuses `f(**{"outside": ...}, outside=...)` with *got multiple
                # values*, so a reserved name that ever reached here would be a
                # crash and never a silently overridden path. `_parse_variables`
                # refuses it first, with a message; this is what stands behind
                # the message if the message is ever removed.
                **variables,
                # Criterion 8's leak target, and the only route it has: it is
                # per-run and absolute, and nothing in `Prepared.environment`
                # takes a value from here (F-D17). See the `describe` agent's
                # `env` block in `examples/demo/steps/describe.yaml`.
                #
                # **The only variable the CLI itself supplies.** `package_root` and
                # `store_root` used to be passed beside it: the first filled body
                # paths and the assets convention now finds those, and the second
                # was referenced by no spec in the package — measured by grep over
                # every source before it was dropped, not assumed.
                outside=str(layout.outside),
            )
        ],
        env=EnvManager(context),
        handoff_root=str(layout.handoffs),
        knowledge_root=str(layout.knowledge),
    )
    # `interfaces.md` §5.9: `install_excepthook` is named by §2 and called by
    # nobody, and `task_graph` declined it for the right reason —
    # `threading.excepthook` is **one slot for the whole interpreter**, so a
    # composition root claiming it takes a decision belonging to whoever owns
    # the process. This IS that owner. Without it, an uncaught exception on an
    # attempt's thread prints a traceback and vanishes: the process lives, the
    # exit code is unchanged, and every producer sees nothing.
    handoffs.bind(registry)
    install_excepthook(recorder=registry.get("recorder"), sink=NullUserSink())
    # `validation_env` is the name `PhaseRunner._build_environment` already
    # resolves for the global configuration, and the demo uses it for one thing
    # that is a workaround and one that is not.
    #
    # **The workaround** is the store root: a script body is handed `args.json`
    # and `inputs.json` — handoff **ids** — in a fresh zone with nothing pointing
    # at the content it must read. F-D5, and `interfaces.md` §5.8 at its widest.
    #
    # **`PATH` is not a workaround and my first report of it was wrong.**
    # Measured (`scratch/impl-2026-08/demo/p2_sh_default_path.py`): POSIX `sh`
    # substitutes a built-in default when none is inherited, so a body starts
    # with `/usr/local/sbin:...:/bin` and finds `python3` either way. Setting it
    # here is choosing a **policy** — which binaries a validation may reach —
    # which `validator` design §8.3 puts in `env_mgr`'s allow-list and which
    # nobody owns yet. Pinning it makes the demo reproducible; it fixes nothing.
    registry.register("validation_env", _validation_env(root, layout))
    # **Both roots, and supplying them is what makes the relative flip a no-op
    # today rather than a regression.** Body paths are package-relative now
    # (`_common.schema.json`'s own words), so somebody has to resolve them:
    #
    # - `Runner(package_root=)` reproduces exactly what the absolute fill used
    #   to produce. When `agent` lands per-attempt resolution it prefers the
    #   **staged** root and this becomes the fallback, which is the shape they
    #   described — so this file does not change again.
    # - `PhaseRunner(package_root=)` is the one that would break first.
    #   `build_registry` constructs it as `PhaseRunner(level)` and
    #   `ScriptBodyRunner` joins `self._root / spec.body.entry`, defaulting to
    #   `Path.cwd()` — so a relative validator entry would resolve against
    #   wherever the demo happened to be started from. Re-registered here for
    #   the same reason `runner` is: registration order is free and a name
    #   resolves at use time (`interfaces.md` §2.2).
    registry.register("runner", Runner(registry, package_root=root))
    registry.register("phase_runner", PhaseRunner(StrictLevel.DEFAULT, package_root=root))
    _emit_loaded(root, registry, stream, packages=1)
    return registry


def _validation_env(root: Path, layout: Layout) -> dict[str, str]:
    import os  # noqa: PLC0415 — one use, and only on the run path

    return {
        # **The operator's own allow-list, and the only route by which a
        # validator body can authenticate against a model.**
        #
        # Measured, one run, both phases
        # (`scratch/single-real-task-2026-08/probe_out/`): an *output* phase
        # takes §8.2's PRODUCER row, which is `Prepared.environment`, which
        # `env_mgr/material.py:69` already fills from `harness_env` — so a body
        # shelling out to `claude` there answers `OK`. An *input* phase has no
        # consumer row and falls through to this one, where the same body
        # answered `Not logged in · Please run /login`. The two rows disagreed
        # about whether a validation may reach a model, and only one of them
        # had a reason.
        #
        # `harness_env` is an allow-list of **names**: a key is forwarded only
        # if `~/.claude/settings.json`'s `env` block names it, and its value is
        # read from the live environment. **No value is written down anywhere** —
        # not in a package, not in a `--var`, not in `args.json`. That is the
        # whole reason this is the seam rather than `agent.env` or
        # `validator.args`, both of which would put a credential in a file.
        #
        # Merged **under** the four keys below, deliberately: those are facts
        # about *this run* — where the package is, where the store is, which
        # interpreter — and an operator's settings file is not entitled to an
        # opinion about them. `harness_env`'s own reserved set already refuses
        # `PATH`, `TMPDIR`, `CLAUDE_CONFIG_DIR` and `CLAUDE_CODE_TMPDIR`.
        **_harness_env(),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        # The **checkout**, and deliberately not the staged copy: this block
        # is `validation_env`, which `ScriptBodyRunner` hands a validator
        # body — and a validation zone is placed by `prepare_validation`,
        # which stages the handoffs it validates and **not** the package.
        # A task body prefers `AGENT_SYS_TASK_PACKAGE` and never reaches
        # this; a validator body has no prepared environment and does.
        "AGENT_SYS_DEMO_PACKAGE": str(root),
        "AGENT_SYS_DEMO_STORE": str(layout.handoffs),
        "AGENT_SYS_DEMO_PYTHON": sys.executable,
    }


def _harness_env() -> dict[str, str]:
    """The operator's harness `env` block, or a reported precondition failure.

    `harness_env` raises `ValueError` when a settings file **exists and does not
    parse** — an operator error one character wide, and the alternative it was
    built to remove is an agent that reports `Not logged in` and blames itself.
    Left to escape, it would come out of registry construction as an uncaught
    traceback; `CredentialsMissing` is the family `main()` already maps to the
    PRECONDITION exit code, which is what this is.

    **The message names the file and never a value.** `harness_env`'s own
    message carries the path and the parser's complaint and nothing else, and
    this adds no more.
    """
    from env_mgr.harness import harness_env  # noqa: PLC0415 — one use, run path only

    try:
        return harness_env()
    except ValueError as exc:
        raise CredentialsMissing(str(exc)) from exc


def _main_repo(root: Path) -> Path:
    """The repository `env_mgr` cuts each task's workspace from.

    **The demo's first act is to modify the reviewer's repository** — `env_mgr`
    §7.2 requires `extensions.preciousObjects` on it so an ordinary `gc` cannot
    delete a pack an agent's clone is reading through
    `objects/info/alternates`, and `prepare()` enforces it. It is one reversible
    config key and it is genuinely required, but it happens before anything has
    been demonstrated, so the demo says what it is doing rather than doing it
    silently. `demo` design O1 is whether it should ask first; this does not
    settle that.
    """
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return root


def _emit_loaded(root: Path, registry: Any, stream: Stream, *, packages: int) -> None:
    closures = registry.get("closures")
    stream.emit(
        EventKind.PACKAGE_LOADED,
        f"loaded {packages} task package(s) from {root}",
        package=str(root),
        packages=packages,
        closures=sorted(closures.names()),
        handoff_kinds=sorted(registry.get("handoff_specs").names()),
        validators=sorted(registry.get("validator_specs").names()),
        agents=sorted(registry.get("agent_specs").names()),
    )
    for name in sorted(closures.names()):
        doc = closures.get(name)
        spec = task_of(doc)
        # **`agent` may be absent, and what is reported is what the document
        # says.** A non-leaf declares none — `closure.schema.json` requires one
        # of a leaf and of nothing else — and `task_graph` fills
        # `SUBGRAPH_AGENT_SPEC` at unfold. That fill is a graph fact, not a
        # package fact, so printing it here would report the package as saying
        # something it does not; this event is the *load*.
        declared = doc.get("agent")
        stream.emit(
            EventKind.CLOSURE_RESOLVED,
            f"{name}: agent {declared!r}"
            + (" (a non-leaf declares none)" if not declared else "")
            + f", {len(spec.get('inputs') or ())} in, "
            f"{len(spec.get('outputs') or ())} out",
            closure=name,
            agent=declared,
            inputs=list(spec.get("inputs") or ()),
            outputs=list(spec.get("outputs") or ()),
            validators=list(doc.get("validators") or ()),
            subgraph=[e.get("closure") for e in (spec.get("subgraph") or ())],
        )


def _graph(registry: Any, stream: Stream) -> list[Task]:
    """Build the root, unfold it by hand, wire `depends_on`, report the shape.

    **`show` and `--dry-run` only**, and the reason is a mistake this made once:
    a non-leaf unfolds inside `Task.enter_phase(RUNNING)`, so a caller that
    unfolds *and* submits gets the subgraph twice — four tasks become seven, and
    the second `consume` waits for ever on a handoff nothing produces. Unfolding
    here is what lets these two verbs see the whole graph while dispatching
    nothing; a run does the opposite and submits only the root.
    """
    root = build.root_task("main", registry)
    task_mgr = registry.get("task_mgr")
    task_mgr.add(root)
    tasks = [root, *root.unfold()]
    for subtask in tasks[1:]:
        task_mgr.add(subtask)
    build.wire(tasks)
    _emit_graph(stream, tasks, resumed=False)
    return tasks


def _start(registry: Any, stream: Stream) -> Task:
    """Hand the **root only** to the scheduler, and let the graph grow.

    `submit` is the one entrance that declares a task's outputs, warns about a
    missing `depends_on` and places it in a pool; the subgraph arrives when the
    root's main phase runs, which is `enter_phase(RUNNING)` calling `unfold` and
    submitting each subtask itself.
    """
    root = build.root_task("main", registry)
    registry.get("scheduler").submit(root)
    return root


def _emit_graph(stream: Stream, tasks: Sequence[Task], *, resumed: bool) -> None:
    roots = [task for task in tasks if task.parent is None]
    stream.emit(
        EventKind.GRAPH_BUILT,
        f"{len(tasks)} tasks: {len(roots)} root and {len(tasks) - len(roots)} subtasks",
        tasks=len(tasks),
        root=str(roots[0].id) if roots else None,
        root_parent=None,
        resumed=resumed,
        subtasks=[str(task.id) for task in tasks if task.parent is not None],
        attempts={str(task.id): len(task.history) for task in tasks},
    )


def _describe(registry: Any, tasks: Sequence[Task], stream: Stream) -> None:
    """One line per task: its parent, its phases, its validators.

    Spec §4.2's *"the graph as loaded: tasks, parents, phases, handoffs"*, and
    the fields are what criteria 2, 3 and 4 are asserted over.
    """
    handoff_specs = registry.get("handoff_specs")
    for task in tasks:
        inputs = [task.kinds.get(h, "?") for h in task.inputs]
        outputs = [task.kinds.get(h, "?") for h in task.outputs]
        stream.emit(
            EventKind.PHASE_START,
            f"{task.closure}: input validation runs "
            f"{len(_validators(handoff_specs, inputs)) or 'nothing'}",
            task=str(task.id),
            closure=task.closure,
            parent=str(task.parent) if task.parent is not None else None,
            # **For a non-leaf this is `subgraph`, the system's built-in, and it
            # is the only place the name surfaces.** `TASK_DISPATCHED` — whose
            # human rendering is the sentence *"X dispatched to agent Y"* — is
            # declared in `events.py` and emitted by nothing in this package;
            # only `tests/cli/test_events.py` emits it, for two leaves. So the
            # feared line *"main dispatched to agent 'subgraph'"* is not
            # produced by any run, and *"…to agent 'compose'"* never was either.
            #
            # What a machine reader does get is `agent: "subgraph"` on `main`'s
            # two `phase_start` events, and that is strictly more truthful than
            # `compose`, whose own description was that it executes nothing while
            # being indistinguishable from a leaf's real executor. **The residual,
            # named rather than built:** a consumer must recognise a reserved
            # name to tell a structural executor from a real one; a sibling
            # `structural: true` would let it branch on a flag, and that is a
            # machine-schema addition rather than a phrasing fix.
            agent=task.agent_spec,
            phase="input_validation",
            inputs=inputs,
            outputs=outputs,
            validators=_validators(handoff_specs, inputs),
            is_start=task.is_start,
            is_end=task.is_end,
            depends_on=[str(t) for t in task.depends_on],
        )
        stream.emit(
            EventKind.PHASE_START,
            f"{task.closure}: output validation runs "
            f"{len(_validators(handoff_specs, outputs)) or 'nothing'}",
            task=str(task.id),
            closure=task.closure,
            parent=str(task.parent) if task.parent is not None else None,
            agent=task.agent_spec,
            phase="output_validation",
            inputs=inputs,
            outputs=outputs,
            validators=_validators(handoff_specs, outputs),
            is_start=task.is_start,
            is_end=task.is_end,
            depends_on=[str(t) for t in task.depends_on],
        )


def _validators(handoff_specs: Any, kinds: Sequence[str]) -> list[str]:
    """Which validators a phase over these kinds would run.

    **Read from the handoff kinds, not from the closure**, because that is what
    `validator.PhaseRunner._select` does — and the closure's own `validators`
    list, which both specs call *the phase validators*, has no runtime consumer.
    F-D4 in `README.md`. Rendering what will actually happen rather than what
    the specs say should is the only honest choice for a demo.
    """
    found: list[str] = []
    for kind in kinds:
        if kind not in handoff_specs:
            continue
        for name in handoff_specs.validators_for(kind):
            if name not in found:
                found.append(name)
    return found


# --------------------------------------------------------------------------- #
# Settling, and the report


#: The absolute ceiling on a run, and **the third value this constant has had**.
#: It was 300 s, which killed a healthy model call; it was then 1800 s, which
#: killed a healthy *bring-up* — the run below stopped at exactly 1800.0 s with
#: both tasks still `running`, the handoff still `generating`, and neither
#: validator ever started, while the agent was mid-way through verifying its own
#: `REPRODUCE.md` by running it. Worse than the lost work: the agent is
#: abandoned rather than asked to stop, so the containers and the eight GPUs it
#: held stayed held until a human removed them.
#:
#: **The pattern in both regressions is the same and it is worth naming.** This
#: deadline is the only exit for a run that is *working*, because `holding`
#: deliberately counts a model call as progress so the 20 s stall branch will
#: not fire on one. So every time the unit of work grew — a model call, then a
#: 27 B bring-up — the ceiling became an execution budget for the one case it
#: was never meant to bound, and it did so by reporting a healthy run as a hang.
#:
#: Sized as `_settle`'s docstring argues it should be: generous, because
#: `stall_after` is what actually catches a broken run and this catches only a
#: thread held for ever. Four hours costs a finishing run nothing — quiescence
#: returns the moment nothing is in a phase — and is now reachable as
#: `--timeout` besides, because how long a task legitimately takes is a property
#: of the package, not of this file.
_SETTLE_TIMEOUT = 14400.0


def _settle(
    registry: Any,
    stream: Stream,
    *,
    timeout: float = _SETTLE_TIMEOUT,
    period: float = 5.0,
    stall_after: float = 20.0,
) -> None:
    """Wait until nothing is running, and say so if a monitor stops turning.

    The run ends **quiescent, not empty**: `consume` never leaves
    `WAITING_HANDOFF`, so "wait for every task to finish" would wait for ever.
    Quiescence — no task in a phase state — is the actual end condition, and it
    is the one the demo reports as correct.

    **And a stall is an ending too.** A non-leaf sits in `RUNNING` for as long
    as its subgraph is live, so if a subtask fails, quiescence never arrives
    and the timeout is the only exit — five minutes to be told something that
    was true after twenty seconds. So the loop also ends when no task's status
    or attempt count has moved for `stall_after` and no attempt holds a
    thread, and says which tasks are stuck where.

    **`timeout` was 300 s and it killed the first real model call in this
    repository**, six seconds after the agent finished writing its answer:
    measured, the run started at 09:14 and the SDK transcript's last write was
    09:19. The run reported *"did not settle within 300 s"*, which reads as a
    hang and was a healthy conversation.

    **The short deadline was already obsolete when it did that**, and by this
    function's own doing. The paragraph above is the reason it was short — the
    failure case had no other exit — and `stall_after` took that job: a stuck
    run now ends in twenty seconds and names the task. What is left for the
    absolute deadline is bounding a run where something genuinely holds a
    thread for ever, and for that it should be generous rather than tight.

    **An AI node is what made tight actively wrong.** A program body is
    sub-second and every constant here was chosen when they all were; a model
    call is minutes, and `holding` deliberately counts it as progress so the
    stall branch will not fire on it. So the one path that legitimately takes
    minutes was the one path the deadline could still kill.

    `_SETTLE_TIMEOUT` is a ceiling, not a target — criterion 1's *"under a
    minute excluding model latency"* is untouched, since it excludes exactly the
    thing this bounds, and a run that finishes never reaches the deadline at
    all. See that constant for why it has now moved twice, and for the failure
    mode both moves share.

    **`check_liveness` is called from here**, and this is the place for it:
    `monitor` design says the checker is a comparison of one float called from
    the main thread, *"which is already sitting there"* — and this loop is that
    thread. It is what turns a run that stops into a run that says it stopped.
    An entry point that starts the loops and never checks them has a system that
    fails silently, which is the failure this demo already found once.
    """
    import time  # noqa: PLC0415 — one use, on the run path only

    runner, task_mgr = registry.get("runner"), registry.get("task_mgr")
    monitors = list(registry.resolve("monitor:*"))
    reported: set[str] = set()
    deadline = time.monotonic() + timeout
    seen, last_change = _snapshot(task_mgr), time.monotonic()

    while time.monotonic() < deadline:
        live = [t for t in task_mgr.all() if t.status in _PHASE_STATES]
        # **"Holding a thread" is not "making progress", and the difference is
        # what made a real run hang past this loop.** After a gate failure
        # `TaskAttempt._main` parks on `_await_wake()` — the thread is alive and
        # unhalted, so `is_running` is True for ever, and requiring `not holding`
        # meant the stall branch never fired and only the 300 s timeout ended the
        # run. Measured: the escalation records were 20 s old and this loop was
        # still spinning.
        #
        # A task awaiting a decision is parked, not working, and this package
        # already knows how to tell those apart — `_awaiting_a_decision` reads
        # the escalation that reached the user. So a parked-and-escalated task
        # does not count as holding, while one genuinely mid-model-call does,
        # which is what stops a 20 s window calling a slow backend a stall.
        holding = [
            t for t in live if _is_running(runner, t) and not _awaiting_a_decision(t, registry)
        ]
        # **And a task waiting for a user is waiting for nobody, in this entry
        # point, by construction.** `_run` installs `NullUserSink`, whose
        # `deliver` appends to a list and returns (`monitor/base.py:212-225`) —
        # *"how a monitor reaches a human is unspecified anywhere in this
        # system"*. So an escalation that reaches the top here is terminal, and
        # waiting it out buys nothing.
        #
        # **`holding` alone could not see that, and the reason is structural.**
        # Measured 2026-08-31 on a refused seal: the leaf's body exited 0, its
        # output was refused, it escalated *to its parent* — `target` unset — and
        # its attempt thread stayed `is_running=True` for ever. The root then
        # escalated to the user. So `holding` was permanently 1, contributed by a
        # task that was parked rather than working and whose own escalation
        # carried no `target`, and the stall branch could never fire whatever the
        # rest of the graph did. Two runs died to the absolute deadline this way,
        # one after 65 minutes.
        #
        # Checking the graph rather than the task is what fixes it: *somebody* is
        # blocked on an answer nobody will give, so the run is over — and it does
        # not matter which task is still holding a thread, because whatever it is
        # holding it for cannot arrive. A healthy run has no such escalation and
        # is untouched.
        blocked = [t for t in live if _awaiting_a_decision(t, registry)]
        if not live:
            return  # fully quiescent

        now = _snapshot(task_mgr)
        if now != seen:
            seen, last_change = now, time.monotonic()
        elif (not holding or blocked) and time.monotonic() - last_change > stall_after:
            # **Stalled, not running.** A non-leaf sits in RUNNING for as long as
            # its subgraph is live, so "no task in a phase state" never becomes
            # true when a subtask has failed — and waiting out the whole timeout
            # would cost a reviewer five minutes to be told what was already
            # true after twenty seconds. No status has changed and no attempt
            # holds a thread: nothing is going to happen.
            stalled = sorted(f"{t.closure}:{t.status.value}" for t in live)
            # The two endings are different facts and a reader has to be able to
            # act on which one happened: *nothing is running* is a graph that
            # died, *somebody is waiting on a decision* names a task and a reason
            # and says a human was asked for something this entry point cannot
            # deliver. Reporting both as "stopped making progress" was how the
            # second one read as the first for two runs.
            if blocked:
                why = _awaiting_a_decision(blocked[0], registry)
                message = (
                    f"{blocked[0].closure} is waiting on a decision no one will "
                    f"make — the escalation reached the top and this entry point "
                    f"installs a sink that records and does not answer "
                    f"({why}). Nothing has changed for {stall_after:.0f} s; "
                    f"still in a phase: {', '.join(stalled)}"
                )
            else:
                message = (
                    f"the graph stopped making progress {stall_after:.0f} s ago; "
                    f"still in a phase: {', '.join(stalled)}"
                )
            stream.emit(
                EventKind.RUN_COMPLETE,
                message,
                settled=False,
                stalled_tasks=stalled,
                awaiting_decision=sorted(t.closure for t in blocked),
                ok=False,
            )
            return

        for record in check_liveness(monitors, period=period):
            name = str(record.reported_by or "")
            if name not in reported:
                reported.add(name)
                stream.emit(
                    EventKind.RUN_COMPLETE,
                    f"monitor {name!r} has stopped turning; tasks will not advance",
                    monitor=name,
                    stalled=True,
                    ok=False,
                )
        time.sleep(0.05)

    stream.emit(
        EventKind.RUN_COMPLETE,
        f"the run did not settle within {timeout:.0f} s; "
        f"{len([t for t in task_mgr.all() if t.status in _PHASE_STATES])} task(s) still in a phase",
        settled=False,
        ok=False,
    )


def _snapshot(task_mgr: Any) -> tuple[tuple[str, str, int], ...]:
    """What "progress" means here: any task's status or attempt count moving.

    Cheap and total — it is the whole graph, so a subtask advancing while a
    parent sits still counts, which is exactly the case a naive
    *"is anything RUNNING"* check gets wrong.
    """
    return tuple(sorted((str(t.id), t.status.value, len(t.history)) for t in task_mgr.all()))


_PHASE_STATES = frozenset(
    {TaskStatus.INPUT_VALIDATING, TaskStatus.RUNNING, TaskStatus.OUTPUT_VALIDATING}
)


def _is_running(runner: Any, task: Task) -> bool:
    """Is an attempt still carrying this task?

    A plain call. `attempt_of` is declared in `agent/protocols.py` and this is
    only reached from `_settle`, which only runs on the real-run path — so a
    `getattr` default here would be tolerance for a case that cannot occur, and
    would turn a rename into a run that silently never settles.
    """
    attempt = runner.attempt_of(task.id)
    return attempt is not None and attempt.is_running


def _report(
    registry: Any, stream: Stream, layout: Layout, promises: expectations.ExpectationSet
) -> int:
    """The final state of every task, the verdicts, and the strict accounting.

    **The last part is the one that matters.** Each expected failure that
    happened is an `EXPECTED_FAILURE` and the demo is green; each one that did
    **not** happen is an `UNEXPECTED_SUCCESS` and the demo fails with exit
    code 3. A demo that reports "all good" because a safety property stopped
    holding is the worst outcome available to it.
    """
    task_mgr, handoff_mgr = registry.get("task_mgr"), registry.get("handoff_mgr")
    store = registry.get("handoff_store")
    observed: set[str] = set()

    for task in sorted(task_mgr.all(), key=lambda t: (t.closure or "", str(t.id))):
        waiting = task.status is TaskStatus.WAITING_HANDOFF
        awaiting = "" if waiting else _awaiting_a_decision(task, registry)
        why, source = (
            (_why_waiting(task, handoff_mgr), "handoff_mgr")
            if waiting
            else (awaiting, "escalation")
            if awaiting
            else _why_failed(task, registry)
        )
        stream.emit(
            EventKind.TASK_FINAL_STATE,
            f"{task.closure}: {task.status.value}"
            f"{' — awaiting a decision: ' if source == 'escalation' else ' — ' if why else ''}"
            f"{why}",
            task=str(task.id),
            closure=task.closure,
            status=task.status.value,
            attempts=len(task.history),
            reason=why or None,
            # Which copy answered. `recorder` here means `Execution.detail`
            # was empty, which it should not be — a fallback that fires
            # unnoticed is the defect this whole line exists to fix.
            reason_source=source or None,
            # **§4.17.** Without this a phase that checked nothing and a phase
            # that passed are the same absence in the report. `nothing_ran` in
            # here is the run saying its green covers less than it looks like.
            evidence=_phase_evidence(task, registry) or None,
            expected=bool(promises.observed_by_task(task)),
        )
        seen = promises.observed_by_task(task)
        if seen:
            observed.add(seen)

    for hid in handoff_mgr.all_ids():
        handoff = handoff_mgr.get(hid)
        stream.emit(
            EventKind.HANDOFF_TRANSITION,
            f"{handoff.type} slot v{handoff.latest.version}: {handoff.latest.status.value}",
            handoff=str(hid),
            kind=handoff.type,
            # `slot_version`, never `version`. There are **two** allocators —
            # `Handoff.open_next` numbers the slot, `HandoffStore.put` numbers
            # the stored bytes — and nothing forces them to agree. They both
            # start at 0, so a graph that publishes once per handoff sees
            # `0 == 0` and looks consistent until something re-runs.
            # `interfaces.md` §5.12: the reference between them has no owner, so
            # this stream names which one it is holding rather than implying
            # there is one number. Naming them is the whole of what `demo` can
            # do about §5.12 without inventing the reference.
            slot_version=handoff.latest.version,
            status=handoff.latest.status.value,
        )
        for version in store.list_versions(hid):
            for verdict in store.read_verdicts(hid, version):
                stream.emit(
                    EventKind.VERDICT_RECORDED,
                    f"{verdict.validator}: {'PASS' if verdict.result else 'FAIL'}",
                    validator=verdict.validator,
                    dimension=verdict.dimension,
                    strength=verdict.strength,
                    handoff=str(hid),
                    kind=handoff.type,
                    store_version=version,
                    result=bool(verdict.result),
                    phase="output_validation",
                    task=str(verdict.task_id),
                    expected=bool(promises.observed_by_verdict(verdict, handoff)),
                )
                seen = promises.observed_by_verdict(verdict, handoff)
                if seen:
                    observed.add(seen)

    unreachable = {
        name
        for name, expectation in promises.promises.items()
        if name not in observed and not expectation.was_judged(registry)
    }
    return _strict(stream, observed, unreachable, layout, promises, _completion_gaps(registry))


def _completion_gaps(registry: Any) -> list[str]:
    """What a completed run would **not** have left behind. One name each.

    **Positive evidence, which is the rule this file already applies one level
    up.** `_grounded_verdict_exists` refuses to call a promise judged on an
    absence; this refuses to call a run finished on one. `OK` for a package with
    no expectations therefore requires every task to have reached `SUCCEEDED`
    and no handoff to be sealed `INVALID` — not merely the absence of a
    complaint, which is what an empty expectation set produces for free.

    **Only `SUCCEEDED` counts.** `TaskStatus.TERMINAL` is `{SUCCEEDED,
    CANCELLED}` and a cancelled task is a terminal state in which the work did
    not happen, so it is named here rather than passed.

    **`INVALID` and not "anything but `VALID`", and the limit is stated rather
    than hidden.** `INVALID` is a *sealed* negative verdict and is unambiguous.
    Whether a `SUCCEEDED` task can leave a handoff `CREATED` or `GENERATING` is
    not something this package has measured, so counting those would be a rule
    resting on a guess. Named as a residual: if such a run is ever seen, this is
    where it belongs.

    Read from `task_mgr` and `handoff_mgr` rather than from the events already
    emitted, because a report that audits its own output can only ever agree
    with itself.
    """
    gaps: list[str] = []
    for task in registry.get("task_mgr").all():
        if task.status is not TaskStatus.SUCCEEDED:
            gaps.append(f"{task.closure}: {task.status.value}")
    handoff_mgr = registry.get("handoff_mgr")
    for hid in handoff_mgr.all_ids():
        handoff = handoff_mgr.get(hid)
        if handoff.latest is not None and handoff.latest.status.value == "invalid":
            gaps.append(f"handoff {handoff.type}: invalid")
    return sorted(gaps)


def _strict(
    stream: Stream,
    observed: set[str],
    unreachable: set[str],
    layout: Layout,
    promises: expectations.ExpectationSet,
    unfinished: Sequence[str] = (),
) -> int:
    """pytest's `xfail(strict=True)`, as an exit code — **over three outcomes,
    and a fourth for the package that makes no promise.**

    | | | |
    |---|---|---|
    | observed | `EXPECTED_FAILURE` | green. The demo works |
    | not observed, **reachable** | `UNEXPECTED_SUCCESS` | **exit 3.** A property stopped holding |
    | not observed, **unreachable** | `EXPECTATION_UNREACHED` | **exit 4.** The run never got to test it |

    The third row is new and it is the one this function got wrong. Reporting
    *"did NOT happen — this is a FAILURE, a property stopped holding"* about a
    promise the run never reached is a claim the demo cannot support, and it is
    the mirror of the thing this artefact exists to prevent: as bad to cry wolf
    about a safety property as to stay quiet about one.

    **Exit 4 rather than 3, and the distinction is the whole point.** 3 means *an
    expected failure did not happen*, which is a statement about the system under
    test. When the run stopped early, the honest code is *an unexpected failure*
    — something else broke, and the promises are untested rather than broken.

    **The fourth row, and why the table above could not cover it.** Every row
    is a statement about a *promise*, so a package with no promises falls
    through all three and lands on `OK` — which is how a graph with a task stuck
    in `output_validating` and its only handoff `invalid` exited 0. *There is no
    promised failure to test* and *the run succeeded* are two facts and this had
    one answer for both, which is the same shape as the `was_judged` incidents
    one level up, at the level of the set rather than of a member.

    | | | |
    |---|---|---|
    | no promises, nothing unfinished | `RUN_COMPLETE` | green |
    | no promises, **something unfinished** | `RUN_COMPLETE`, `ok: false` | **exit 5.** The run did not finish |

    **The rule, stated: for an empty expectation set, `OK` requires positive
    evidence of completion.** The caller supplies it as `unfinished` —
    `_completion_gaps`, which is *every task `SUCCEEDED`, no handoff `INVALID`*.

    **And it is applied only to the empty set, which is a gap and not an
    oversight.** A package with promises has declared what its own ending looks
    like, and `examples/demo`'s ending is a task deliberately left in
    `WAITING_HANDOFF` and a handoff deliberately never made valid — every one of
    which `_completion_gaps` names. So a generic completion rule applied there
    would contradict the package's own specification. What is missing is the
    package saying *which* of those it meant, which is the same
    package-declared expectation `cli/expectations.py`'s docstring records as out
    of scope. Until then, a package with promises is checked by its promises and
    a package without is checked by this.

    Pure accounting: both reachability and the completion evidence are computed
    by the caller, which holds the registry, so all four rows are testable
    without a run.
    """
    for name, expectation in promises.promises.items():
        if name in observed:
            stream.emit(
                EventKind.EXPECTED_FAILURE,
                f"{expectation.description} — observed, as the demo promises",
                expectation=name,
                expected=True,
                observed=True,
                reachable=True,
            )
        elif name in unreachable:
            stream.emit(
                EventKind.EXPECTATION_UNREACHED,
                f"{expectation.description} — the run never got this far, so the "
                f"promise is UNTESTED rather than broken. Not evidence either way.",
                expectation=name,
                expected=True,
                observed=False,
                reachable=False,
            )
        else:
            stream.emit(
                EventKind.UNEXPECTED_SUCCESS,
                f"{expectation.description} — did NOT happen, and the run did "
                f"reach it. This is a FAILURE: the demo promises this failure, and "
                f"one that stops happening means a property stopped holding rather "
                f"than that everything is fine.",
                expectation=name,
                expected=True,
                observed=False,
                reachable=True,
            )

    missed = sorted(set(promises.promises) - observed - unreachable)
    untested = sorted(unreachable - observed)
    code = UNEXPECTED_SUCCESS if missed else (UNEXPECTED_FAILURE if untested else OK)
    # **An empty set is green, and it must not be green for the same reason a
    # kept promise is.** With no promises there is nothing to miss and nothing
    # to leave unreached, so the arithmetic above already lands on `OK` — the
    # risk is not the code but the sentence, because *"0 of 0 expected failures
    # observed"* on its own reads as a run whose promises all held. This file
    # exists to keep *untested* and *unexpected* apart; a package that promises
    # nothing is the widest untested case there is, and the report says so
    # rather than letting a reader supply the wrong half.
    nothing_promised = not promises.promises
    # **Only when nothing was promised** — see the docstring's fourth row. The
    # value is computed on every run because it is evidence, and evidence is
    # cheap; what is conditional is whether it decides the exit code.
    gaps = sorted(unfinished) if nothing_promised else []
    if gaps:
        code = INCOMPLETE
    claim = (
        ("this package promises no failure, and the run did NOT finish: " + ", ".join(gaps))
        if gaps
        else "this package promises no failure, so nothing here was tested for one"
        if nothing_promised
        else f"{len(observed)} of {len(promises.promises)} expected failures "
        f"observed, {len(untested)} never reached"
    )
    # **Reported on every run, including the empty one.** "0 dropped" is a
    # claim; a line that appears only when something was dropped leaves a
    # reader unable to tell a full run from one they are reading too quickly.
    stream.emit(
        EventKind.RUN_COMPLETE,
        f"run complete; {claim}, {len(promises.dropped)} validation(s) dropped",
        exit_code=code,
        ok=not missed and not untested and not gaps,
        run=str(layout.run),
        unobserved=missed,
        unreached=untested,
        # **Empty is a claim too**, for the same reason `dropped` is reported at
        # zero: a field that appears only on a bad run leaves a machine reader
        # unable to tell a finished run from one whose report was truncated.
        unfinished=gaps,
        dropped=sorted(promises.dropped),
        # **The machine reader's half of the distinction above.** `ok: true` with
        # `expectations: 0` is a run that made no promise; `ok: true` with a
        # non-zero count is a run whose promises were kept. Without the field
        # those are one line.
        expectations=len(promises.promises),
    )
    return code


def report_dropped(stream: Stream, promises: expectations.ExpectationSet) -> None:
    """Announce every deliberately-skipped validation, **before the run**.

    At the top rather than the bottom, because this is the size of the claim
    the rest of the output makes and a reader who stops early must still have
    seen it. `RUN_COMPLETE` repeats the count for the machine reader, which
    reads the last line rather than the first.

    Emitting nothing when the drop list is empty is deliberate on the human side —
    the summary line carries `0 validation(s) dropped` either way, so silence
    here is not an absence. What must never happen is the reverse: a drop that
    produces no event at all.
    """
    for name, why in sorted(promises.dropped.items()):
        stream.emit(
            EventKind.VALIDATION_DROPPED,
            f"{name} was NOT run: {why}",
            validator=name,
            why=why,
            dropped=True,
        )


def _awaiting_a_decision(task: Task, registry: Any) -> str:
    """Has an escalation for this task reached the top and found no sink?

    **A resting state that looks identical to a hang is not a resting state**,
    and this is the demo's half of that. `monitor` measured the end state after
    fixing F-D16: a handled gate failure legitimately leaves the task `running`
    — criterion 4 says the gate cycle must not move task status — and once the
    escalation reaches the root, *what the alpha does at the top of an
    escalation chain* is `monitor` spec §11, open. `NullUserSink` records the
    arrival and does nothing, deliberately.

    So the task sits in `running` for ever **as specified**, and this package's
    stall detection ends the run by timing out on absence of change. That is a
    heuristic, and it cannot tell *waiting for a human* from *deadlocked* —
    which is the exact distinction the rest of this system spent two days
    learning to make.

    It does not have to be inferred. The escalation that reaches the top is a
    **record**, and the demo already holds a `Recorder`.

    **`monitor.reached_the_user` rather than the two strings this first used.**
    It read `record.kind.value == "escalated" and attributes["target"] == "user"`
    — documented in their design §7.3, so nothing was invented, but declared in
    neither `protocols.py` nor here: two magic strings across a package
    boundary, which by `interfaces.md` §1.2 became frozen the moment this file
    read them. A rename would have flipped the check false silently and put a
    resting state straight back to reading as a hang — **the defect this
    function exists to prevent, in the mechanism it used to prevent it.** They
    built the question; this asks it.
    """
    if "recorder" not in registry or task.current is None:
        return ""
    for record in reversed(registry.get("recorder").read(task.id, task.current.attempt)):
        if reached_the_user(record):
            why = dict(getattr(record, "attributes", None) or {}).get("why")
            return str(why or "escalated to a user")
    return ""


def _why_failed(task: Task, registry: Any) -> tuple[str, str]:
    """Why a task failed, and **which copy of the answer was used.**

    A `final` line naming a status without the reason answers *what* and not
    *why*, and this artefact is where that costs the most: the demo is what a
    reviewer runs, and every wall this stage was found here and then diagnosed by
    somebody reading a different package's source.

    **Measured before it was built**, because there were three possibilities.
    `Execution.detail` — the field whose comment says *"from the runner; for a
    human"* — was empty on every failed task, and `monitor`'s `Recorder` had the
    answer. Not an oversight in the call: `OnDone` was a `Callable` alias that
    **could not express a keyword argument**, so a runner holding an exception had
    the field, the scheduler's parameter and the plumbing in place and no declared
    way to pass the value. `task_graph` widened it to a Protocol and `agent`
    passes `detail` now.

    **So the recorder read is a real fallback again, and it says when it fires.**
    It used to be reached on *every* failure, which made it the primary path
    wearing a fallback's name — and a fallback that is silently load-bearing is
    the same class of defect as the one this function exists to fix. Returning
    the source, and putting it in `fields`, is what makes a reappearance a signal
    rather than a shrug.
    """
    detail = task.current.detail if task.current is not None else ""
    if detail:
        return str(detail), "execution.detail"
    if "recorder" not in registry or task.current is None:
        return "", ""
    # `Recorder.read` rather than the store's raw event records: the event layout
    # is `monitor`'s, and reading it directly would be a second reader of it.
    for record in reversed(registry.get("recorder").read(task.id, task.current.attempt)):
        message = record.exception_message
        if message:
            return f"{record.exception_type}: {message}", "recorder"
    return "", ""


#: The key `agent` writes its phase classification under, on the `PHASE_DONE`
#: record. **A name copied across a boundary, in exactly one place**, so that a
#: rename is one edit and the risk is stated rather than scattered — this is the
#: shape `AGENT_SYS_CLAUDE_CLI` cost a fortnight of green tests over. Reported to
#: `agent` and `validator` as a seam that wants a declared accessor.
EVIDENCE_KEY = "evidence"


def _phase_evidence(task: Task, registry: Any) -> list[str]:
    """How each validation phase of this task ended, in order. **§4.17.**

    `validator` measured the hole and it was real: `blocks_the_task == False`
    and `passed == False` are two different facts, and this report collapsed
    them. A phase that ran nothing does not block — and a reader who sees no
    verdict and no failure concludes the checks passed.

    `PhaseOutcome.evidence` is the classification, and it is already on the
    wire: `agent` puts it on the `PHASE_DONE` record for **both** arms. So this
    needs no new seam — the record is in hand, and this report was reading past
    the field.

    | value | means |
    |---|---|
    | `unchecked` | **nothing checked what this task produced.** §4.15's fault |
    | `nothing_ran` | the phase was not asked — level `NONE`, or no handoff here |
    | `failed` | it ran and something said no |
    | `established` / `low_confidence` | it ran and passed; the strength qualifies it |

    **`nothing_ran` is the one that must never render as a tick.** It is the
    difference between six checks passing and three checks passing while three
    were never asked, which is `interfaces.md` §4.17 — *a green run that is
    itself the corruption.*

    Read through `Recorder.read` rather than the store's event files, for the
    reason `_why_failed` gives: the on-disk layout is `monitor`'s.
    """
    if "recorder" not in registry or task.current is None:
        return []
    out = []
    for record in registry.get("recorder").read(task.id, task.current.attempt):
        value = record.attributes.get(EVIDENCE_KEY)
        if value:
            out.append(str(value))
    return out


def _why_waiting(task: Task, handoff_mgr: Any) -> str:
    unmet = []
    for hid in task.inputs:
        if handoff_mgr.check_if_latest_valid(hid):
            continue
        # `latest` returns `None` for a slot nothing has declared — a real state,
        # not a missing attribute, so this is a None guard and not a `getattr`
        # standing in for one.
        version = handoff_mgr.latest(hid)
        state = version.status.value if version is not None else "not declared"
        unmet.append(f"{task.kinds.get(hid, '?')} is {state}")
    return "; ".join(unmet) if unmet else "no input is valid yet"


if __name__ == "__main__":  # pragma: no cover — the console script calls `main`
    sys.exit(main())
