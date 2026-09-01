"""The `Context` the demo hands `env_mgr.prepare`, and the run's own layout.

`env_mgr` owns every mechanism here. What the demo owns is the context — and
that context turned out to be the module's most surprising research result:
**`env_mgr`'s default granted set does not start the backend.** Three grants
were discovered by being refused, each broke differently, and two of the three
are now in `env_mgr.isolation.policy.DEFAULT_SYSTEM_SET` because they were
general rather than the demo's. What is left here is the third and the ones
that depend on where this machine put things.

| Grant | Symptom without it, measured |
|---|---|
| `/dev/urandom` | The `claude` binary here is a standalone Bun ELF. It aborts in **3 ms** with *"oh no: Bun has crashed. This indicates a bug in Bun, not your code"* and a crash-report URL **for the wrong project**. Granting the whole of `$HOME` read-write does not help. Now in `DEFAULT_SYSTEM_SET` |
| `/run/systemd/resolve/stub-resolv.conf` | `/etc/resolv.conf` is a symlink and **Landlock rules apply to the resolved path**, so granting `/etc` grants the symlink and not its target. `getent` fails `rc=2` in **0.0 s**; `claude -p` **hangs ~184 s** then reports a timeout. One missing file, two tools, symptoms with nothing in common |
| the backend's install directory | `claude` lives under `$HOME`, and `$HOME` is not granted |

**`CLAUDE_CONFIG_DIR` into the zone is what removes the `$HOME` grant
entirely**, and it is not tidiness. Measured: with `~/.claude` granted, the
confined demo agent read the **operator's personal `CLAUDE.md`** and obeyed its
language rule. A demo whose transcript changes with the reviewer's dotfiles is
not a demo, and criterion 13 is a reproducibility claim. `env_mgr.material`
computes it; this module's job is to grant nothing that would let it be
bypassed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from env_mgr.fs.domain import DomainRegistry
from env_mgr.isolation.policy import Granted, Mode, interpreter_grants
from env_mgr.isolation.probe import Availability, probe, select
from env_mgr.protocols import Context, DomainKind, NoConfinement, Tier

__all__ = [
    "CredentialsMissing",
    "RepositoryNotPrepared",
    "Layout",
    "LiveHandoffs",
    "build_context",
    "confinement",
    "demo_grants",
    "latest_run",
    "layout_for",
    "preflight_credentials",
    "preflight_repository",
    "unconfinable",
]

#: The backend the demo's one AI node runs on, and the binary the preflight asks.
BACKEND = "claude"


class CredentialsMissing(RuntimeError):
    """The backend reported that it is not logged in.

    **Its own words, not ours.** Measured with an empty config directory and
    `ANTHROPIC_*` scrubbed: `rc=1`, 0.6 s, and on **stdout** —
    `'Not logged in · Please run /login'`. Identical confined and unconfined.
    So the demo invents no credentials check; it runs the preflight, catches the
    non-zero exit, and surfaces what the backend said, which names what is
    missing better than a rewrite would.

    And it must **not print only stderr**, or it loses the message. That is one
    line of code and it is criterion 6.
    """


class RepositoryNotPrepared(RuntimeError):
    """`extensions.preciousObjects` is not set on the repository to be cloned.

    **A refusal, not a fix, and that is design O1 answered the narrow way.** O1
    asks whether the demo may set it silently, prompt, or refuse without a flag.
    Refusing and naming the exact command takes no decision on the reviewer's
    behalf, which is the only one of the three that is safe to pick alone —
    `--allow-repo-config` is the opt-in for anyone who would rather the demo did
    it.

    It is genuinely required, and `env_mgr` measured why: `man git-clone` warns
    that a borrower *"will become corrupt"*, triggered by an ordinary `git
    commit` in the **source** via automatic maintenance. Reproduced there as
    total rather than degraded — `fatal: bad object HEAD` — so `workspace.cut`
    refuses without it and every output-producing dispatch dies in `prepare`.

    **`ensure_precious` had no production caller**, measured across the tree:
    only its own `__all__` entry, its own `def`, and `tests/env_mgr`. So the
    demo has never cut a workspace in this repository, and `_main_repo`'s
    docstring claimed the demo *"says what it is doing rather than doing it
    silently"* while doing neither. Found by `demo-2` driving the run.
    """


class Layout(NamedTuple):
    """One run's directories. `demo` design §10.1.

    Criterion 13 — *running twice succeeds without hand-editing* — is a
    statement about naming. Two runs produce two sets of `TaskId`s and
    `HandoffId`s and `env_mgr` names a zone `task.<uuid>.<attempt>`, so zones
    never collide. What can collide is the store root, so each run gets its own
    and `latest` points at the newest.

    Nothing is cleaned automatically: criterion 12 needs the previous run's
    state to still be there. `--clean` removes `runs/` and exits.
    """

    root: Path
    run: Path

    @property
    def store(self) -> Path:
        """`JsonFileStoreMgr`'s root — the task, handoff and agent records."""
        return self.run / "store"

    @property
    def handoffs(self) -> Path:
        """`FilesystemStore`'s root — published handoff content."""
        return self.run / "handoffs"

    @property
    def knowledge(self) -> Path:
        return self.run / "knowledge"

    @property
    def zones(self) -> Path:
        """`env_mgr`'s storage domain: one subtree per task attempt."""
        return self.run / "zones"

    @property
    def playground(self) -> Path:
        return self.run / "playground"

    @property
    def outside(self) -> Path:
        """A directory **outside every zone**, for criterion 8's scripted write.

        It is inside the run root so `--clean` reaches it, and outside `zones/`
        so the confinement has something real to refuse. A path under `/tmp`
        would also work and would leave litter a reviewer has to find.
        """
        return self.run / "outside"

    def create(self) -> Layout:
        for path in (
            self.store,
            self.handoffs,
            self.knowledge,
            self.zones,
            self.playground,
            self.outside,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self._point_latest()
        return self

    def _point_latest(self) -> None:
        link = self.root / "latest"
        # `os.replace` on a symlink is atomic and `Path.symlink_to` is not, so
        # an interrupted run cannot leave `latest` absent — which is the one
        # thing `--resume` needs to still be true after `os._exit`.
        tmp = self.root / f".latest.{os.getpid()}"
        tmp.symlink_to(self.run, target_is_directory=True)
        os.replace(tmp, link)


class LiveHandoffs(Mapping):
    """`HandoffId -> Handoff`, read through to `handoff_mgr`. **Not a snapshot.**

    `Context.handoffs` is typed `Mapping[HandoffId, Any]` and `env_mgr.grants`
    reads it to learn a slot's kind — a grant names a **kind**, because it is
    written at declaration time where no instance exists, so resolving one to
    `<root>/<hid>/v<N>/` needs something that can answer *what kind is this
    slot*. Only `handoff_mgr` can.

    **It has to be live, and that is a `Context` ordering fact rather than a
    preference.** A `Context` is built *before* `build_registry`, because
    `EnvManager(ctx)` is one of its arguments; every handoff in the run is
    declared *after*, by `scheduler.submit`. So a dict passed at construction is
    empty for ever, and the demo passed one — `UnresolvedGrant: grant for kind
    'facts' … matches no handoff this attempt has`, measured, with the slot
    right there in the message.

    Empty until `bind`, and **`bind` is not optional**: an unbound view answers
    nothing and every kind-named grant fails loudly, which is the same failure as
    the dict and is at least noisy in the same way.
    """

    def __init__(self) -> None:
        self._mgr: Any = None

    def bind(self, registry: Any) -> LiveHandoffs:
        """Point at the registry's `handoff_mgr`. Called once, after the root."""
        self._mgr = registry.get("handoff_mgr")
        return self

    def __getitem__(self, hid: Any) -> Any:
        if self._mgr is None:
            raise KeyError(hid)
        try:
            return self._mgr.get(hid)
        except KeyError:
            raise KeyError(hid) from None

    def __iter__(self) -> Iterator[Any]:
        return iter(() if self._mgr is None else self._mgr.all_ids())

    def __len__(self) -> int:
        return 0 if self._mgr is None else len(self._mgr.all_ids())


def default_root() -> Path:
    """`$XDG_STATE_HOME/agent-sys-demo`, or `~/.local/state/...`.

    State rather than cache or data: it is *"state that should persist between
    restarts but is not important enough for the data directory"*, which is what
    a demo run is. A cache directory would be correct until somebody cleared it
    between the interrupt and the resume.
    """
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "agent-sys-demo"


def layout_for(root: Path | None = None, *, run_id: str | None = None) -> Layout:
    """A run's layout under `root`, with a **unique** run directory.

    **The timestamp alone was not unique and the collision was silent.**
    `%Y%m%dT%H%M%S` has second resolution, and `create()` uses
    `mkdir(exist_ok=True)`, so two runs started inside one second shared a
    directory and the second adopted the first's store — measured: same `run`
    path, and run B reads a file run A wrote. That is criterion 13 —
    *running twice succeeds without hand-editing* — failing without a word.

    `test_two_runs_do_not_collide` could not see it: it gives each run its own
    **root**, so it tests that two roots do not collide, which nothing threatens.
    The collision is between two runs under **one** root, which is the case a
    reviewer actually creates.

    A suffix rather than a finer clock: `%f` narrows the window and does not
    close it, and two processes can start in the same microsecond. The
    timestamp stays the prefix so `runs/` still sorts chronologically — nothing
    parses these names, but a human reads them.
    """
    base = Path(root) if root is not None else default_root()
    stamp = run_id or f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    return Layout(root=base, run=base / "runs" / stamp)


def latest_run(root: Path | None = None) -> Layout | None:
    """What `--resume` follows. `None` when nothing has run."""
    base = Path(root) if root is not None else default_root()
    link = base / "latest"
    if not link.exists():
        return None
    return Layout(root=base, run=link.resolve())


# --------------------------------------------------------------------------- #
# Confinement


def confinement(availability: Availability | None = None) -> str:
    """Which mechanism will be used, or raise `NoConfinement`.

    **Criterion 9 is this function plus not catching it.** On a machine with
    neither `bwrap` nor Landlock the demo refuses to start and says so, and that
    refusal is correct behaviour rather than a demo failure (spec §3.2) — which
    is why `cli/main.py` gives it exit code 2 and not 1 or 4.

    Takes an `Availability` for `env_mgr`'s own reason: **no machine runs all
    three branches**, so the only way to test the refusal is to inject one.
    """
    return select(availability if availability is not None else probe())


# --------------------------------------------------------------------------- #
# Credentials


def preflight_credentials(*, cli: str = BACKEND, timeout: float = 90.0) -> str:
    """Ask the backend whether it can run at all, **before any zone is built**.

    A reviewer with no credentials then waits 0.6 s rather than four seconds of
    `git clone`, and that ordering is the whole reason this is a separate step
    rather than the first task failing.

    Returns the backend's own reassuring output on success; raises
    `CredentialsMissing` carrying **stdout and stderr both** on failure.

    **It does not test what the run does, and saying so is the point.** This
    runs `claude -p` *unconfined*, against the operator's own config directory.
    A confined task gets a different arm: `material.deploy` points
    `CLAUDE_CONFIG_DIR` into the zone — correctly, it is what removed the `$HOME`
    grant — which also moves away the `env` block in `~/.claude/settings.json`
    holding the endpoint and the auth header. `env_mgr` measured all three arms:

    ::

        operator config dir, no injection    rc=0  OK
        relocated config dir, no injection   rc=1  Not logged in
        relocated config dir, injected       rc=0  OK

    So before `env_mgr/harness.py` injected the block, this check **passed and
    the run then failed**, surfacing `'Not logged in'` — *correct about the
    symptom and wrong about the cause*, because nothing was missing from the
    machine. Fixed on their side; the asymmetry is not, and it is this
    function's to declare rather than theirs.

    **What it establishes is narrow: the backend exists and this operator can
    authenticate.** Whether the *confined* task can is a property of the
    prepared environment, and the first thing that tests it is the run.

    The timeout is 90 s rather than the 5.5 s a warm call takes: a *missing*
    credential answers in 0.6 s, so the long tail is never the case this is
    guarding, and a demo that reported `Not logged in` because the machine
    was busy would send a reviewer to fix the one thing that is not wrong.
    """
    binary = shutil.which(cli)
    if binary is None:
        raise CredentialsMissing(
            f"the {cli!r} backend is not on PATH. The demo's one AI node runs on "
            f"claude-agent-sdk and there is no fake fallback: a demo that silently "
            f"ran against a fake would prove less than it appears to (spec §3.1)."
        )
    try:
        done = subprocess.run(  # noqa: S603 — `binary` came from `shutil.which`
            [binary, "-p", "Reply with exactly one word: ready"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CredentialsMissing(
            f"{cli} did not answer within {timeout:.0f} s. Unconfined that is a "
            f"network or credential problem rather than a sandbox one."
        ) from exc
    if done.returncode != 0:
        raise CredentialsMissing(
            # stdout FIRST, and both. Measured: the message that satisfies
            # criterion 6 — 'Not logged in · Please run /login' — arrives on
            # **stdout**, and a demo that printed only stderr would lose it.
            f"{cli} exited {done.returncode} unconfined, so this operator cannot "
            f"authenticate at all — a confined task would not get further.\n"
            f"  stdout: {done.stdout.strip() or '(empty)'}\n"
            f"  stderr: {done.stderr.strip() or '(empty)'}\n"
            f"  Credentials come from config (spec §3.1); the demo supplies no "
            f"fallback and does not invent a check of its own."
        )
    return done.stdout.strip()


# --------------------------------------------------------------------------- #
# The context


def preflight_repository(main_repo: str | Path, *, allow_config: bool = False) -> str:
    """The third precondition: the repository each zone is cloned from.

    Runs beside the credentials check and before any zone is built, for the same
    reason — a reviewer whose checkout is not prepared should learn it in
    milliseconds rather than after a clone.

    **Two sentences a reviewer needs and one they would not think to ask for.**
    The key is required, it is one command, and — in a **git worktree**, which
    this repository is checked out as — `.git` is a *file* and the key lands in
    the **shared common config**. So it affects the main checkout and every other
    worktree, and it makes `git gc` refuse in all of them. That is reversible and
    it is not local, which is exactly the sort of thing to say before doing.
    """
    from env_mgr.workspace import PRECIOUS, ensure_precious, is_precious

    repo = str(main_repo)
    if is_precious(repo):
        return repo
    if not allow_config:
        raise RepositoryNotPrepared(
            f"{repo} does not set {PRECIOUS}, and `env_mgr.workspace.cut` refuses "
            f"without it — so every output-producing task would die in `prepare`.\n"
            f"  It stops an ordinary `git gc` in this repository destroying the "
            f"history in an agent's `--shared` clone, which git's own manual calls "
            f"corruption rather than degradation.\n"
            f"  Run:      git -C {repo} config {PRECIOUS} true\n"
            f"  Undo:     git -C {repo} config --unset {PRECIOUS}\n"
            f"  Or pass:  --allow-repo-config, and the demo sets it for you.\n"
            f"  Note: in a git worktree this lands in the SHARED common config, so "
            f"it affects the main checkout and every other worktree, and `git gc` "
            f"will refuse in all of them until it is unset."
        )
    ensure_precious(repo)
    return repo


def demo_grants(layout: Layout, *, backend: str = BACKEND) -> tuple[Granted, ...]:
    """What the demo adds to `DEFAULT_SYSTEM_SET`, and why each one is here.

    Everything general has already moved into `env_mgr`. What is left depends on
    where *this machine* put things, which is exactly the class of grant
    `docs/interfaces.md` O2 says is **undiscoverable in advance** — D4's three
    were found by running one binary and watching it break in three different
    ways, and a different harness has a different set found the same way.
    """
    # `/run/systemd/resolve/stub-resolv.conf` is **not** here, and its absence is
    # the finding closing. The demo measured it — `/etc/resolv.conf` is a symlink
    # and Landlock applies to the resolved path, so granting `/etc` gives the
    # symlink and not DNS; `getent` fails in 0.0 s and `claude -p` hangs ~184 s.
    # `env_mgr` took it into `DEFAULT_SYSTEM_SET` as `optional=True`, and
    # `prepare` builds `Policy(DEFAULT_SYSTEM_SET).with_(...)`, so adding it here
    # would be a second writer of a grant that is now general. Verified present
    # in their set before removing it from ours.
    grants = [
        # The zone's siblings a run needs to reach: the store the validators
        # read, and the directory criterion 8's write must NOT reach — granted
        # read-only so that the refusal is a *write* refusal rather than an
        # unreachable path, which is the difference between demonstrating a
        # boundary and demonstrating a typo.
        #
        # **No run-level `content/`.** A task writes its output into the
        # pre-allocated `<store>/<hid>/v<N>/content/`, which `grants.resolve_all`
        # grants per attempt — so a shared writable directory outside the zone
        # was the pre-§4.14 shape and is gone. It was empty on every run.
        Granted(str(layout.handoffs), Mode.READ_EXEC),
        Granted(str(layout.outside), Mode.READ_EXEC),
    ]
    binary = shutil.which(backend)
    if binary is not None:
        # The backend's own install directory. It lives under `$HOME` on every
        # ordinary install, and `$HOME` is granted nowhere — that is §9.3's
        # whole point and this is the narrowest thing that keeps it true.
        grants.append(Granted(str(Path(binary).resolve().parent.parent), Mode.READ_EXEC))
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if ca:
        grants.append(Granted(ca, Mode.READ_EXEC, optional=True))
    return tuple(grants)


def build_context(
    layout: Layout,
    *,
    main_repo: str | Path,
    package: Path | None = None,
    handoffs: Mapping[Any, Any] | None = None,
    tier: Tier = Tier.PRODUCTION,
    mapping: Mapping[str, str] | None = None,
    transports: Mapping[str, Any] | None = None,
) -> Context:
    """The `Context` `EnvManager` binds once.

    `env=EnvManager(ctx)` is passed to `build_registry`, which has **no default**
    for it and deliberately so: a `Context` is composition-time configuration —
    domains, store root, main repository, sync mapping, tier — and none of it is
    derivable at run time. An `EnvManager` over a fabricated one is worse than an
    unregistered name, because a task would then be prepared against an
    environment nobody configured (`docs/interfaces.md` §2.4).

    `mapping` is **supplied by the caller and empty by default**, and empty is a
    legitimate configuration rather than a degradation: spec §3 says no cluster
    and no remote host, so the local↔remote mapping degenerates to a strong
    same-machine mapping and `prepare` skips sync entirely.

    What changed is that empty is now a *configured* answer rather than the only
    one. `cli/main.py` reads `env_mgr.meta` — the `--meta` / `$ENV_MGR_META` /
    `~/.config/env_mgr/meta.json` route that already shipped for `env-mgr
    domain|zone` — and passes its weak mappings here. With no meta file the
    result is `{}` and every run behaves exactly as before; that is the control,
    and it is what the whole test suite runs under.

    This function still reads no environment of its own: *is this run remote* is
    the composition root's decision (`docs/interfaces.md` §2.4), and a
    constructor that consulted `$ENV_MGR_META` itself would make it two.
    """
    domains = DomainRegistry()
    domains.register("demo-storage", str(layout.zones), DomainKind.HANDOFF_STORAGE)
    domains.register("demo-playground", str(layout.playground), DomainKind.PLAYGROUND)
    return Context(
        domains=domains,
        # **Not `dict(...)`.** Copying it here would freeze a view whose whole
        # job is to be live — see `LiveHandoffs`.
        handoffs=handoffs if handoffs is not None else LiveHandoffs(),
        store_root=str(layout.handoffs),
        main_repo=str(main_repo),
        mapping=dict(mapping or {}),
        # Keyed by the same `local_root` as `mapping`, and empty means both ends
        # are on this machine — which is what every run before R1 was.
        transports=dict(transports or {}),
        interpreter_grants=interpreter_grants() + demo_grants(layout),
        # **Staged, not granted — `interfaces.md` §4.16 reversed F19.**
        # `layout.stage_package` copies the package into `<zone>/package/`
        # and `prepare` exports the copy as `AGENT_SYS_TASK_PACKAGE`. The
        # read-exec grant this module used to add is gone: a staged copy is
        # inside the zone, so nothing outside it has to be reachable.
        #
        # `package_stage` stays `None` on `env_mgr`'s advice and I agree with
        # the reasoning: an allow-list of `('bin', 'lib', 'bodies')` would
        # close the demo's package route today and is **a deny-list in
        # allow-list clothing** — a validator directory added later would
        # silently not be excluded. §4.16 accepts that staging *moves*
        # criterion 13's leak rather than closing it, and `TODO.md` 4a is
        # where it closes properly. The honest hole, not the local fix.
        package=str(package) if package is not None else None,
        package_stage=None,
        tier=tier,
        # **Declared, not discovered, and the demo is what declares it.**
        # `prepare` does `agent_cli=resolve_strict(ctx.agent_cli) if
        # ctx.agent_cli else None`, and an unset value makes
        # `ClaudeSdkBackend` refuse with `BackendUnavailable` — correctly.
        # The SDK's `_find_cli` returns its own **bundled** binary before it
        # ever calls `shutil.which`, so a run with this left `None` would
        # either refuse or, if the backend let it through, execute a
        # different build from the one `env_mgr` installed plugins into and
        # succeed without them. Measured: `describe` dispatched and failed
        # here, which is the guard doing its job on a value nobody supplied.
        #
        # `shutil.which` and not a configured path: this is the same lookup
        # `demo_grants` already does to grant the backend's install
        # directory, so the binary granted and the binary run are one
        # decision. `None` when the CLI is absent is the honest answer — the
        # credentials preflight is what turns that into a message.
        agent_cli=shutil.which(BACKEND),
    )


def unconfinable() -> NoConfinement:
    """The message criterion 9 asks for, as a value rather than a raise.

    `cli/main.py` needs to print it and exit 2; a helper that raised would make
    the caller catch its own exception to format it.
    """
    return NoConfinement(
        "no confinement mechanism is available: bwrap is absent and Landlock is "
        "unavailable on this kernel.\n"
        "  The demo refuses to start, which is correct behaviour and not a demo "
        "failure (spec §3.2). Criterion 7 needs a working sandbox, and an agent "
        "started without one runs with the operator's full privileges while the "
        "system reports that it is confined."
    )
