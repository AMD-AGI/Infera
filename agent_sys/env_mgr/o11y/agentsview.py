# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AgentsView, started as a side-car and never allowed to fail a run.

**The one rule this module exists to enforce:** an observability panel that can
break the thing it observes is worse than no panel. Every function here returns
a `Status` and raises nothing. There is a test per failure mode holding that
line, because the failure mode of a warning-only component is that someone
later "improves" it into a raise.

**Why we decide the port instead of letting AgentsView decide.** `agentsview
serve` auto-discovers a free port when the requested one is busy. That is a
sensible default for a human at a terminal and the wrong one here: the mission
asks for a *warning and a skip* on a taken port, and a daemon that quietly
moved to 18889 is a panel nobody knows the address of. So the bind probe
happens here, before launch.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .prefix import Prefix

__all__ = [
    "DEFAULT_PORT",
    "RECIPE_PATH",
    "Status",
    "check_disabled_agents",
    "discover_providers",
    "ensure_installed",
    "ensure_running",
    "port_is_free",
    "resolve_port",
    "write_config",
]

log = logging.getLogger("env_mgr.o11y.agentsview")

#: The mission's number.
DEFAULT_PORT = 18888

PORT_ENV_VAR = "AGENTSVIEW_PORT"


@dataclass(frozen=True)
class Status:
    """What happened. `url` is set only when `running` is true."""

    running: bool
    reason: str
    url: str | None = None


def resolve_port(flag: int | None, environ: Mapping[str, str]) -> int:
    """Flag, then environment, then 18888.

    **An unparseable environment value is the default, not an error.** This is
    a side-car; refusing to start the whole deployment over a typo in a
    variable nobody needed would invert the priority the module is built on.
    """
    if flag is not None:
        return int(flag)
    raw = environ.get(PORT_ENV_VAR)
    if raw is None:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        log.warning(
            "%s=%r is not a port number; using the default %d",
            PORT_ENV_VAR,
            raw,
            DEFAULT_PORT,
        )
        return DEFAULT_PORT


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """A real bind, not a connect.

    Connecting answers "is someone accepting", which is a different question:
    a socket bound and not listening still makes our own bind fail. We ask the
    question we actually need answered.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


#: How long the launch subprocess itself may take. `serve --background`
#: daemonises and returns immediately, so anything slower is a hung binary.
LAUNCH_TIMEOUT_S = 20.0

#: How long we then wait for the daemon to answer. Cold-start reads the whole
#: session archive, so this is generous.
HEALTH_TIMEOUT_S = 30.0

#: How long we spend deciding whether something already on the port is *our*
#: panel rather than a stranger. Short deliberately: this is on the path of
#: every deployment, and the answer either arrives at once or the thing on the
#: port is not an agentsview that would have served us anyway.
#:
#: A named constant rather than a literal so a test can shrink it. The whole
#: module is timeouts, and a suite that must really sleep through them is a
#: suite people stop running.
REUSE_PROBE_TIMEOUT_S = 2.0

#: AgentsView's own JSON endpoint (`docs/session-api.md:112`), used as the
#: identity probe. Not `/`: every web server on the machine answers `/` with a
#: 200, and this component's whole job on an occupied port is telling *our*
#: daemon apart from a stranger's.
IDENTITY_PATH = "/api/v1/agents"

#: Cap on the identity response we read. The endpoint returns a short list of
#: providers; the cap is there so a stranger streaming without end cannot hang
#: a deployment on the o11y probe.
IDENTITY_MAX_BYTES = 1 << 20

#: Written by a successful launch, read by the reuse gate. The record of *which
#: port we put our own daemon on* — and therefore the only thing separating
#: "reuse the panel we started" from "adopt whatever is listening".
PORT_FILE = "agentsview.port"

#: **The pinned, reviewable statement of intent** — every provider AgentsView
#: can scan, minus Claude Code, hand-maintained and version-controlled. This
#: is what `write_config` actually disables; `check_disabled_agents` below is
#: what tells us when reality has moved past it, in *both* directions.
#:
#: Deriving this at runtime from `discover_providers` (tried first, reverted)
#: was rejected: a silent upstream AgentsView change would then silently
#: change what the panel shows, with no commit, no diff, no review — worse
#: than a pinned list that occasionally drifts *loudly*, via the check.
#:
#: Measured against `agentsview v0.42.0 doctor sync`'s own "Agent roots:"
#: report, with `claude` removed and `aider` added back (accepted by the
#: config parser directly even though `doctor sync` prints no root for it in
#: this version). History: a first version of this list was guessed from
#: AgentsView's human-readable provider table and five slugs were wrong
#: (`claude-cowork`, `command-code`, `copilot-cli`, `cortex-code`,
#: `gemini-cli` for `cowork`, `commandcode`, `copilot`, `cortex`, `gemini`) —
#: caught from a real acceptance run's `stderr`, not from re-reading the docs.
#: To re-measure after an AgentsView upgrade, see `discover_providers`'s
#: docstring for the exact command; `check_disabled_agents` runs it
#: automatically at install time and warns if this tuple has drifted from it.
OTHER_PROVIDERS = (
    "aider", "amp", "antigravity", "antigravity-cli", "codebuff", "codex",
    "commandcode", "copilot", "cortex", "cowork", "cursor", "cursor-ide",
    "deepseek-harness", "deepseek-tui", "devin", "forge", "gemini", "goose",
    "gptme", "grok", "hermes", "icodemate", "iflow", "kilo", "kilo-legacy",
    "kimi", "kimi-work", "kiro", "kiro-ide", "mimocode", "omnigent", "omp",
    "openclaude", "openclaw", "opencode", "openhands", "pi", "piebald",
    "poolside", "posit-assistant", "positron", "prime-agent", "qclaw",
    "qoder", "qwen", "qwenpaw", "reasonix", "roocode", "shelley", "trae",
    "traex", "vibe", "visualstudio-copilot", "vscode-copilot", "warp",
    "windsurf", "workbuddy", "zcode", "zed", "zencoder",
)

#: The provider `OTHER_PROVIDERS` must never contain: gate 3 exists to disable
#: everything *except* the one source `agent_sys` itself writes.
_KEEP_ENABLED = "claude"

#: `agentsview doctor sync`'s report has an "Agent roots:" section listing
#: every provider name the binary recognizes, one two-space-indented
#: `name: path (status)` line per root (a provider with several default
#: search paths repeats its name on several lines). Measured directly against
#: a real v0.42.0 binary — this is the mechanical enumeration `discover_providers`
#: parses, not a re-derivation of AgentsView's own provider table.
_AGENT_ROOTS_HEADER = "Agent roots:"
_AGENT_ROOT_LINE_RE = re.compile(r"^  ([a-z0-9_-]+):", re.MULTILINE)

#: How long `doctor sync` may take. It only stats candidate directories and
#: reads existing sync-state metadata — no full session parse — so this is
#: short relative to the other probes in this module.
DISCOVER_PROVIDERS_TIMEOUT_S = 10.0


def discover_providers(prefix: Prefix) -> tuple[str, ...] | None:
    """Ask the installed binary which session providers it recognizes, now.

    Used only by `check_disabled_agents`'s completeness direction — the
    write path uses the pinned `OTHER_PROVIDERS` (see its docstring for why).

    To re-measure `OTHER_PROVIDERS` by hand after an AgentsView upgrade:

        AGENTSVIEW_DATA_DIR=<scratch> CLAUDE_PROJECTS_DIR=<scratch-empty-dir> \\
          agentsview doctor sync | sed -n '/^Agent roots:/,/^Recent/p' \\
          | sed -E 's/^\\s+([a-z0-9_-]+):.*/\\1/' | sort -u

    **Returns `None`, never an empty tuple, on anything not trustworthy.** A
    missing binary, non-zero exit, unparseable report, or a report with zero
    recognized names all return `None` — collapsing "found nothing" and
    "couldn't tell" into one signal, because an empty tuple would read as "no
    providers exist", which `check_disabled_agents` must not conclude from a
    probe failure. Never raises.
    """
    exe = prefix.bin / "agentsview"
    try:
        env = {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}
        proc = subprocess.run(  # noqa: S603
            [str(exe), "doctor", "sync"],
            env=env,
            capture_output=True,
            text=True,
            timeout=DISCOVER_PROVIDERS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout or ""
    start = text.find(_AGENT_ROOTS_HEADER)
    if start == -1:
        return None
    section = text[start + len(_AGENT_ROOTS_HEADER) :]
    names = {m.group(1) for m in _AGENT_ROOT_LINE_RE.finditer(section)}
    names.discard(_KEEP_ENABLED)
    return tuple(sorted(names)) if names else None


def write_config(prefix: Prefix, disabled_agents: Sequence[str]) -> None:
    """The prefix's `config.toml`. Idempotent, and ours alone.

    **A pure writer.** `disabled_agents` is the caller's decision — normally
    `OTHER_PROVIDERS`, the pinned list — kept as a parameter rather than read
    from a module constant so this function is testable against an arbitrary
    list without a fake binary in the loop.

    Written into `AGENTSVIEW_DATA_DIR`, never `~/.agentsview`: a user who
    already runs AgentsView keeps their own archive and settings untouched.

    **`daemon_idle_timeout = "0s"`, documented in AgentsView's own
    `configuration.md`, default `"20m"` otherwise.** Measured directly: the
    production daemon self-exited twice with no override present, matching
    the doc exactly. Design §4 promises the panel "persists across runs" and
    is only ever started from the CLI's own startup path — with the default
    20-minute idle exit, a user opening the URL an hour after their run ends
    finds nothing, which is precisely the case the panel exists for. Verified
    with a real background daemon left running with this key set and zero
    client traffic past the 20-minute default window; see PHASE0.md §0.7.
    """
    cfg = prefix.agentsview_data / "config.toml"
    disabled = ", ".join(json.dumps(name) for name in disabled_agents)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "# Written by agent_sys. AgentsView itself is unmodified.\n"
        f"disabled_agents = [{disabled}]\n"
        'host = "127.0.0.1"\n'
        "disable_update_check = true\n"
        'daemon_idle_timeout = "0s"\n'
    )


#: The exact substring AgentsView's config parser puts around the offending
#: name (measured directly: `disabled_agents: unknown session provider
#: "claude-cowork"`). A regex on the real message rather than a re-derivation
#: of the parser's own logic.
_UNKNOWN_PROVIDER_RE = re.compile(r'unknown session provider "([^"]+)"')

#: How long the validation probe may take. It runs against a config with
#: every provider disabled and `CLAUDE_PROJECTS_DIR` still whatever the
#: caller set, so a cold sync is at most the size of that one directory —
#: generous, not open-ended.
CHECK_DISABLED_AGENTS_TIMEOUT_S = 15.0


def check_disabled_agents(prefix: Prefix) -> tuple[str, ...]:
    """Has reality moved past the pinned `OTHER_PROVIDERS`? Checks both ways.

    **Direction 1 — rename or removal.** Runs `health`, the same cheap
    command `serve` would fail identically to (it loads config before doing
    anything else), against the exact `config.toml` `write_config` produced.
    If the binary's own parser rejects a name in it, that name comes back —
    `OTHER_PROVIDERS` lists something this installed version no longer
    recognizes, so the panel is not coming up until it is corrected.

    **Direction 2 — addition, the one that leaks.** Runs `discover_providers`
    (`doctor sync`'s own enumeration) and reports every name it finds that is
    in neither `OTHER_PROVIDERS` nor `claude`. This is the dangerous
    direction: a provider the binary gained but `OTHER_PROVIDERS` never
    mentions loads with **no error at all** — AgentsView just scans its
    default directory and puts its sessions on the panel. Only a genuine
    completeness check against the binary's own vocabulary can catch that; a
    validator that merely accepts our list proves nothing about what we
    forgot to list, which is why this direction exists as a separate probe
    rather than being inferred from direction 1's silence.

    **Every name found, from either direction, in one tuple** (empty if
    clean, or if a probe could not be trusted — a probe failure is not
    evidence of a problem in either direction: `suspend, don't conclude`).
    One warning at the call site regardless of how many names came back.
    Never raises.
    """
    problems: list[str] = []

    exe = prefix.bin / "agentsview"
    try:
        env = {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}
        proc = subprocess.run(  # noqa: S603
            [str(exe), "health", "--limit", "1"],
            env=env,
            capture_output=True,
            text=True,
            timeout=CHECK_DISABLED_AGENTS_TIMEOUT_S,
        )
        if proc.returncode != 0:
            m = _UNKNOWN_PROVIDER_RE.search(proc.stderr or "")
            if m:
                problems.append(m.group(1))
    except (OSError, subprocess.SubprocessError):
        pass

    discovered = discover_providers(prefix)
    if discovered is not None:
        problems.extend(sorted(set(discovered) - set(OTHER_PROVIDERS)))

    return tuple(problems)


def _owns_port(prefix: Prefix, port: int) -> bool:
    """Did *we* start what is on this port?

    **A live AgentsView on 18888 is not evidence that it is ours.** A user who
    already runs AgentsView has a daemon with their own `AGENTSVIEW_DATA_DIR`,
    listing every session on the machine — adopting it would hand back a panel
    that breaks the single requirement this component exists to satisfy. The
    port file is written only by our own successful launch, so it is the only
    evidence available that the daemon answering was configured by us.

    Never raises: an unreadable or malformed file is a "no", because the safe
    answer to *is this ours* is the one that declines to adopt a stranger.
    """
    try:
        return int((prefix.run / PORT_FILE).read_text().strip()) == port
    except (OSError, ValueError):
        return False


def _identifies_as_agentsview(url: str) -> bool:
    """One request. `200` **and** a JSON body, or it is not AgentsView.

    A status code is not an identity: any web server on the port answers 200,
    and returning `Status(True, …)` for one hands the operator a URL to a
    stranger's application labelled as their panel. `IDENTITY_PATH` is
    AgentsView's own endpoint, so a service that both answers it and returns
    JSON is as close to proof as a probe gets.
    """
    try:
        with urllib.request.urlopen(url + IDENTITY_PATH, timeout=2) as r:  # noqa: S310
            if r.status != 200:
                return False
            json.loads(r.read(IDENTITY_MAX_BYTES).decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return True


def _wait_for_health(url: str, timeout: float) -> bool:
    """Poll until AgentsView identifies itself, or the deadline passes.

    Always makes at least one attempt, so a zero timeout still asks once, and
    always sleeps between attempts — including after an answer that was *wrong*
    rather than absent. Without that second half, a stranger returning a prompt
    200 turns this into a busy loop hammering somebody else's service.
    """
    deadline = time.monotonic() + timeout
    while True:
        if _identifies_as_agentsview(url):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def ensure_running(prefix: Prefix, port: int) -> Status:
    """Start the panel, or say in one line why there is none.

    **Every return is a `Status` and every failure logs exactly one warning.**
    One, not two: a caller that sees the same problem reported twice starts
    hunting for two problems.
    """
    url = f"http://127.0.0.1:{port}"
    exe = prefix.bin / "agentsview"

    if not port_is_free(port):
        # **Two gates, and neither alone is enough.** Ownership is checked
        # first because it is a file read rather than a network round trip,
        # and because a `no` here means we must not probe further anyway.
        if _owns_port(prefix, port) and _wait_for_health(
            url, timeout=REUSE_PROBE_TIMEOUT_S
        ):
            return Status(True, "already running", url)
        log.warning(
            "agentsview: port %d is in use by something else; skipping the o11y "
            "panel. Pass --agentsview-port to choose another.",
            port,
        )
        return Status(False, f"port {port} in use")

    if not exe.is_file():
        log.warning(
            "agentsview: not installed at %s; skipping the o11y panel. "
            "Run `env-mgr install env_mgr/recipes/agentsview.o11y.yaml`.",
            exe,
        )
        return Status(False, "not installed")

    try:
        prefix.create()
        write_config(prefix, OTHER_PROVIDERS)
        env = {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}
        proc = subprocess.run(  # noqa: S603
            # `--replace`: measured directly (a stray daemon left over from an
            # earlier run, or any daemon already alive for this
            # AGENTSVIEW_DATA_DIR) that `serve --background --port N` without
            # this flag silently attaches to whatever is already running and
            # reports *its* port, exit 0, `N` completely ignored. By the time
            # this line runs the reuse gate above has already confirmed
            # `port_is_free(port)` is True, so anything still alive here is
            # necessarily *not* on the port we asked for -- there is no
            # legitimate case at this call site where replacing it is wrong.
            [str(exe), "serve", "--background", "--no-browser", "--replace",
             "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            capture_output=True,
            text=True,
            timeout=LAUNCH_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("agentsview: could not launch (%s); skipping the o11y panel.", e)
        return Status(False, f"launch failed: {e}")

    if proc.returncode != 0:
        log.warning(
            "agentsview: `serve --background` exited %d; skipping the o11y panel. stderr: %s",
            proc.returncode,
            (proc.stderr or "").strip()[:400],
        )
        return Status(False, f"exit {proc.returncode}")

    if not _wait_for_health(url, HEALTH_TIMEOUT_S):
        log.warning(
            "agentsview: started but did not answer %s within %.0fs; skipping the o11y panel.",
            url,
            HEALTH_TIMEOUT_S,
        )
        return Status(False, "health check timed out")

    (prefix.run / PORT_FILE).write_text(str(port))
    return Status(True, "started", url)


#: The recipe item `ensure_installed` drives. A fixed, in-package path rather
#: than a caller-supplied one: there is exactly one recipe for this component,
#: and a parameter here would just be a second way to point at the same file.
RECIPE_PATH = Path(__file__).resolve().parent.parent / "recipes" / "agentsview.o11y.yaml"


@contextlib.contextmanager
def _patched_environ(extra: Mapping[str, str]) -> Iterator[None]:
    """Patch `os.environ` with `extra` for exactly the duration of the block.

    **Why this exists at all.** The recipe's `check_cmd:` and `install:` both
    reference `$AGENT_SYS_HOME` (confirmed by reading `recipe.py::load_recipe`
    and `installers/base.py::run_cmd`: neither expands `${VAR}` itself; the
    shell that `run_cmd`'s `subprocess.run(cmd, shell=True)` invokes does, and
    only from whatever environment that call was made with — `run_cmd` takes
    no `env=` parameter). This context manager is the one place that
    environment is assembled, and it exists only for the width of one
    `runner.run(...)` call — never wider.

    **The cleaner long-term shape, deliberately not built now:** an `env=`
    parameter threaded through `run_cmd` → the installer protocol → `runner.run`.
    Not done here because `run_cmd` is shared machinery every installer in the
    registry uses, and widening it to serve this one caller is a bigger blast
    radius than a local, scoped patch justifies. If a second caller ever needs
    the same thing, that is the point to revisit this.

    **Restored in a `finally`, on every exit path including an exception.**
    This module's whole reason to exist is "never touches what it did not
    mean to" (`agent_environment`'s guard on `os.environ` is the same
    discipline); a half-restored environment on the exception path would be
    the worst possible way to break that promise.
    """
    saved = dict(os.environ)
    try:
        os.environ.update(extra)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def ensure_installed(prefix: Prefix, install_item: Callable[[], Sequence[Any]]) -> Status:
    """Install the `agentsview` binary via its recipe item, or say why not.

    **`install_item` is injected rather than looked up here.** `env_mgr` spec
    §9 draws a decoupling wall — nothing new may import the installer
    machinery (`recipe`, `runner`, `installers/…`), checked structurally by
    `tests/env_mgr/test_imports.py::test_nothing_new_imports_the_installer_machinery`.
    A first draft of this function called `recipe.load_recipe` and
    `runner.run` directly and failed exactly that test. `install_item` is a
    zero-argument callable returning the list of `Outcome`s from running the
    `agentsview` recipe item's `install` stage (duck-typed on `.level` /
    `.message` below — no `Outcome` import needed even for typing, since
    `outcome` is also below the wall); the caller assembles it, typically by
    loading `RECIPE_PATH`, overriding `target.path` to `str(prefix.root)`
    (the recipe's own `target.path` is a placeholder — see the recipe file's
    header comment), and calling `runner.run(target, items, "install",
    Filters(item="agentsview"))`. That caller lives outside this wall, e.g.
    `env_mgr/cli.py` (the one module spec §9 exempts) or `agent_sys/cli/main.py`
    (a different package entirely, not subject to this wall at all).

    **Never raises.** Everything from "the injected callable raised" to "the
    download failed inside it" collapses to one `Status(False, ...)` and one
    `log.warning`, the same law `ensure_running` already holds.
    `importance: suggested` in the recipe is not re-derived here — the
    outcome's own `.level` (computed by `installers/base.level_for_missing`
    from that one field) is read directly, so there is exactly one place that
    decides "is a missing agentsview fatal", and it is the recipe.
    """
    try:
        prefix.create()
        with _patched_environ(prefix.environment()):
            outs = list(install_item())
    except Exception as e:  # noqa: BLE001 - see ensure_running's docstring
        log.warning("agentsview: install failed (%s); skipping the o11y panel.", e)
        return Status(False, f"install error: {e}")

    if not outs:
        log.warning(
            "agentsview: recipe item 'agentsview' not found in %s; skipping the o11y panel.",
            RECIPE_PATH,
        )
        return Status(False, "recipe item not found")

    outcome = outs[-1]
    if outcome.level == "ok":
        # **Validated here, at install time, not left for `serve` to discover
        # later.** `OTHER_PROVIDERS` is pinned, hand-maintained, and can drift
        # from the installed binary in two directions: a name it lists that
        # the binary no longer recognizes (breaks `serve` loudly, the way it
        # broke here once already), or a name the binary recognizes that the
        # list never mentions (leaks that provider's sessions onto the panel,
        # silently — `check_disabled_agents` is what makes that visible).
        #
        # In its own `try`, separate from `install_item()` above: the binary
        # is genuinely installed at this point regardless of what this check
        # finds, so a bug in the check itself must not turn a successful
        # install into a reported failure — "never raises" is this module's
        # one law, and it applies to every line, not just the obvious ones.
        try:
            write_config(prefix, OTHER_PROVIDERS)
            bad = check_disabled_agents(prefix)
        except Exception as e:  # noqa: BLE001 - see ensure_running's docstring
            log.warning(
                "agentsview: could not validate OTHER_PROVIDERS against the "
                "installed binary (%s); continuing without that check.",
                e,
            )
            bad = ()
        if bad:
            log.warning(
                "agentsview: OTHER_PROVIDERS in env_mgr/o11y/agentsview.py has "
                "drifted from the installed agentsview: %s. A name here the "
                "binary no longer recognizes will keep the panel from "
                "starting; a name the binary recognizes but this list omits "
                "means that provider's sessions may appear on the panel.",
                ", ".join(bad),
            )
        return Status(True, outcome.message)

    log.warning("agentsview: %s; skipping the o11y panel.", outcome.message)
    return Status(False, outcome.message)
