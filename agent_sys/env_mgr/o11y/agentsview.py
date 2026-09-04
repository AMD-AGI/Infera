# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AgentsView, started as a side-car and never allowed to fail a run.

**The one rule this module exists to enforce:** an observability panel that can
break the thing it observes is worse than no panel. Every function here returns
a `Status` and raises nothing. There is a test per failure mode holding that
line, because the failure mode of a warning-only component is that someone
later "improves" it into a raise.

**Why we decide the port instead of letting AgentsView decide.** `agentsview
serve` auto-discovers a free port when the requested one is busy — sensible for
a human at a terminal, wrong here: the mission asks for a warning and a skip on
a taken port, and a daemon that quietly moved to 18889 is a panel nobody knows
the address of. So the bind probe happens here, before launch.

Rationale, measurements and rejected alternatives: `../docs/design.md` §17.1
and `README.md`.
"""

from __future__ import annotations

import contextlib
import http.client
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
    from ..prefix import Prefix

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


#: A port we may ask for. **`0` is excluded deliberately**: it binds, then means
#: "any free port", handing the choice to AgentsView's auto-discovery.
_LOWEST_PORT = 1
_HIGHEST_PORT = 65535


def _in_range(port: int) -> bool:
    return _LOWEST_PORT <= port <= _HIGHEST_PORT


def resolve_port(flag: int | None, environ: Mapping[str, str]) -> int:
    """Flag, then environment, then 18888.

    **An unusable value is the default, not an error** — refusing a deployment
    over a typo in a variable nobody needed inverts this module's priority.
    Range matters as well as parseability: `bind` answers an out-of-range port
    with `OverflowError`, which is not `OSError` and nothing downstream catches.
    """
    if flag is not None:
        if _in_range(flag):
            return int(flag)
        log.warning(
            "agentsview: --agentsview-port %d is not in %d-%d; using the default %d",
            flag,
            _LOWEST_PORT,
            _HIGHEST_PORT,
            DEFAULT_PORT,
        )
        return DEFAULT_PORT
    raw = environ.get(PORT_ENV_VAR)
    if raw is None:
        return DEFAULT_PORT
    try:
        parsed = int(raw)
    except ValueError:
        parsed = None
    if parsed is not None and _in_range(parsed):
        return parsed
    log.warning(
        "%s=%r is not a usable port number; using the default %d",
        PORT_ENV_VAR,
        raw,
        DEFAULT_PORT,
    )
    return DEFAULT_PORT


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """A real bind, not a connect.

    Connecting answers "is someone accepting", a different question: a socket
    bound and not listening still makes our own bind fail. `OverflowError`
    alongside `OSError` because that is what `bind` raises out of range.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
    except (OSError, OverflowError):
        return False
    return True


#: How long the launch subprocess itself may take. `serve --background`
#: daemonises and returns at once, so anything slower is a hung binary.
LAUNCH_TIMEOUT_S = 20.0

#: How long we then wait for the daemon to answer. Cold-start reads the whole
#: session archive, so this is generous.
HEALTH_TIMEOUT_S = 30.0

#: How long we spend deciding whether something on the port is *our* panel.
#: Short deliberately: this is on every deployment's path, and the answer either
#: arrives at once or the thing there would not have served us anyway. Named so
#: a test can shrink it.
REUSE_PROBE_TIMEOUT_S = 2.0

#: AgentsView's own JSON endpoint (`docs/session-api.md:112`), used as the
#: identity probe. Not `/`: every web server answers that with a 200.
IDENTITY_PATH = "/api/v1/agents"

#: Cap on the identity response, so a stranger streaming without end cannot
#: hang a deployment on the o11y probe.
IDENTITY_MAX_BYTES = 1 << 20

#: One identity request's deadline, as distinct from `REUSE_PROBE_TIMEOUT_S`
#: which bounds the series of them. Equal by coincidence, unrelated.
IDENTITY_PROBE_TIMEOUT_S = 2.0

#: **AgentsView's own artefact, not ours.** `serve` writes one per running
#: daemon into its `AGENTSVIEW_DATA_DIR` and removes it on a clean stop
#: (measured, v0.42.0). That directory is the prefix's and nobody else's, so a
#: record here was written by a daemon we configured. Read only.
DAEMON_RECORD_GLOB = "daemon.*.json"

#: Checked so an unrelated file matching the glob is not read as a daemon.
SERVICE_NAME = "agentsview"


def _binary_env(prefix: Prefix) -> dict[str, str]:
    """The environment every `agentsview` subprocess gets. One definition.

    **`HOME` is gate 5**: AgentsView derives every provider's default root from
    it, so this scopes providers we have never heard of, with no list to go
    stale. A replacement, not an overlay — unverified whether any AgentsView
    path wants a `TMPDIR` or `LANG` this drops.
    """
    return {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}


#: **The pinned, reviewable statement of intent** — every provider AgentsView
#: can scan, minus Claude Code. Pinned rather than derived at runtime so an
#: upstream change alters the panel through a diff and a review, not silently.
#: Measured against `v0.42.0 doctor sync`'s "Agent roots:"; `check_disabled_agents`
#: re-runs that at install time and warns on drift in either direction.
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

#: The one `OTHER_PROVIDERS` must never contain: gate 3 disables everything
#: except the source `agent_sys` itself writes.
_KEEP_ENABLED = "claude"

#: `doctor sync`'s report lists every provider the binary recognizes under an
#: "Agent roots:" header, one two-space-indented `name: path (status)` line per
#: root. Measured against a real v0.42.0, not re-derived from the docs' table.
_AGENT_ROOTS_HEADER = "Agent roots:"
_AGENT_ROOT_LINE_RE = re.compile(r"^  ([a-z0-9_-]+):", re.MULTILINE)

#: The end of that section: the first line that is not indented.
_AGENT_ROOTS_END_RE = re.compile(r"^(?=\S)", re.MULTILINE)

#: `doctor sync` only stats candidate directories and reads sync metadata.
DISCOVER_PROVIDERS_TIMEOUT_S = 10.0


def _parse_agent_roots(stdout: str) -> tuple[str, ...] | None:
    """The "Agent roots:" section of a `doctor sync` report -> sorted names.

    One parser for both consumers, so they cannot drift apart. `None` rather
    than `()` when the section is missing or names none — see
    `discover_providers` for why the caller needs that distinction.
    """
    start = stdout.find(_AGENT_ROOTS_HEADER)
    if start == -1:
        return None
    # Bounded at the next unindented line: scanning past it would read any
    # later `  something:` as a phantom provider and warn about drift.
    section = _AGENT_ROOTS_END_RE.split(
        stdout[start + len(_AGENT_ROOTS_HEADER) :], maxsplit=1
    )[0]
    names = {m.group(1) for m in _AGENT_ROOT_LINE_RE.finditer(section)}
    names.discard(_KEEP_ENABLED)
    return tuple(sorted(names)) if names else None


def discover_providers(prefix: Prefix) -> tuple[str, ...] | None:
    """Ask the installed binary which session providers it recognizes, now.

    **`doctor sync`, never `health`** — measured: `health`, `projects` and
    `session list` each autostart a daemon on a port AgentsView picks.
    **`None`, never `()`, on anything untrustworthy**: an empty tuple would read
    as "no providers exist", which a probe failure must not assert. To
    re-measure `OTHER_PROVIDERS` after an upgrade:

        AGENTSVIEW_DATA_DIR=<scratch> CLAUDE_PROJECTS_DIR=<empty> \\
          agentsview doctor sync | sed -n '/^Agent roots:/,/^Recent/p' \\
          | sed -E 's/^\\s+([a-z0-9_-]+):.*/\\1/' | sort -u
    """
    exe = prefix.bin / "agentsview"
    try:
        proc = subprocess.run(  # noqa: S603
            [str(exe), "doctor", "sync"],
            env=_binary_env(prefix),
            capture_output=True,
            text=True,
            timeout=DISCOVER_PROVIDERS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return _parse_agent_roots(proc.stdout or "")


def write_config(prefix: Prefix, disabled_agents: Sequence[str]) -> None:
    """The prefix's `config.toml`. Idempotent, and ours alone.

    Written into `AGENTSVIEW_DATA_DIR`, never `~/.agentsview`, so a user who
    already runs AgentsView keeps their archive. `daemon_idle_timeout = "0s"`
    because the default 20m empties the panel for anyone opening the URL after
    their run — measured, the daemon self-exited twice without it.
    """
    cfg = prefix.agentsview_data / "config.toml"
    disabled = ", ".join(json.dumps(name) for name in disabled_agents)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    # Written whole, by rename: a reader catching a truncated `write_text` gets
    # a short `disabled_agents`, i.e. every other provider silently re-enabled.
    tmp = cfg.with_name(cfg.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        "# Written by agent_sys. AgentsView itself is unmodified.\n"
        f"disabled_agents = [{disabled}]\n"
        'host = "127.0.0.1"\n'
        "disable_update_check = true\n"
        'daemon_idle_timeout = "0s"\n'
    )
    os.replace(tmp, cfg)


#: The substring AgentsView's config parser puts around the offending name
#: (measured: `disabled_agents: unknown session provider "claude-cowork"`).
_UNKNOWN_PROVIDER_RE = re.compile(r'unknown session provider "([^"]+)"')

#: A cold sync of one `CLAUDE_PROJECTS_DIR`. Generous, not open-ended.
CHECK_DISABLED_AGENTS_TIMEOUT_S = 15.0


def check_disabled_agents(prefix: Prefix) -> tuple[str, ...]:
    """Has reality moved past the pinned `OTHER_PROVIDERS`? Checks both ways.

    **Rename or removal:** `doctor sync` exits non-zero naming the entry this
    version dropped, and the panel will not start. **Addition, the one that
    leaks:** its "Agent roots:" report is diffed against `OTHER_PROVIDERS`,
    because a provider we forgot to disable loads with no error at all. Only
    one can surface per call — a failing sync never prints the report. Empty
    means clean *or* untrusted; a probe failure is evidence of neither.
    """
    exe = prefix.bin / "agentsview"
    try:
        proc = subprocess.run(  # noqa: S603
            [str(exe), "doctor", "sync"],
            env=_binary_env(prefix),
            capture_output=True,
            text=True,
            timeout=CHECK_DISABLED_AGENTS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    if proc.returncode != 0:
        m = _UNKNOWN_PROVIDER_RE.search(proc.stderr or "")
        return (m.group(1),) if m else ()

    discovered = _parse_agent_roots(proc.stdout or "")
    if discovered is None:
        return ()
    return tuple(sorted(set(discovered) - set(OTHER_PROVIDERS)))


def _pid_is_alive(pid: int) -> bool:
    """`ESRCH` is a no; `EPERM` is a yes — a process we may not signal exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _owns_port(prefix: Prefix, port: int) -> bool:
    """Did *we* start what is on this port, and is it still alive?

    **A live AgentsView here is not evidence that it is ours**: a user's own
    lists every session on the machine. The witness is AgentsView's
    `daemon.<pid>.json` in *our* data directory — a stranger's goes in theirs,
    so the isolation is the filesystem's. Removed on a clean stop, so only an
    unclean death leaves a stale one and the pid catches that. Anything
    missing, unreadable or foreign is a "no".
    """
    try:
        records = sorted(prefix.agentsview_data.glob(DAEMON_RECORD_GLOB))
    except OSError:
        return False
    for path in records:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("service") != SERVICE_NAME:
            continue
        if not _record_names_port(record, port):
            continue
        pid = record.get("pid")
        if isinstance(pid, int) and _pid_is_alive(pid):
            return True
    return False


def _record_names_port(record: Mapping[str, Any], port: int) -> bool:
    """`metadata.port` first, `address` as the fallback.

    Both are in a real v0.42.0 record. Two readings because this is an external
    artefact: dropping either field leaves the gate working, dropping both
    fails it closed.
    """
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping) and str(metadata.get("port", "")) == str(port):
        return True
    address = record.get("address")
    return isinstance(address, str) and address.rsplit(":", 1)[-1] == str(port)


def _identifies_as_agentsview(url: str) -> bool:
    """One request. `200` **and** a JSON body, or it is not AgentsView.

    A status code is not an identity: any web server answers 200, and returning
    `Status(True, …)` for one hands the operator a stranger's application
    labelled as their panel.
    """
    try:
        with urllib.request.urlopen(  # noqa: S310
            url + IDENTITY_PATH, timeout=IDENTITY_PROBE_TIMEOUT_S
        ) as r:
            if r.status != 200:
                return False
            json.loads(r.read(IDENTITY_MAX_BYTES).decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
        # `HTTPException` because `IncompleteRead` — raised on a truncated
        # chunked body, the framing a Go server uses with no Content-Length —
        # descends from neither `OSError` nor `ValueError`.
        return False
    return True


def _wait_for_health(url: str, timeout: float) -> bool:
    """Poll until AgentsView identifies itself, or the deadline passes.

    Always tries once, so a zero timeout still asks, and always sleeps between
    attempts *including after a wrong answer* — otherwise a stranger returning
    a prompt 200 turns this into a busy loop against someone else's service.
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
    One, not two: a caller shown the same problem twice hunts for two problems.
    """
    url = f"http://127.0.0.1:{port}"
    exe = prefix.bin / "agentsview"

    if not port_is_free(port):
        # Two gates, neither enough alone. Ownership first: it is a file read
        # rather than a round trip, and a `no` means we must not probe further.
        # The two failures are separate warnings because their fixes are
        # opposite — "something else has your port" sends an operator hunting
        # for a process that does not exist when ours is merely wedged.
        if _owns_port(prefix, port):
            if _wait_for_health(url, timeout=REUSE_PROBE_TIMEOUT_S):
                return Status(True, "already running", url)
            log.warning(
                "agentsview: our own daemon holds port %d but did not answer %s "
                "within %.1fs; skipping the o11y panel. Stop it with "
                "`AGENTSVIEW_DATA_DIR=%s agentsview serve stop`.",
                port,
                url,
                REUSE_PROBE_TIMEOUT_S,
                prefix.agentsview_data,
            )
            return Status(False, f"our daemon on port {port} is not answering")
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

    # `--replace`: measured, without it `serve --background --port N` silently
    # attaches to any daemon already alive for this data directory and reports
    # *its* port, exit 0, `N` ignored. `port_is_free(port)` was true above, so
    # anything still alive is necessarily not on the port we asked for and
    # there is no legitimate case here where replacing it is wrong.
    try:
        prefix.create()
        write_config(prefix, OTHER_PROVIDERS)
        proc = subprocess.run(  # noqa: S603
            [str(exe), "serve", "--background", "--no-browser", "--replace",
             "--host", "127.0.0.1", "--port", str(port)],
            env=_binary_env(prefix),
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

    # Nothing is recorded here: AgentsView wrote its own `daemon.<pid>.json`
    # when it started and that is what `_owns_port` reads. A second record of
    # ours would only be a thing that can disagree.
    return Status(True, "started", url)


#: The recipe item `ensure_installed` drives. Fixed and in-package: there is one
#: recipe for this component, and a parameter would be a second way to say so.
RECIPE_PATH = Path(__file__).resolve().parent.parent / "recipes" / "agentsview.o11y.yaml"


@contextlib.contextmanager
def _patched_environ(extra: Mapping[str, str]) -> Iterator[None]:
    """Patch `os.environ` for exactly the duration of the block.

    The recipe references `$AGENT_SYS_HOME` and `installers/base.run_cmd` takes
    no `env=`, so the shell expands from the ambient environment or not at all.
    **This does mutate the process environment**, unlike the rest of the
    feature — but not `CLAUDE_CONFIG_DIR`, which `Prefix.environment()` does not
    carry, so the promise about the user's Claude Code holds. Not thread-safe;
    the CLI path is single-threaded.
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

    **`install_item` is injected, not looked up**: spec §9 walls the installer
    machinery off from everything under `env_mgr/`, so `cli/main.py` assembles
    the call. **Never raises** — one `Status` and one warning. Whether a missing
    agentsview is fatal is decided once, by the recipe's `importance:`.
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
        # Validated at install time rather than left for `serve` to discover.
        # In its own `try`: the binary is installed either way, so a bug in the
        # check must not turn a successful install into a reported failure.
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
