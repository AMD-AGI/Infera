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


#: A port we may actually ask for. **`0` is excluded deliberately**: it binds
#: successfully and then means "any free port", handing the choice to
#: AgentsView's own auto-discovery — the delegation this module's header says
#: it exists to prevent.
_LOWEST_PORT = 1
_HIGHEST_PORT = 65535


def _in_range(port: int) -> bool:
    return _LOWEST_PORT <= port <= _HIGHEST_PORT


def resolve_port(flag: int | None, environ: Mapping[str, str]) -> int:
    """Flag, then environment, then 18888.

    **An unusable value is the default, not an error.** This is a side-car;
    refusing to start the whole deployment over a typo in a variable nobody
    needed would invert the priority the module is built on. Unusable covers
    both halves — a value that is not a number, and a number that is not a
    port. The second is not pedantry: `socket.bind` answers an out-of-range
    port with `OverflowError`, which is not `OSError` and so is not caught by
    anything downstream except the CLI's blanket backstop, which is meant to
    be a fuse rather than the mechanism.
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

    Connecting answers "is someone accepting", which is a different question:
    a socket bound and not listening still makes our own bind fail. We ask the
    question we actually need answered.

    `OverflowError` alongside `OSError` because that is what `bind` raises for
    a port outside 0-65535, and it descends from neither. `resolve_port` keeps
    such a value from reaching here at all; this is the second line, for a
    caller that passes a port directly.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
    except (OSError, OverflowError):
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

#: One identity request's own deadline, as distinct from `REUSE_PROBE_TIMEOUT_S`
#: which bounds the *series* of them. They happen to be equal and are unrelated;
#: the equality is why this was a bare `timeout=2` for a while, in a module that
#: names every other timeout.
IDENTITY_PROBE_TIMEOUT_S = 2.0

#: **AgentsView's own artefact, not ours.** `serve` writes one
#: `daemon.<pid>.json` into its `AGENTSVIEW_DATA_DIR` while it is running and
#: removes it on a clean stop (measured, v0.42.0). Because that directory is
#: the prefix's and nobody else's, a record found here was written by a daemon
#: we configured — which is what separates "reuse the panel we started" from
#: "adopt whatever is listening". Read only; never written by us.
DAEMON_RECORD_GLOB = "daemon.*.json"

#: The `service` field of that record, checked so an unrelated file matching
#: the glob is not mistaken for a daemon.
SERVICE_NAME = "agentsview"


def _binary_env(prefix: Prefix) -> dict[str, str]:
    """The environment every `agentsview` subprocess gets. One definition.

    **`HOME` is gate 5 and it is the reason this is a function.** AgentsView
    computes every provider's *default* session root from `HOME`, so pointing
    it at the prefix means every root it could scan resolves inside the prefix
    — regardless of whether that provider is named in `OTHER_PROVIDERS`, or
    even exists yet. Unlike the denylist, this gate cannot go stale when
    upstream adds a provider we have never heard of, because it needs no list
    at all. It was written out at three call sites, with a test holding the
    line at each; a test compensating for a missing helper is a helper that
    should exist.

    **A replacement, not an overlay.** Nothing of the caller's environment
    reaches the binary — no `TMPDIR`, no `LANG`, no proxy variables. That is
    deliberate (an inherited `CLAUDE_PROJECTS_DIR` or `AGENTSVIEW_DATA_DIR`
    would undo the scoping) and it is what production has run on throughout;
    it is *unverified* whether any AgentsView code path wants one of the
    variables this drops.
    """
    return {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}

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

#: The end of that section: the first line that is neither indented nor blank.
_AGENT_ROOTS_END_RE = re.compile(r"^(?=\S)", re.MULTILINE)

#: How long `doctor sync` may take. It only stats candidate directories and
#: reads existing sync-state metadata — no full session parse — so this is
#: short relative to the other probes in this module.
DISCOVER_PROVIDERS_TIMEOUT_S = 10.0


def _parse_agent_roots(stdout: str) -> tuple[str, ...] | None:
    """The "Agent roots:" section of a `doctor sync` report -> sorted names.

    Shared by `discover_providers` and `check_disabled_agents` so there is
    exactly one parser for this report, not two that could drift apart.
    `None`, never an empty tuple, when the section is missing or names to
    zero — see `discover_providers`'s docstring for why that distinction
    matters to the caller.
    """
    start = stdout.find(_AGENT_ROOTS_HEADER)
    if start == -1:
        return None
    # **Bounded at the next unindented line.** The section's own lines are all
    # two-space indented, so the first line that is not is the next section —
    # and scanning past it would turn any later `  something:` line into a
    # phantom provider and a spurious drift warning. `discover_providers`'s
    # docstring shows the hand-run command bounded the same way, with `sed`.
    section = _AGENT_ROOTS_END_RE.split(
        stdout[start + len(_AGENT_ROOTS_HEADER) :], maxsplit=1
    )[0]
    names = {m.group(1) for m in _AGENT_ROOT_LINE_RE.finditer(section)}
    names.discard(_KEEP_ENABLED)
    return tuple(sorted(names)) if names else None


def discover_providers(prefix: Prefix) -> tuple[str, ...] | None:
    """Ask the installed binary which session providers it recognizes, now.

    Used only by `check_disabled_agents`'s completeness direction — the
    write path uses the pinned `OTHER_PROVIDERS` (see its docstring for why).

    **`doctor sync`, never `health` or any other command that reads session
    data.** Measured directly (`zonelink`'s isolated probe, confirmed here):
    `health`, `projects`, and `session list` all silently autostart a
    background `serve` daemon on a port *AgentsView* picks by
    auto-discovery, exactly the "port decision made by agent_sys, never
    delegated" rule this component exists to hold. `doctor sync` does
    not — confirmed by running it from a cold prefix and counting real
    `agentsview serve` processes by exact argv before and after: zero, both
    times. It only stats candidate directories and reads config; it never
    opens a daemon-owned connection.

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
        env = _binary_env(prefix)
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
    return _parse_agent_roots(proc.stdout or "")


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
    client traffic past the 20-minute default window; see PHASE0.md §0.9.
    """
    cfg = prefix.agentsview_data / "config.toml"
    disabled = ", ".join(json.dumps(name) for name in disabled_agents)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    # **Written whole, by rename.** `write_text` truncates and then writes, and
    # the reader is a separate process: two concurrent deployments — the case
    # `ensure_running` reasons about explicitly — can have one mid-truncate
    # while the other's `doctor sync` or `serve` reads. A partial read of this
    # particular file is not a crash, it is `disabled_agents` coming back
    # short, which is every other provider silently re-enabled on the panel.
    # `os.replace` is atomic within a directory, so a reader sees the old file
    # or the new one and never a half of either.
    tmp = cfg.with_name(cfg.name + f".{os.getpid()}.tmp")
    tmp.write_text(
        "# Written by agent_sys. AgentsView itself is unmodified.\n"
        f"disabled_agents = [{disabled}]\n"
        'host = "127.0.0.1"\n'
        "disable_update_check = true\n"
        'daemon_idle_timeout = "0s"\n'
    )
    os.replace(tmp, cfg)


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

    **One `doctor sync` call answers both directions; `health` is not used
    at all.** A first version ran `health --limit 1` against the config
    `write_config` produced, on the reasoning that it is "the same cheap
    command `serve` would fail identically to". Measured (`zonelink`'s
    isolated probe, confirmed here by counting real `agentsview serve`
    processes by exact argv before/after): `health` silently autostarts a
    background daemon on a port *AgentsView* auto-discovers, exactly the
    "port decision made by agent_sys, never delegated to agentsview" rule
    this whole component exists to hold — happening on a read-only
    validation path neither `ensure_running` nor anyone watching for it
    would notice. `doctor sync` needs neither: measured to never touch the
    database or start a daemon, in either the config-valid or
    config-invalid case, and it validates `disabled_agents` with the
    *identical* "unknown session provider" error `health` produces, so
    nothing is lost by dropping `health` entirely.

    **`AGENTSVIEW_NO_DAEMON=1` was tried here and rejected — do not reach
    for it.** It is the obvious repair for the paragraph above (AgentsView
    documents it as exactly the "don't autostart" knob) and its absence
    from this code says nothing about whether it works, so: it was
    measured, and it does not make these commands read the database
    directly instead of autostarting — it makes them refuse outright.
    With it set, `health`, `projects` and `session list` all exit with
    `daemon autostart is disabled; direct SQLite reads are not supported
    for this command`, **even against an already-populated `sessions.db`**
    (verified by creating the database through one clean daemon lifecycle
    first, then re-testing). There is no fallback path. Adopting it would
    therefore have left a `health` probe that fails on every single call —
    and since this function treats a probe it cannot trust as `()` (see
    the last paragraph), that permanent failure would have collapsed into
    a permanently empty result: a provider check that never warns about
    anything, never looks broken, and is a no-op forever. Worse than the
    stray daemon it was meant to fix. Measured in `PHASE0.md` §0.10.

    **Direction 1 — rename or removal.** `doctor sync` exits non-zero and
    names the offending entry when `config.toml`'s `disabled_agents`
    contains something this installed version no longer recognizes; the
    panel is not coming up until it is corrected.

    **Direction 2 — addition, the one that leaks.** When `doctor sync`
    succeeds, its own "Agent roots:" enumeration (`_parse_agent_roots`, the
    same parser `discover_providers` uses) is diffed against
    `OTHER_PROVIDERS`; anything present in neither `OTHER_PROVIDERS` nor
    `claude` is a provider the binary recognizes that we are not disabling.
    This is the dangerous direction: it loads with **no error at all** —
    AgentsView just scans that provider's default directory and puts its
    sessions on the panel. Only a genuine completeness check against the
    binary's own vocabulary catches that; a validator that merely accepts
    our list proves nothing about what we forgot to list.

    **Only one direction can surface per call, and that is the price of
    dropping `health`.** A `doctor sync` that fails (direction 1) never
    reaches the point of printing "Agent roots:", so a config with both a
    stale entry *and* a missing one only reports the stale one here; fixing
    it and re-running finds the missing one next, the same iterative
    fix-and-rerun shape a human debugging this by hand would follow. The
    old two-probe design could report both problems from one call — trading
    that for never starting a daemon on a validation path.

    Empty result means either direction was clean, or the probe could not
    be trusted (a probe failure is not evidence of a problem in either
    direction: `suspend, don't conclude`). One warning at the call site
    regardless of how many names came back. Never raises.
    """
    exe = prefix.bin / "agentsview"
    try:
        env = _binary_env(prefix)
        proc = subprocess.run(  # noqa: S603
            [str(exe), "doctor", "sync"],
            env=env,
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
    """`ESRCH` is a no; `EPERM` is a yes.

    A process we may not signal still exists, and one living in our own data
    directory is ours whether or not the current uid can touch it.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _owns_port(prefix: Prefix, port: int) -> bool:
    """Did *we* start what is on this port, and is it still alive?

    **A live AgentsView on 18888 is not evidence that it is ours.** A user who
    already runs AgentsView has a daemon with their own `AGENTSVIEW_DATA_DIR`,
    listing every session on the machine — adopting it would hand back a panel
    that breaks the single requirement this component exists to satisfy.

    **The witness is AgentsView's own `daemon.<pid>.json`, read out of our own
    `AGENTSVIEW_DATA_DIR`.** A stranger's daemon writes its record into
    *their* data directory, so it structurally cannot appear here — the
    isolation is the filesystem's, not a convention we maintain. The record is
    the dependency's published artefact, read and never written; AgentsView
    stays unmodified.

    This replaced a file of our own holding just the port number, which
    recorded that we *once* started a daemon here and never expired. That is
    evidence about the past: after our daemon died and the user started their
    own AgentsView on the same port, both gates passed and the operator was
    handed a URL to a panel listing their whole machine, with no warning.
    Two things fix it, and both come from the record. It is removed by
    AgentsView itself on a clean `serve stop` — measured on a real v0.42.0,
    file gone and pid reaped — so the ordinary case leaves no stale evidence
    at all; and for the unclean case (SIGKILL, OOM, reboot) the recorded pid
    is checked for liveness, which the bare port number could not be.

    It also retires the *other* half of that bug. The port file was written
    only after the health check passed, so one slow cold start left a live
    daemon nobody could recognize and the panel was skipped on every run
    thereafter, permanently, blaming a stranger. AgentsView writes this record
    when it starts, so a daemon that is up is recognized as ours whether or
    not we were still waiting when it finished booting.

    Never raises: a missing, unreadable, malformed or foreign-shaped record is
    a "no", because the safe answer to *is this ours* is the one that declines
    to adopt a stranger.
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

    Both are present in a real v0.42.0 record (`"port": "18888"` and
    `"address": "127.0.0.1:18888"`). Two readings rather than one because this
    is an external artefact: a version that drops either field still leaves the
    gate working, and a version that drops both fails closed.
    """
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping) and str(metadata.get("port", "")) == str(port):
        return True
    address = record.get("address")
    return isinstance(address, str) and address.rsplit(":", 1)[-1] == str(port)


def _identifies_as_agentsview(url: str) -> bool:
    """One request. `200` **and** a JSON body, or it is not AgentsView.

    A status code is not an identity: any web server on the port answers 200,
    and returning `Status(True, …)` for one hands the operator a URL to a
    stranger's application labelled as their panel. `IDENTITY_PATH` is
    AgentsView's own endpoint, so a service that both answers it and returns
    JSON is as close to proof as a probe gets.
    """
    try:
        with urllib.request.urlopen(  # noqa: S310
            url + IDENTITY_PATH, timeout=IDENTITY_PROBE_TIMEOUT_S
        ) as r:
            if r.status != 200:
                return False
            json.loads(r.read(IDENTITY_MAX_BYTES).decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
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
        #
        # The two failures are reported separately because their fixes are
        # opposites: "something else has your port" sends an operator hunting
        # for a process that does not exist when the truth is that our own
        # daemon is wedged and wants killing.
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

    # `--replace` on the `serve --background` call below: measured directly
    # (a stray daemon left over from an earlier run, or any daemon already
    # alive for this AGENTSVIEW_DATA_DIR) that `serve --background --port N`
    # without this flag silently attaches to whatever is already running and
    # reports *its* port, exit 0, `N` completely ignored. By the time that
    # line runs the reuse gate above has already confirmed `port_is_free(port)`
    # is True, so anything still alive there is necessarily *not* on the port
    # we asked for -- there is no legitimate case at this call site where
    # replacing it is wrong.
    #
    # **Two concurrent `agent_sys` deployments on this box, asked in review,
    # and measured rather than assumed.** Both resolve the same prefix
    # (`AGENTSVIEW_DATA_DIR` is not per-run), so this is the one scenario
    # where `--replace` could plausibly cost something. Measured directly
    # (scratch/replace_concurrency): `--replace` against an *already-settled*
    # daemon on the exact same port still unconditionally kills and restarts
    # it -- it is not a same-port no-op, so this is a real, not hypothetical,
    # question. But by the time either deployment reaches the `serve
    # --replace` call, `port_is_free(port)` must have been True for it -- so
    # the only window where *two* deployments can both reach it for the same
    # port is the narrow race where neither daemon is up yet (a cold start,
    # or the moment after the idle-timeout self-exit this same fix
    # addresses). A synchronized two-thread race against a real binary
    # landed only one actual daemon start; the loser's own `serve --replace`
    # invocation saw the winner's daemon and did not visibly disrupt it in
    # that run -- but this is empirical, not a documented guarantee, and
    # should not be read as "the race is safe by design". What *is* true
    # regardless of who wins: both deployments point at the identical
    # prefix, config, and port, so a replacement here swaps one daemon for a
    # functionally identical one serving the same archive -- at worst a
    # sub-second connection drop for anyone with the URL open mid-restart
    # (~650-700ms measured start time), never a lasting outage or a panel
    # showing different data.
    try:
        prefix.create()
        write_config(prefix, OTHER_PROVIDERS)
        env = _binary_env(prefix)
        proc = subprocess.run(  # noqa: S603
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

    # Nothing is recorded here. AgentsView wrote its own `daemon.<pid>.json`
    # into our data directory when it started, and that is what `_owns_port`
    # reads — one witness, written by the process it describes, removed by it
    # on a clean stop. A second record of ours would only be a thing that can
    # disagree, and the last line of this function is a poor place to discover
    # that `run/` has become unwritable.
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
