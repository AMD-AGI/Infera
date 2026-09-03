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

import json
import logging
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .prefix import Prefix

__all__ = [
    "DEFAULT_PORT",
    "Status",
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

#: Every provider AgentsView can scan, minus Claude Code. Written into the
#: prefix's own `config.toml` so the panel physically cannot read a directory
#: belonging to some other tool the user happens to have installed.
OTHER_PROVIDERS = (
    "aider", "amp", "antigravity", "antigravity-cli", "claude-cowork",
    "codebuff", "codex", "command-code", "copilot-cli", "cortex-code",
    "cursor", "cursor-ide", "deepseek-tui", "deepseek-harness", "devin",
    "forge", "gemini-cli", "goose", "gptme", "kilo", "kimi-work", "kiro",
    "openclaude", "opencode", "poolside", "positron", "roocode", "trae",
    "vscode-copilot", "windsurf", "zed",
)


def write_config(prefix: Prefix) -> None:
    """The prefix's `config.toml`. Idempotent, and ours alone.

    Written into `AGENTSVIEW_DATA_DIR`, never `~/.agentsview`: a user who
    already runs AgentsView keeps their own archive and settings untouched.
    """
    cfg = prefix.agentsview_data / "config.toml"
    disabled = ", ".join(json.dumps(name) for name in OTHER_PROVIDERS)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "# Written by agent_sys. AgentsView itself is unmodified.\n"
        f"disabled_agents = [{disabled}]\n"
        'host = "127.0.0.1"\n'
        "disable_update_check = true\n"
    )


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
        write_config(prefix)
        env = {**prefix.environment(), "PATH": str(prefix.bin), "HOME": str(prefix.root)}
        proc = subprocess.run(  # noqa: S603
            [str(exe), "serve", "--background", "--no-browser",
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
