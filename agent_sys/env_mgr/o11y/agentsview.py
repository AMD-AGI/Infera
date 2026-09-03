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


def _wait_for_health(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310
                if 200 <= r.status < 400:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def ensure_running(prefix: Prefix, port: int) -> Status:
    """Start the panel, or say in one line why there is none.

    **Every return is a `Status` and every failure logs exactly one warning.**
    One, not two: a caller that sees the same problem reported twice starts
    hunting for two problems.
    """
    url = f"http://127.0.0.1:{port}"
    exe = prefix.bin / "agentsview"

    if not port_is_free(port):
        if _wait_for_health(url, timeout=REUSE_PROBE_TIMEOUT_S):
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

    (prefix.run / "agentsview.port").write_text(str(port))
    return Status(True, "started", url)
