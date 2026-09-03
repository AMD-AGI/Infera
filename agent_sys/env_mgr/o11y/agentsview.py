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

import logging
import socket
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["DEFAULT_PORT", "Status", "port_is_free", "resolve_port"]

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
