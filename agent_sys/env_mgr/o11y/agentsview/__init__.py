# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AgentsView as `agent_sys`'s o11y panel. Design: `design.md` beside this file.

Two halves. `agentsview.py` owns the daemon — install, port, launch, ownership,
health. `mapping.py` owns what the panel *shows* — one project per run.

Re-exported here so a caller writes `from env_mgr.o11y.agentsview import
ensure_running`, unchanged from when this package was a single module.
"""

from .agentsview import (
    DEFAULT_PORT,
    OTHER_PROVIDERS,
    RECIPE_PATH,
    Status,
    check_disabled_agents,
    discover_providers,
    ensure_installed,
    ensure_running,
    freshly_installed,
    pinned_version,
    port_is_free,
    resolve_port,
    write_config,
)
from .mapping import ensure_run_project, name_for_run

__all__ = [
    "DEFAULT_PORT",
    "OTHER_PROVIDERS",
    "RECIPE_PATH",
    "Status",
    "check_disabled_agents",
    "discover_providers",
    "ensure_installed",
    "ensure_run_project",
    "ensure_running",
    "freshly_installed",
    "name_for_run",
    "pinned_version",
    "port_is_free",
    "resolve_port",
    "write_config",
]
