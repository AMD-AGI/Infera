# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deploying an agent's rules, hooks and skills into the zone. Design section 11.5.

This module parses nothing: these are paths in Claude Code's canonical form,
and a file is placed, not read. ``rules``/``hooks``/``skills`` are lists of
files; a Claude Code *component* (skill directory, marketplace, MCP server) is
a tree, and placing those, plus deciding what a ``settings.json`` merge must
precede, is `agent_assets.py`'s job. This module keeps the four original keys
and calls the one module that owns the other three.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, NamedTuple

from env_mgr import agent_assets, harness, paths
from env_mgr.fs.layout import LOGS, copy_out
from env_mgr.fs.zone import Zone
from env_mgr.protocols import PrepareRefused

__all__ = ["CONFIG_DIR", "MATERIAL_KEYS", "Deployed", "deploy"]

#: Placed under a per-attempt config directory rather than ``$HOME``. Measured:
#: with ``~/.claude`` granted, a demo agent read the **operator's** personal
#: ``CLAUDE.md`` and obeyed its language rule. A run whose transcript changes
#: with the reviewer's dotfiles is not reproducible, and pointing
#: ``CLAUDE_CONFIG_DIR`` at the zone removes the ``$HOME`` grant entirely.
CONFIG_DIR = "config"

#: The three `agent` hands over, in Claude Code's own directory names.
MATERIAL_KEYS = ("rules", "hooks", "skills")


class Deployed(NamedTuple):
    """What one agent's deployment produced, for three different destinations.

    `environment` is first so the two-argument call still unpacks; `mcp_servers`
    and `report` do not fit a mapping and are returned alongside it.
    """

    environment: dict[str, str]
    #: External and bundled MCP servers, keyed by the name the model addresses
    #: them under. Reaches the backend through `Prepared.mcp_servers`.
    mcp_servers: dict[str, Any]
    #: Per-install `Outcome`s from `agent_assets`. Carried out rather than logged
    #: here, so that whoever renders a prepared environment renders these too and
    #: a failed component install is not a line in a log nobody opened.
    report: tuple[Any, ...] = ()


def deploy(
    agent_spec: Any,
    zone: Zone,
    staged_package: str | None = None,
    workspace: str | None = None,
    base_env: Mapping[str, str] | None = None,
    agent_cli: str | None = None,
) -> Deployed:
    """Place this agent's material in the zone and return what it needs.

    Runs before confinement. `staged_package` is the copy in the zone, never
    `Context.package`. `base_env` seeds installs' child environment.
    """
    config = os.path.join(zone.root, CONFIG_DIR)
    os.makedirs(config, exist_ok=True)
    # A temp directory inside the zone: per attempt, and it dies with the zone.
    # The backend refuses a temp directory it cannot read, and says so well.
    tmp = os.path.join(zone.root, "tmp")
    os.makedirs(tmp, exist_ok=True)

    for key in MATERIAL_KEYS:
        for src in _paths(agent_spec, key):
            dst = os.path.join(config, key, os.path.basename(src))
            if not os.path.exists(src):
                # Declared and absent is an error, not a silent skip: skipping
                # would leave the agent to discover the absence hours later as
                # `Unknown skill: <name>` with nothing naming the cause.
                raise PrepareRefused(
                    f"agent {getattr(agent_spec, 'name', '?')!r} declares "
                    f"{key} {src!r} and it does not exist. It would have been "
                    f"skipped and the agent would meet the absence as a failure "
                    f"of its own, with nothing naming this as the cause"
                )
            copy_out(src, dst)

    env = {"CLAUDE_CONFIG_DIR": config, "CLAUDE_CODE_TMPDIR": tmp, "TMPDIR": tmp}
    # **The other half of the relocation above.** Moving `CLAUDE_CONFIG_DIR` into
    # the zone also moves away the `env` block that holds the endpoint and the
    # credentials, and the agent then reports `Not logged in` and blames itself.
    # `harness` carries that block across; its reserved set is what stops it
    # overwriting the three keys this function just decided, or the derived `PATH`.
    env.update(harness.harness_env())

    # The declared block is resolved before the installs (an agent's `env`
    # feeds the recipe machinery) and applied again after them, so it still
    # outranks the names `agent_assets` contributes.
    declared = _declared_env(agent_spec)
    material = agent_assets.install(
        agent_spec,
        staged_package=staged_package,
        config_dir=config,
        workspace=workspace,
        logs_dir=os.path.join(zone.root, LOGS),
        # `base_env` first so that everything this function decided, and then
        # everything the author declared, still outranks it.
        environ={**(base_env or {}), **env, **declared},
        agent_cli=agent_cli,
    )
    env.update(material.env)
    env.update(declared)

    # The asset directory, copied into the workspace root as a subdirectory
    # (not its contents, to avoid colliding with `workspace.cut`'s clone).
    # Read back out of `material.env` rather than resolved a second time.
    assets = material.env.get(paths.AGENT_ASSETS_ENV_VAR)
    if workspace and assets:
        copy_out(assets, os.path.join(workspace, os.path.basename(assets)))

    return Deployed(
        environment=env,
        mcp_servers=dict(material.mcp_servers),
        report=material.report,
    )


def _paths(agent_spec: Any, key: str) -> tuple[str, ...]:
    value = _get(agent_spec, key)
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


def _declared_env(agent_spec: Any) -> dict[str, str]:
    value = _get(agent_spec, "env")
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return {}


def _get(agent_spec: Any, key: str) -> Any:
    if isinstance(agent_spec, dict):
        return agent_spec.get(key)
    return getattr(agent_spec, key, None)
