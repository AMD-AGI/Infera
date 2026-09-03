# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deploying an agent's rules, hooks and skills into the zone. Design §11.5.

`agent` spec §3.1 says ``env`` is *"resolved by env_mgr"* and `agent` design
§3.4 says ``rules``, ``hooks`` and ``skills`` are handed *"to env_mgr to
deploy"*. Four keys, one named consumer, and until design rev. 4 no route.

**This module parses nothing.** Those are paths in Claude Code's canonical form;
converting between harness formats is an independent module that does not exist.
A file is placed, not read.

**Three keys later joined them and they did not fit that sentence**, which is
why `agent_assets.py` exists rather than a fourth loop here. ``rules`` /
``hooks`` / ``skills`` are lists of *files*, and a Claude Code component is a
*tree* — a skill is a directory, a marketplace is a directory of directories, an
MCP server is a process to register. Placing those is still placing; deciding
which of them a `settings.json` merge has to precede is not, and that decision
belongs to whoever knows which levels resolved. So this module keeps the four
original keys and calls the one that owns the other three.
"""

from __future__ import annotations

import os
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

    **A value where there was a `dict[str, str]`**, and the change is not
    tidiness: `mcp_servers` and `tools` do not fit in an environment mapping and
    never could — one is a nested JSON document and the other is a tuple of live
    Python objects. Returning them alongside is what stops a caller having to
    call a second function afterwards and remember the order
    (`engineer_principle.md` §1).

    `environment` is first because it is what the two-argument call has always
    produced; a caller updating from it reads the same at the call site.
    """

    environment: dict[str, str]
    #: External and bundled MCP servers, keyed by the name the model addresses
    #: them under. Reaches the backend through `Prepared.mcp_servers`.
    mcp_servers: dict[str, Any]
    #: `ToolDef`-shaped in-process tools, appended to `Prepared.tools`. Typed
    #: loosely for that field's reason — it crosses to `agent`.
    tools: tuple[Any, ...]
    #: Per-install `Outcome`s from `agent_assets`. Carried out rather than logged
    #: here, so that whoever renders a prepared environment renders these too and
    #: a failed component install is not a line in a log nobody opened.
    report: tuple[Any, ...] = ()


def deploy(
    agent_spec: Any,
    zone: Zone,
    staged_package: str | None = None,
    workspace: str | None = None,
) -> Deployed:
    """Place this agent's material in the zone and return what it needs.

    Runs at prepare step 6b — **before** confinement and after the zone exists,
    because deploying is writing into the zone and confinement makes writing
    impossible. It sits beside handoff staging because it is the same kind of
    act: putting something the executor will need where the executor can reach
    it.

    A declared ``env`` requirement is what the shipped recipe machinery already
    resolves; what is new is only that it now has a route from the agent spec.
    Returned rather than applied, because this module does not own the
    executor's process.

    **The two new parameters are defaulted so that `interfaces.md` §4.6's frozen
    two-argument call still type-checks** — the allowance `agent_spec` already
    has on `EnvManager.prepare`. Omitting them is not a degraded mode with a
    silent cost: `staged_package` absent means there is no staged copy to
    resolve components against, which is exactly the state of a run that
    configured no package, and `workspace` absent means there is no workspace to
    copy the asset directory into.

    `staged_package`, never `Context.package`: the original checkout is outside
    every grant (`interfaces.md` §4.16), so a component installed from it would
    place files the confined session cannot read.
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
                # **Declared and absent is an error, not a shrug.** This was
                # `if os.path.exists(src): copy_out(...)` with no else, and the
                # failure it produces is invisible at every point where anyone
                # could act on it: the copy is skipped silently, the run
                # proceeds, and the agent discovers it hours later as
                # `Unknown skill: <name>` from inside its own session — with
                # nothing in the zone, the events or the logs naming the cause.
                #
                # Measured 2026-08-31: an agent mid-run called
                # `Skill{"experiment-result-packup"}`, got `Unknown skill`, and
                # started `find / -name ...` looking for it. That instance was a
                # package declaring no skill at all; this guard is for the one
                # after it, where the declaration is present and the path is
                # wrong — which is the same bug wearing a fix.
                #
                # `fail closed` is this package's own rule (`locality.py`: "an
                # oracle whose prefix cannot be formed is an error, not a
                # silently widened blind spot"), and no shipped package declares
                # any material today, so nothing existing changes behaviour.
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

    # **The declared block is resolved BEFORE the installs, and then applied
    # again after them.** That reads like a duplication and is two different
    # facts.
    #
    # *Before*, because it is an input: an agent declaring `recipes: [serena]`
    # also declares `UV_TOOL_DIR` / `UV_TOOL_BIN_DIR` / `UV_CACHE_DIR` in its
    # `env` block, and `uv tool install` writes `~/.local/share/uv` — host state
    # on a shared box — if they are not in the child's environment. Measured
    # 2026-09-03 (probe D): the install *succeeds* while doing it, so nothing
    # downstream reports anything wrong. The same block is what a component's
    # `.mcp.json` expands `${UV_TOOL_BIN_DIR}` against, so the recipe that
    # installed the binary and the entry that launches it cannot disagree.
    #
    # *After*, because it is also the last word: `an author saying so outranks a
    # default` is the precedence
    # `test_a_declared_env_overrides_every_contributor` pins, and it must still
    # outrank the three names `agent_assets` contributes.
    declared = _declared_env(agent_spec)
    material = agent_assets.install(
        agent_spec,
        staged_package=staged_package,
        config_dir=config,
        workspace=workspace,
        logs_dir=os.path.join(zone.root, LOGS),
        environ={**env, **declared},
    )
    env.update(material.env)
    env.update(declared)

    # **The asset directory, into the workspace root as a subdirectory.** The
    # staged copy under `<zone>/package/` is where it is *installed from* and is
    # the agent's own package, several directories away from its `cwd`; this is
    # the copy it can `ls`. A subdirectory rather than the contents, because
    # spilling a component's files into the workspace root would collide with
    # whatever `workspace.cut` cloned there — and a collision between an
    # agent's material and its working tree is the one nobody would attribute
    # correctly.
    #
    # **Read back out of `material.env`, not resolved a second time.**
    # `agent_assets` owns the rule that turns `AgentSpec.assets` into a path and
    # already published the answer under the name `paths` spells; asking it
    # again here would be a second reader of one fact and would go stale the
    # first time that rule changes.
    assets = material.env.get(paths.AGENT_ASSETS_ENV_VAR)
    if workspace and assets:
        copy_out(assets, os.path.join(workspace, os.path.basename(assets)))

    return Deployed(
        environment=env,
        mcp_servers=dict(material.mcp_servers),
        tools=material.tools,
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
