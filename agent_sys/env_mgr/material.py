# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deploying an agent's rules, hooks and skills into the zone. Design §11.5.

`agent` spec §3.1 says ``env`` is *"resolved by env_mgr"* and `agent` design
§3.4 says ``rules``, ``hooks`` and ``skills`` are handed *"to env_mgr to
deploy"*. Four keys, one named consumer, and until design rev. 4 no route.

**This module parses nothing.** Those are paths in Claude Code's canonical form;
converting between harness formats is an independent module that does not exist.
A file is placed, not read.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from env_mgr import harness
from env_mgr.fs.layout import copy_out
from env_mgr.fs.zone import Zone
from env_mgr.o11y.prefix import Prefix
from env_mgr.protocols import PrepareRefused

__all__ = ["CONFIG_DIR", "MATERIAL_KEYS", "PROJECTS_DIR", "deploy"]

log = logging.getLogger("env_mgr.material")

#: Placed under a per-attempt config directory rather than ``$HOME``. Measured:
#: with ``~/.claude`` granted, a demo agent read the **operator's** personal
#: ``CLAUDE.md`` and obeyed its language rule. A run whose transcript changes
#: with the reviewer's dotfiles is not reproducible, and pointing
#: ``CLAUDE_CONFIG_DIR`` at the zone removes the ``$HOME`` grant entirely.
CONFIG_DIR = "config"

#: Claude Code's own name for *where the transcripts go*: one subdirectory per
#: working directory, named by slugifying that path, one JSONL per session.
PROJECTS_DIR = "projects"

#: The three `agent` hands over, in Claude Code's own directory names.
MATERIAL_KEYS = ("rules", "hooks", "skills")


def deploy(agent_spec: Any, zone: Zone) -> dict[str, str]:
    """Place this agent's material in the zone and return the environment it needs.

    Runs at prepare step 6b — **before** confinement and after the zone exists,
    because deploying is writing into the zone and confinement makes writing
    impossible. It sits beside handoff staging because it is the same kind of
    act: putting something the executor will need where the executor can reach
    it.

    A declared ``env`` requirement is what the shipped recipe machinery already
    resolves; what is new is only that it now has a route from the agent spec.
    Returned rather than applied, because this module does not own the
    executor's process.
    """
    config = os.path.join(zone.root, CONFIG_DIR)
    os.makedirs(config, exist_ok=True)
    _share_projects(config)
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
    env.update(_declared_env(agent_spec))
    return env


def _share_projects(config: str) -> None:
    """Point this attempt's ``config/projects`` at the o11y prefix's.

    **Everything else in ``config/`` stays per-attempt.** The relocation above
    is the reason this module exists and it is not being undone — credentials,
    settings, ``sessions/`` and ``backups/`` remain the zone's own. Only
    ``projects/`` is shared, because it is the one subdirectory whose contents
    nobody in the zone reads: it is Claude Code's *output*, one JSONL per
    session, and the o11y panel is its only consumer.

    Measured on `examples/demo2` (`recon/ACCEPTANCE.md`, check 2): nine agent
    transcripts landed under nine different ``<zone>/config/projects/`` and the
    panel — which reads one fixed directory — showed none of them. The prefix is
    the only anchor stable across runs and across ``--demo-root`` changes, which
    is what a resident daemon needs.

    Sharing one physical directory cannot collide: Claude Code names each
    subdirectory after the slugified cwd, and every attempt runs in its own zone.

    **Never raises.** A panel that cannot see a run is a degraded panel; a
    prepare step that throws is a dead run. One warning, and the attempt
    proceeds with whatever is really at that path.
    """
    link = Path(config) / PROJECTS_DIR
    try:
        target = Prefix.resolve(os.environ).claude_home / PROJECTS_DIR
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == target.resolve():
                return  # idempotent: re-running finds its own work and stops
            # One named path, and one we placed ourselves: a symlink at exactly
            # `<zone>/config/projects`. Never a tree, never a variable target.
            link.unlink()
        elif link.exists():
            # A real directory. `rmdir` **refuses** a non-empty one, which is
            # the whole reason it is the call used here: an empty directory is
            # ours to replace, and one holding transcripts raises `OSError` and
            # falls to the warning below. Losing a zone from the panel is
            # cheaper than deleting somebody's evidence.
            os.rmdir(link)
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        # `Prefix.resolve` used to raise `KeyError` with neither `AGENT_SYS_HOME`
        # nor `$HOME` set, and this was the only call site that survived it. It
        # is total now — the fix went where the other two callers could reach
        # it — so the only thing left to catch here is the filesystem.
        log.warning(
            "could not share %s with the o11y prefix (%s); this attempt's "
            "transcripts stay in its zone and the panel will not show them",
            link,
            exc,
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
