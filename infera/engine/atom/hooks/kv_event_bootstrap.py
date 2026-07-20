###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Interpreter-startup hook that installs ATOM KV-cache event publishing.

ATOM runs its ``EngineCore`` (which owns the ``BlockManager`` we tap) in a
``multiprocessing`` *spawn* subprocess, so a patch applied only in the
infera launcher or the ATOM API-server process never reaches the block
manager. A site ``.pth`` that ``import``\\s this module runs in *every*
interpreter in the tree — launcher, ATOM server, and the spawned EngineCore
— which is the one reliable place to install the hooks.

The work is gated on ``INFERA_ATOM_KV_EVENTS_ENDPOINT`` so that unrelated
Python invocations (pip, debug shells, the launcher itself before it spawns
ATOM) pay nothing. The infera ATOM launcher sets that env var (to the ZMQ
bind address) only for the ATOM subprocess it spawns.
"""

from __future__ import annotations

import os


def _activate() -> None:
    endpoint = os.environ.get("INFERA_ATOM_KV_EVENTS_ENDPOINT")
    if not endpoint:
        return
    try:
        from infera.engine.atom.hooks.kv_events import arm_kv_event_hooks

        arm_kv_event_hooks(endpoint)
    except Exception:
        # A sitecustomize/.pth hook must never take down an unrelated
        # interpreter; surface the traceback and continue.
        import traceback

        traceback.print_exc()


_activate()
