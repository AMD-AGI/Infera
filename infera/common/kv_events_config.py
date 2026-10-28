###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A caller-supplied ``--kv-events-config`` in an engine's forwarded argv."""

from __future__ import annotations

import json

# The router subscribes to exactly this topic; events on any other are dropped.
KV_EVENTS_TOPIC = "kv-events"


def explicit_kv_events_port(argv: list[str]) -> int | None:
    """The publisher port of a caller-supplied ``--kv-events-config``, or None when there is none.

    A launcher that owns port allocation (srt-slurm) passes the config itself so
    the publisher binds the port it allocated; the worker advertises that port
    instead of picking another one that would collide with whatever the
    launcher placed on it.
    """
    raw = None
    for i, arg in enumerate(argv):
        if arg == "--kv-events-config" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif arg.startswith("--kv-events-config="):
            raw = arg.split("=", 1)[1]
    if raw is None:
        return None
    config = json.loads(raw)
    topic = config.get("topic", KV_EVENTS_TOPIC)
    if topic != KV_EVENTS_TOPIC:
        raise ValueError(
            f"--kv-events-config topic {topic!r}: the router only subscribes to {KV_EVENTS_TOPIC!r}"
        )
    endpoint = str(config.get("endpoint", ""))
    _, sep, port = endpoint.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(
            f"--kv-events-config needs a tcp endpoint with a port to advertise; got {raw!r}"
        )
    return int(port)
