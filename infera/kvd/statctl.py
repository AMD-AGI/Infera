###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``python -m infera.kvd.statctl`` — print a kvd daemon's current
stats as JSON. Tiny ops tool for inspecting a running kvd.

Usage:

    python -m infera.kvd.statctl --socket /run/infera-kvd/kvd.sock

In a container:

    docker exec <kvd-container> python -m infera.kvd.statctl

Output (stdout, one JSON object per call):

    {
      "entries": 1234,
      "host_bytes": 524288,
      "spillover_bytes": 0,
      "long_bytes": 0,
      "gets_total": 42,
      "sets_total": 5,
      "hits_total": 38,
      "misses_total": 4,
      "evictions_total": 0
    }

Exit codes:
- 0: connected, stats printed
- 1: connection / protocol error (kvd unreachable, version mismatch)

Why a separate CLI rather than just a /metrics endpoint:
- kvd is a node-local daemon, intentionally UDS-only — no HTTP surface.
- Prometheus scraping happens via the infera server's /metrics
  (which can be wired to call this client lazily); this CLI is for
  ad-hoc debugging from a shell inside the container.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys

from infera.kvd.client import KvdClient, KvdConnectionError, KvdProtocolError


async def _print_stats(socket_path: str, client_id: str) -> int:
    client = KvdClient(socket_path, client_id=client_id)
    try:
        await client.connect()
    except KvdConnectionError as exc:
        print(f"kvd statctl: failed to connect to {socket_path}: {exc}", file=sys.stderr)
        return 1

    try:
        stats = await client.stats()
    except KvdProtocolError as exc:
        print(f"kvd statctl: protocol error: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()

    # dataclass-asdict preserves the field order from wire.py — handy
    # for grep/cut pipelines that some ops scripts will rely on.
    print(json.dumps(dataclasses.asdict(stats), indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="infera-kvd stats inspector")
    parser.add_argument(
        "--socket",
        default="/var/run/infera-kvd.sock",
        help="Path to the kvd UDS socket. Default: /var/run/infera-kvd.sock",
    )
    parser.add_argument(
        "--client-id",
        default="statctl",
        help="Client ID announced in the HELLO frame. Default: statctl",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_print_stats(args.socket, args.client_id)))


if __name__ == "__main__":
    main()
