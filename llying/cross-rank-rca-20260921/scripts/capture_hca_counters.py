#!/usr/bin/env python3
"""Snapshot all ionic HCA counters/topology on the two RCA nodes as JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

from rca_nodes import NODES

REMOTE = r"""
import glob
import json
import os
import pathlib
import re
import subprocess

def read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return None

def number(value):
    if value is None:
        return None
    try:
        return int(value, 0)
    except ValueError:
        return value

result = {}
for ib_path in sorted(glob.glob("/sys/class/infiniband/ionic_*")):
    ib = os.path.basename(ib_path)
    device = os.path.realpath(os.path.join(ib_path, "device"))
    netdevs = sorted(os.listdir(os.path.join(ib_path, "device", "net")))
    record = {
        "device": device,
        "bdf": os.path.basename(device),
        "numa_node": number(read(os.path.join(device, "numa_node"))),
        "state": read(os.path.join(ib_path, "ports", "1", "state")),
        "active_mtu": read(os.path.join(ib_path, "ports", "1", "active_mtu")),
        "gid1": read(os.path.join(ib_path, "ports", "1", "gids", "1")),
        "netdevs": netdevs,
        "sysfs": {},
        "ethtool": {},
    }
    for counter in sorted(glob.glob(os.path.join(ib_path, "ports", "1", "hw_counters", "*"))):
        record["sysfs"][os.path.basename(counter)] = number(read(counter))
    for netdev in netdevs:
        cp = subprocess.run(
            ["ethtool", "-S", netdev],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        stats = {}
        for line in cp.stdout.splitlines():
            match = re.match(r"\s*([^:]+):\s*(-?(?:0x[0-9a-fA-F]+|\d+))\s*$", line)
            if match:
                stats[match.group(1).strip()] = number(match.group(2))
        record["ethtool"][netdev] = stats
    result[ib] = record
print(json.dumps(result, sort_keys=True))
"""


def ssh_json(node: str) -> dict:
    command = shlex.join(["python3", "-c", REMOTE])
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, command],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=90,
    )
    return json.loads(completed.stdout)


def snapshot(nodes: tuple[str, ...] = NODES) -> dict:
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nodes": {node: ssh_json(node) for node in nodes},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
