#!/usr/bin/env python3
"""Purpose: Read and validate the role, node, and data_ip topology columns.
Usage:
    python3 tools/topology.py validate|rows|nodes|count|node-ip TOPOLOGY [VALUE]
Artifacts:
    None; query commands write their result to stdout.
Artifact paths:
    Not applicable.

The ``rows`` command prints: index, instance, role, node, data_ip.
"""

import csv
import ipaddress
import re
import sys
from pathlib import Path


def load(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["role", "node", "data_ip"]:
            raise SystemExit(f"{path}: expected columns: role, node, data_ip")
        source = list(reader)
    if not source:
        raise SystemExit(f"{path}: topology is empty")
    rows = []
    nodes: set[str] = set()
    ips: set[str] = set()
    counts = {"prefill": 0, "decode": 0}
    for line, row in enumerate(source, 2):
        role = (row.get("role") or "").strip().lower()
        node = (row.get("node") or "").strip()
        data_ip = (row.get("data_ip") or "").strip()
        if role not in counts:
            raise SystemExit(f"{path}:{line}: role must be prefill or decode")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", node):
            raise SystemExit(f"{path}:{line}: unsafe node name {node!r}")
        try:
            parsed = ipaddress.ip_address(data_ip)
        except ValueError:
            raise SystemExit(f"{path}:{line}: invalid data_ip {data_ip!r}") from None
        if parsed.version != 4:
            raise SystemExit(f"{path}:{line}: data_ip must be IPv4")
        if node in nodes or data_ip in ips:
            raise SystemExit(f"{path}:{line}: node and data_ip must be unique")
        nodes.add(node)
        ips.add(data_ip)
        instance = f"{role}-{counts[role]}"
        counts[role] += 1
        rows.append(
            {
                "index": str(line - 2),
                "instance": instance,
                "role": role,
                "node": node,
                "data_ip": data_ip,
            }
        )
    if not all(counts.values()):
        raise SystemExit(f"{path}: at least one prefill and one decode are required")
    return rows


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    command, path, *rest = sys.argv[1:]
    rows = load(path)
    if command == "validate":
        if rest:
            raise SystemExit("validate takes only TOPOLOGY")
    elif command == "rows":
        if rest:
            raise SystemExit("rows takes only TOPOLOGY")
        for row in rows:
            print(
                "\t".join(
                    row[key]
                    for key in ("index", "instance", "role", "node", "data_ip")
                )
            )
    elif command == "nodes":
        if rest:
            raise SystemExit("nodes takes only TOPOLOGY")
        for row in rows:
            print(row["node"])
    elif command == "count":
        if len(rest) != 1 or rest[0] not in {"prefill", "decode"}:
            raise SystemExit("count: expected prefill or decode")
        print(sum(row["role"] == rest[0] for row in rows))
    elif command == "node-ip":
        if len(rest) != 1:
            raise SystemExit("node-ip: expected one node")
        for row in rows:
            if row["node"] == rest[0]:
                print(row["data_ip"])
                return
        raise SystemExit(f"node not found in topology: {rest[0]}")
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
