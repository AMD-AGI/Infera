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
import os
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
    # Several instances may share one node and its data_ip, including two of the
    # same role (two TP4 prefills on one 8-GPU node). Rows are told apart by their
    # instance name; launch.sh rejects overlapping GPU sets on one node.
    # "mixed" rows are aggregated workers (no PD disaggregation); a topology is either all
    # mixed or has at least one prefill and one decode.
    counts = {"prefill": 0, "decode": 0, "mixed": 0}
    for line, row in enumerate(source, 2):
        role = (row.get("role") or "").strip().lower()
        node = (row.get("node") or "").strip()
        data_ip = (row.get("data_ip") or "").strip()
        if role not in counts:
            raise SystemExit(f"{path}:{line}: role must be prefill, decode or mixed")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", node):
            raise SystemExit(f"{path}:{line}: unsafe node name {node!r}")
        try:
            parsed = ipaddress.ip_address(data_ip)
        except ValueError:
            raise SystemExit(f"{path}:{line}: invalid data_ip {data_ip!r}") from None
        if parsed.version != 4:
            raise SystemExit(f"{path}:{line}: data_ip must be IPv4")
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
    def complete(roles: set[str]) -> bool:
        return roles == {"mixed"} or ("mixed" not in roles and {"prefill", "decode"} <= roles)

    if not complete({role for role, count in counts.items() if count}):
        raise SystemExit(f"{path}: need only mixed workers, or at least one prefill and one decode")
    # EXCLUDE_INSTANCES (e.g. "prefill-0") drops instances that are deliberately down on a
    # live stack (2P1D out of a 3P1D topology); the others keep their index, name and ports.
    excluded = set(os.environ.get("EXCLUDE_INSTANCES", "").replace(",", " ").split())
    unknown = excluded - {row["instance"] for row in rows}
    if unknown:
        raise SystemExit(f"{path}: EXCLUDE_INSTANCES names unknown instances {sorted(unknown)}")
    rows = [row for row in rows if row["instance"] not in excluded]
    if not complete({row["role"] for row in rows}):
        raise SystemExit(f"{path}: EXCLUDE_INSTANCES leaves an incomplete topology")
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
