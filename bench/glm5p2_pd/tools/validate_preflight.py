#!/usr/bin/env python3
"""Validate byte-verified production Mooncake WRITE evidence for every P→D edge."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_topology(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts = {"prefill": 0, "decode": 0}
    expanded = []
    for row in rows:
        role = row["role"].strip().lower()
        item = dict(row)
        item["role"] = role
        item["instance_id"] = f"{role}-{counts[role]}"
        counts[role] += 1
        expanded.append(item)
    return (
        [row for row in expanded if row["role"] == "prefill"],
        [row for row in expanded if row["role"] == "decode"],
    )


def load_host_record(directory: Path, host: str) -> dict[str, Any]:
    candidates = []
    for path in directory.glob("*.json"):
        if path.name == "pair.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("host") == host:
            candidates.append(payload)
    if len(candidates) != 1:
        raise ValueError(
            f"{directory}: expected one preflight record for {host}, found {len(candidates)}"
        )
    return candidates[0]


def validate_pair(
    directory: Path,
    prefill: dict[str, str],
    decode: dict[str, str],
    expected_gpus: int,
) -> dict[str, Any]:
    metadata_path = directory / "pair.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing pair metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_instance") != prefill["instance_id"]:
        raise ValueError(f"{metadata_path}: source instance mismatch")
    if metadata.get("destination_instance") != decode["instance_id"]:
        raise ValueError(f"{metadata_path}: destination instance mismatch")
    source = load_host_record(directory, prefill["node"])
    findings = (source.get("sections") or {}).get("mooncake") or []
    prefix = f"{prefill['node']} -> {decode['node']} "
    by_label: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        message = str(finding.get("message") or "")
        details = finding.get("details") or {}
        if message.startswith(prefix) and details.get("operation") == "write":
            by_label.setdefault(message[len(prefix) :], []).append(finding)
    required = ["rdma", *(f"rdma-gpu{gpu}" for gpu in range(expected_gpus))]
    checks = []
    errors = []
    for label in required:
        matches = by_label.get(label, [])
        passed = [
            finding
            for finding in matches
            if finding.get("level") == "info"
            and (finding.get("details") or {}).get("verified") is True
            and float((finding.get("details") or {}).get("moved_GiB") or 0) > 0
        ]
        checks.append({"label": label, "matches": len(matches), "verified": len(passed) == 1})
        if len(passed) != 1:
            errors.append(
                f"{prefill['instance_id']}->{decode['instance_id']} {label}: "
                f"expected one verified WRITE record, found {len(passed)}"
            )
    return {
        "source_instance": prefill["instance_id"],
        "source_node": prefill["node"],
        "destination_instance": decode["instance_id"],
        "destination_node": decode["node"],
        "checks": checks,
        "errors": errors,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.expected_gpus <= 0:
        raise SystemExit("--expected-gpus must be positive")
    prefills, decodes = load_topology(args.topology)
    results = []
    errors = []
    for prefill in prefills:
        for decode in decodes:
            directory = args.root / f"{prefill['instance_id']}__to__{decode['instance_id']}"
            try:
                result = validate_pair(directory, prefill, decode, args.expected_gpus)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                result = {
                    "source_instance": prefill["instance_id"],
                    "destination_instance": decode["instance_id"],
                    "checks": [],
                    "errors": [str(exc)],
                }
            results.append(result)
            errors.extend(result["errors"])
    payload = {
        "schema": "infera.glm5p2_pd.preflight_validation",
        "expected_edges": len(prefills) * len(decodes),
        "expected_gpus": args.expected_gpus,
        "passed": not errors,
        "pairs": results,
        "errors": errors,
    }
    atomic_json(args.output, payload)
    if errors:
        for error in errors:
            print(f"[preflight] ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"[preflight] PASS: {len(results)} P->D edges, "
        f"{args.expected_gpus} GPU WRITE paths per edge"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
