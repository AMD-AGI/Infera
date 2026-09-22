#!/usr/bin/env python3
"""Fail closed when sampler outputs cannot support the planned RCA."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def records(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if record.get("record_type") == "sample":
            result.append(record)
    return result


def latest_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, str):
            result[value] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    engine = latest_by(records(args.engine), "endpoint")
    nodes = latest_by(records(args.nodes), "role")
    now = dt.datetime.now(dt.timezone.utc)
    for kind, samples in (("engine", engine), ("node", nodes)):
        for role, row in samples.items():
            captured = dt.datetime.fromisoformat(row['captured_at'])
            age = (now - captured).total_seconds()
            if age > 30 or age < -5:
                errors.append(f"{role} {kind} sample is stale or future-dated: age={age:.1f}s")

    required_metrics = {
        "sglang:num_queue_reqs",
        "sglang:num_running_reqs",
        "sglang:token_usage",
    }
    for role in ("prefill", "decode"):
        row = engine.get(role)
        if row is None:
            errors.append(f"no engine sample for {role}")
            continue
        if row.get("error"):
            errors.append(f"{role} engine sample: {row['error']}")
            continue
        series = row.get("series")
        if not isinstance(series, list):
            errors.append(f"{role} engine sample has no series list")
            continue
        names = {
            item.get("metric")
            for item in series
            if isinstance(item, dict)
        }
        missing = sorted(required_metrics - names)
        if missing:
            errors.append(f"{role} is missing metrics: {missing}")
        running = [
            item
            for item in series
            if isinstance(item, dict)
            and item.get("metric") == "sglang:num_running_reqs"
        ]
        label_sets = {
            json.dumps(item.get("labels", {}), sort_keys=True)
            for item in running
        }
        if len(label_sets) < 8:
            errors.append(
                f"{role} exposes only {len(label_sets)} labelled "
                "num_running_reqs series; expected at least 8"
            )
        if not {"sglang:cuda_graph_passes", "sglang:cuda_graph_passes_total"} & names:
            warnings.append(
                f"{role} has no cuda_graph_passes yet; recheck after load starts"
            )

    for role in ("prefill", "decode"):
        row = nodes.get(role)
        if row is None:
            errors.append(f"no node sample for {role}")
            continue
        if row.get("error"):
            errors.append(f"{role} node sample: {row['error']}")
            continue
        sample = row.get("sample", {})
        hcas = sample.get("hcas", {}) if isinstance(sample, dict) else {}
        if len(hcas) != 8:
            errors.append(f"{role} exposes {len(hcas)} ionic HCAs; expected 8")
        gpu = sample.get("gpu", {}) if isinstance(sample, dict) else {}
        if gpu.get("returncode") != 0 or not isinstance(gpu.get("json"), dict):
            errors.append(f"{role} rocm-smi JSON sample is invalid")

    result = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
