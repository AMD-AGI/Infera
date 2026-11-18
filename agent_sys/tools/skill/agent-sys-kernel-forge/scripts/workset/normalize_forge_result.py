#!/usr/bin/env python3
"""Translate Hyperloom's native result without turning missing/error data into success."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Export measured values and opaque identities, not upstream runtime metadata.
# The complete records (including host paths and agent diagnostics) stay in the
# run's native JSON/stdout; new upstream fields are not automatically published.
_NUMBERS = (
    "baseline_ms",
    "pristine_baseline_ms",
    "search_start_ms",
    "best_ms",
    "baseline_wall_ms",
    "best_wall_ms",
    "mean_case_speedup",
    "search_start_mean_case_speedup",
    "total_speedup",
    "incremental_speedup",
    "speedup",
    "snr_db",
    "iteration",
    "iteration_count",
    "next_iteration",
    "best_iteration",
    "session_index",
    "segment_index",
)
_FLAGS = (
    "improved",
    "total_improved",
    "incremental_improved",
    "improved_during_search",
    "correctness_passed",
    "persistence_degraded",
)
_IDENTITIES = (
    "best_commit",
    "commit_hash",
    "campaign_id",
    "experiment_id",
    "termination_reason",
    "agent_backend",
)


def _number(value, name: str):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"Forge's {name} must be numeric or null")
    return value


def _portable_summary(record: dict) -> dict:
    summary = {name: _number(record[name], name) for name in _NUMBERS if name in record}
    for name in _FLAGS:
        if name in record:
            if record[name] is not None and not isinstance(record[name], bool):
                raise ValueError(f"Forge's {name} must be boolean or null")
            summary[name] = record[name]
    for name in _IDENTITIES:
        value = record.get(name)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
            summary[name] = value
    return summary


def normalize(workdir: Path, workspace: Path, operator: str) -> dict:
    result = workdir / "forge_native_result.json"
    if result.is_file():
        native = json.loads(result.read_text())
        origin = "--result-json"
    else:
        stdout = (workdir / "forge_stdout.log").read_text()
        records = re.findall(r"__FORGE_RESULT__(.*?)__FORGE_RESULT__", stdout, re.DOTALL)
        if not records:
            raise ValueError("Forge emitted neither --result-json nor its final stdout result")
        native = json.loads(records[-1])
        origin = "__FORGE_RESULT__"
    if not isinstance(native, dict) or "error" in native:
        raise ValueError(f"Forge failed: {native}")
    improved = native.get("total_improved", native.get("improved"))
    if not isinstance(improved, bool):
        raise ValueError("Forge's final result contains no boolean improvement verdict")
    rounds = native.get("iteration_count")
    if rounds is None and isinstance(native.get("next_iteration"), int):
        rounds = native["next_iteration"] - 1
    if rounds is not None and (
        isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0
    ):
        raise ValueError("Forge's iteration count is invalid")
    best_file = workspace / "forge_experiments/best_result.json"
    best = json.loads(best_file.read_text()) if best_file.is_file() else {}
    return {
        "mock": False,
        "ran": True,
        "improved": improved,
        "mean_case_speedup": _number(native.get("mean_case_speedup"), "mean_case_speedup"),
        "snr_db": _number(native.get("snr_db", best.get("snr_db")), "snr_db"),
        "iteration_count": rounds,
        "operator": operator,
        "_what_this_file_is": "Portable scalar summary of Hyperloom AKLR's final campaign result; full native JSON/stdout remain in the run logs. Independent workset measurements decide the claim.",
        "forge_native": {
            origin: _portable_summary(native),
            **({"best_result.json": _portable_summary(best)} if best else {}),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    record = normalize(args.workdir, args.workspace, args.operator)
    (args.workdir / "forge_result.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
