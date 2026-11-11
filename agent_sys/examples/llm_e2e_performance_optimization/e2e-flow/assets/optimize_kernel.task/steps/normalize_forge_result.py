#!/usr/bin/env python3
"""Translate Hyperloom's native result without turning missing/error data into success."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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
        "mean_case_speedup": native.get("mean_case_speedup"),
        "snr_db": native.get("snr_db", best.get("snr_db")),
        "iteration_count": rounds,
        "operator": operator,
        "_what_this_file_is": "Hyperloom AKLR's final campaign result, adapted for the Infera handoff; independent workset measurements decide the claim.",
        "forge_native": {origin: native, **({"best_result.json": best} if best else {})},
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
