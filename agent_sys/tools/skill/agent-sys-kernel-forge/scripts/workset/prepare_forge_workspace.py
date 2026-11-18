#!/usr/bin/env python3
"""Prepare Forge-owned metadata before committing the engine baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(workspace: Path, operator: str) -> None:
    if (
        not operator
        or operator in {".", ".."}
        or Path(operator).is_absolute()
        or "/" in operator
        or "\\" in operator
        or "\0" in operator
    ):
        raise ValueError("operator must be one nonempty safe path component")
    root = workspace / ".infera_forge_workset/operators" / operator
    source = root / "invocation_spec.json"
    driver = root / "scripts/forge_driver.py"
    if not driver.is_file():
        raise ValueError(f"the protected staged driver is missing: {driver}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("the invocation specification must be a JSON object")
    destination = driver.with_name("invocation_spec.json")
    if destination.is_symlink():
        raise ValueError("the driver-adjacent invocation specification cannot be a symlink")
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != payload:
            raise ValueError(
                "the driver-adjacent specification differs from the authoritative workset"
            )
    else:
        # Match Hyperloom's _materialize_invocation_spec. Its startup preflight
        # preserves an existing equal JSON payload, leaving this tracked file
        # unchanged rather than creating an untracked measurement dependency.
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    git_dir = workspace / ".git"
    if not git_dir.is_dir():
        raise ValueError("initialize the fresh engine workspace with git init first")
    exclude = git_dir / "info/exclude"
    if exclude.is_symlink():
        raise ValueError("the owned workspace's Git exclude file cannot be a symlink")
    existing = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
    patterns = ("/forge_experiments/", "/.stop")
    missing = [pattern for pattern in patterns if pattern not in existing.splitlines()]
    if missing:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text(
            existing
            + ("\n" if existing and not existing.endswith("\n") else "")
            + "# Forge-owned runtime state; kernel sources and the harness remain tracked.\n"
            + "\n".join(missing)
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    prepare(args.workspace, args.operator)


if __name__ == "__main__":
    main()
