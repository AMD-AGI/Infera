# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small optional helpers for ordinary catalog files."""

import argparse
import json
import shutil
import sys
from importlib.resources import files
from pathlib import Path

from .capture import capture_environment
from .registry import Registry, file_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "validate", "history", "file"):
        command = commands.add_parser(name)
        command.add_argument("catalog", type=Path)
        if name == "history":
            command.add_argument("--kernel")
            command.add_argument(
                "--json", action="store_true", help="show full records and qualifiers"
            )
        if name == "file":
            command.add_argument("path")
    capture = commands.add_parser("capture-env")
    capture.add_argument("--name", required=True)
    capture.add_argument("--repo", action="append", default=[])
    capture.add_argument("--image")
    capture.add_argument("--image-digest")
    capture.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        status = 0
        if args.command == "init":
            if args.catalog.exists():
                raise ValueError("destination already exists; choose a new catalog directory")
            shutil.copytree(str(files("kernel_registry") / "templates"), args.catalog)
            result = {
                "catalog": str(args.catalog),
                "next": "edit the example, then validate and commit",
            }
        elif args.command == "file":
            result = file_record(args.catalog, args.path)
        elif args.command == "validate":
            result = Registry(args.catalog).validate()
            status = 0 if result["ok"] else 1
        elif args.command == "history":
            result = Registry(args.catalog).history(args.kernel)
            if not args.json:
                print("Run\tType\tCorrectness (scope)\tMeasured ratios\tCaveats")
                for row in result:
                    correctness = row["correctness"]
                    verdict = f"{correctness.get('status', 'not_tested')} ({correctness.get('scope', 'unspecified')})"
                    ratios = (
                        ", ".join(f"{m['name']}={m['ratio']:.3f}x" for m in row["metrics"]) or "—"
                    )
                    caveats = set(row["qualifiers"])
                    for metric in row["metrics"]:
                        caveats.update(metric["qualifiers"])
                    print(f"{row['name']}\t{row['run_type']}\t{verdict}\t{ratios}\t{len(caveats)}")
                print(
                    "Ratios are observations, not recommendations. Use --json for all caveats, stack and raw values."
                )
                return 0
        else:
            result = capture_environment(args.name, args.repo, args.image, args.image_digest)
            if args.output:
                with args.output.open("x") as stream:
                    stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
                result = {"output": str(args.output), "missing": result["missing"]}
        print(json.dumps(result, indent=2, allow_nan=False))
        return status
    except (OSError, ValueError) as exc:
        print(f"kernel-registry: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
