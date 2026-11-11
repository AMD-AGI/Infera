#!/usr/bin/env python3
"""Refresh candidate and patch from the current reviewed Forge workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def collect(inputs: Path, workdir: Path) -> None:
    workspace = workdir / "engine_src"
    relative = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("resolve_source.py")),
         "--inputs", str(inputs), "--relative"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    source = (workspace / relative).resolve()
    if not source.is_relative_to(workspace.resolve()) or not source.is_file():
        raise ValueError(f"reviewed candidate is not a file inside {workspace}: {relative}")
    base = (workdir / "engine_baseline.sha").read_text().strip()
    if not base:
        raise ValueError("baseline SHA is missing; refusing to diff against HEAD")

    def diff(commit: str, *paths: str) -> str:
        return subprocess.run(["git", "-C", str(workspace), "diff", commit, *paths],
                              check=True, capture_output=True, text=True).stdout

    if diff(base, "--", ".infera_forge_workset"):
        raise ValueError("Forge changed the protected measurement harness; restore it before collecting")
    interface_file = workdir / "engine_interface.sha"
    interface = interface_file.read_text().strip() if interface_file.is_file() else ""
    patch = "" if interface and not diff(interface) else diff(base)
    shutil.copy2(source, workdir / "optimized_kernel.py")
    (workdir / "engine.patch").write_text(patch, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.inputs, args.workdir)


if __name__ == "__main__":
    main()
