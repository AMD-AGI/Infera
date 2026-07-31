#!/usr/bin/env python3
"""Delete payload distributions the vendor base already provides.

The overlay's Python trees are built with `pip install --target`, and --target
*ignores the environment*: pip installs the full transitive closure whether or
not the base already has it. That quietly breaks the property the baked engine
images relied on — a plain `pip install` onto the base leaves already-satisfied
requirements untouched and pulls only what is missing.

The cost is not merely size. A --target tree lands FIRST on PYTHONPATH, so every
duplicate SHADOWS the vendor's copy at whatever version PyPI happened to resolve.
Measured against vllm/vllm-openai-rocm on 2026-07-31, an unpruned tree carried 52
distributions of which exactly two were new, and eleven shadowed the base at a
different version:

    numpy         2.5.1  over 2.3.5    <- broke the base's numba on import
    cryptography 50.0.0  over 3.4.8
    protobuf      7.35.1 over 6.33.6
    grpcio        1.83.0 over 1.78.0
    ...

So: run this inside the vendor base after the --target install. Whatever the base
already ships is removed from the tree and the base's own copy is used, exactly as
in the baked images. What remains is genuinely additive — infera plus the handful
of packages no vendor base carries.

    python3 prune_base_dists.py /payload/py312
"""

from __future__ import annotations

import csv
import importlib.metadata as md
import shutil
import sys
from pathlib import Path

# Never prune these even if the base happens to ship a copy: they ARE the payload.
KEEP = {"amd-infera"}


def _canon(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _base_distributions(tree: Path) -> dict[str, str]:
    """Distributions visible in the running interpreter, excluding the tree itself."""
    tree = tree.resolve()
    found: dict[str, str] = {}
    for dist in md.distributions():
        located = getattr(dist, "_path", None)
        if located is not None and tree in Path(located).resolve().parents:
            continue  # a dist inside the tree we are pruning, not the base's
        name = dist.metadata["Name"]
        if name:
            found[_canon(name)] = dist.version or "?"
    return found


def _files_of(dist_info: Path, tree: Path) -> list[Path]:
    """Paths a distribution owns, from its RECORD (relative to the tree)."""
    record = dist_info / "RECORD"
    if not record.is_file():
        return [dist_info]
    paths: list[Path] = []
    with record.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or not row[0]:
                continue
            target = (tree / row[0]).resolve()
            if tree.resolve() in target.parents:  # never escape the tree
                paths.append(target)
    paths.append(dist_info)
    return paths


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <payload-tree>", file=sys.stderr)
        return 2
    tree = Path(sys.argv[1])
    if not tree.is_dir():
        print(f"prune: {tree} is not a directory", file=sys.stderr)
        return 2

    base = _base_distributions(tree)
    pruned: list[str] = []
    kept: list[str] = []

    for dist_info in sorted(tree.glob("*.dist-info")):
        name = dist_info.name.rsplit("-", 2)[0]
        canon = _canon(name)
        if canon in KEEP:
            kept.append(f"{name} (payload)")
            continue
        if canon not in base:
            kept.append(name)
            continue

        for path in _files_of(dist_info, tree):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        shutil.rmtree(dist_info, ignore_errors=True)
        pruned.append(f"{name} (base has {base[canon]})")

    # Empty package directories left behind once their files are gone.
    for _ in range(4):  # nested packages need a few passes
        for path in sorted(tree.rglob("*"), key=lambda p: -len(p.parts)):
            if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
                path.rmdir()

    print(f"prune: {tree}")
    print(f"prune: removed {len(pruned)} distribution(s) the base already provides")
    for line in pruned:
        print(f"  - {line}")
    print(f"prune: kept {len(kept)} — this is what the overlay actually adds")
    for line in kept:
        print(f"  + {line}")

    if not any(k.endswith("(payload)") for k in kept):
        print("prune: FATAL — infera itself is missing from the tree", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
