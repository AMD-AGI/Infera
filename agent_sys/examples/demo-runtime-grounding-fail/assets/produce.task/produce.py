#!/usr/bin/env python3
"""Walk the package tree and write a facts manifest.

Same structure as demo's collect.py but simplified: no duration measurement,
deterministic output. The manifest carries rows with path/lines/sha256_prefix
and totals with files/lines counts.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}

README = """# facts

## Purpose

A manifest of every file under the package tree: one row per file, each
carrying its package-relative path, line count, and SHA-256 prefix.

## Schema

`items/text.json` is a JSON object with `rows` and `totals`.
"""


def main() -> int:
    root = Path(
        os.environ.get("AGENT_SYS_TASK_PACKAGE")
        or os.environ["AGENT_SYS_DEMO_PACKAGE"]
    ).resolve()
    dst = Path(os.environ["AGENT_SYS_OUTPUT_FACTS"])

    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if SKIP & set(path.relative_to(root).parts):
            continue
        data = path.read_bytes()
        found.append({
            "path": str(path.relative_to(root)),
            "lines": data.count(b"\n"),
            "sha256_prefix": hashlib.sha256(data).hexdigest()[:8],
        })

    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README, encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps({
            "rows": found,
            "totals": {"files": len(found), "lines": sum(r["lines"] for r in found)},
        }, indent=2),
        encoding="utf-8",
    )
    print(f"produce: {len(found)} files -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
