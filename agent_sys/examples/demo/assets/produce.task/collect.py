#!/usr/bin/env python3
"""What `produce` runs: walk a tree and write a `facts` content directory.

**This file imports nothing from `agent_sys`.** It is package data, run as a
subprocess by `agent.backends.program.ProgramExecutor`, and the wall in `demo`
design §3.1 says `examples/demo/` is imported by nobody at all — including by
itself. Everything it needs arrives as an environment variable.

| | |
|---|---|
| `AGENT_SYS_TASK_PACKAGE` | the **staged copy** of the package, inside the zone — `interfaces.md` §4.16. It is also the tree walked, so the manifest inventories the copy the task can actually reach |
| `AGENT_SYS_DEMO_PACKAGE` | the fallback, for a run with no prepared environment |
| `AGENT_SYS_OUTPUT_FACTS` | where to write `README.md` and `items/` — the **pre-allocated, granted** `<store>/<hid>/v<N>/content/` |

It writes a `structured_text` content directory: a `README.md` carrying the
`Purpose` and `Schema` sections that content type requires, and one item,
`items/text.json`.

**It does not measure how long it took.** `demo` design §4.3: the gap is in the
specification, not in this program, and it is what makes `check_grounded`'s
failure structural.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

#: Never walked into. `.git` would make the manifest depend on the reviewer's
#: history, and `__pycache__` on whether anything has been run yet — either
#: would break the recomputability `check_facts` claims.
SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}

README = """# facts

## Purpose

A manifest of every file under `{root}`: one row per file, each carrying its
package-relative `path`, its `lines` count, and the first eight hex digits of
its SHA-256. A `totals` object carries the file count and the line count.

## Schema

`items/text.json` is a JSON object:

```json
{{"rows": [{{"path": "...", "lines": 0, "sha256_prefix": "0123abcd"}}],
 "totals": {{"files": 0, "lines": 0}}}}
```

Every number in it is recomputable from the tree by rerunning
`assets/produce.task/collect.py`, which is what lets a validator be honestly `strong` about it.
"""


def rows(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if SKIP & set(path.relative_to(root).parts):
            continue
        data = path.read_bytes()
        out.append(
            {
                "path": str(path.relative_to(root)),
                "lines": data.count(b"\n"),
                "sha256_prefix": hashlib.sha256(data).hexdigest()[:8],
            }
        )
    return out


def _required(name: str) -> str:
    """A named refusal instead of a bare `KeyError`.

    **The declared name is `AGENT_SYS_OUTPUT_<KIND>`**, exported per output slot
    by `env_mgr.grants.output_env` at every dispatch. This body wrote to
    `AGENT_SYS_DEMO_CONTENT` — the pre-§4.14 shape, exported by nobody — and the
    two sides never met: it exited 1 before writing a byte, and the run saw only
    `declared output … was never delivered`. `env_mgr`'s own docstring names that
    symptom from their side.

    **Kept loud rather than defaulted, on their advice.** `output_env` exports
    *nothing* for a kind naming two output slots, deliberately: an author writing
    `outputs: ['facts', 'facts']` cannot address either one, so `env_mgr` refuses
    to invent a scheme. A `KeyError` there would then be **correct behaviour**,
    and this message is the only thing that would say so.
    """
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Since interfaces.md §4.14 a task writes its "
            f"output into the pre-allocated <store>/<hid>/v<N>/content/, which is "
            f"granted to it — but nothing exports that path under a declared "
            f"name, so this body has nowhere to write. See demo/README.md F-D17."
        )
    return value


def main() -> int:
    root = Path(
        os.environ.get("AGENT_SYS_TASK_PACKAGE") or _required("AGENT_SYS_DEMO_PACKAGE")
    ).resolve()
    dst = Path(_required("AGENT_SYS_OUTPUT_FACTS"))
    found = rows(root)
    (dst / "items").mkdir(parents=True, exist_ok=True)
    (dst / "README.md").write_text(README.format(root=root.name), encoding="utf-8")
    (dst / "items" / "text.json").write_text(
        json.dumps(
            {
                "rows": found,
                "totals": {"files": len(found), "lines": sum(r["lines"] for r in found)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"collect: {len(found)} files -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
