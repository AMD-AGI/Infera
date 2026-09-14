#!/usr/bin/env python3
"""Publish a workset directory as a `code`-typed handoff, unrenamed.

The three variables this body reads, and what they point at
(`env_mgr/grants.py`, `env_mgr/paths.py`):

| variable | value |
|---|---|
| `AGENT_SYS_OUTPUT_WORKSET` | `<store>/<hid>/v<N>/content` — **`content/` itself**. Write `README.md` and `items/` directly here; do not create a `content/` hop |
| `AGENT_SYS_TASK_PACKAGE`   | the **staged** copy of this package inside the zone |
| `KFO_WORKSET` / `KFO_WORKSET_DIR` | which workset, from the agent spec's `env:` block |

**The read of the output variable stays loud.** `output_env` deliberately
exports nothing for a task whose `outputs` name one kind twice, because an
author who wrote that cannot address either slot — so a `KeyError` here is
correct behaviour and this message is the only thing that would say so.

**This body creates and copies. It deletes nothing.** The output directory is
pre-allocated and granted; if it already holds a workset that is a re-run
against a sealed version and the right answer is to fail, not to clear it.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn

_README = """# Workset — {name}

## Purpose

Everything needed to test, optimize and re-integrate the operator `{name}`: the
operator's own source, a PyTorch-naive baseline, a one-shot correctness and
benchmark driver, at least three correctness cases, the measured baseline, and
the environment all of that was measured in.

This is stage 3's deliverable and stage 4's input. It was published from
{origin}.

## Interface

The workset directory is at `items/codes/{name}/`. Entry points:

- `README.md` — the operator definition and its provenance. **Read this first.**
- `program.md` — the brief handed to an optimizer: objective, headroom, rules.
- `kernel/driver.py` — the measurement oracle. Three modes on stdout:
  correctness (default), `--bench-mode`, `--profile-run`.
- `kernel/measure_baseline.py` — the 5-round baseline protocol, re-runnable.
- `baseline_measurement.md` — the numbers the driver produces, and the
  cross-check against the profile that anchors them.
- `integration.md` — where the operator plugs back into the serving stack.
- `environment.md` — hardware, image, versions, and what was installed on top.

A consumer optimizes the kernel and **must not** edit `driver.py` or
`graph_harness.py`: they are the oracle, and a change to them makes every
measurement incomparable with the baseline.

## Boundary

What this workset does **not** carry:

- No re-integration patch and no end-to-end claim. Kernel-level only.
- No multi-GPU and no TP > 1.
- Only the dtype and vocabulary the operator was traced at. A different shape
  may want a different strategy and is not covered by these cases.
- The baseline was measured on the host named in `environment.md`, on a shared
  machine, with no clock lock and no exclusive reservation.
"""


def _fail(message: str) -> NoReturn:
    """Exit nonzero *after* saying why.

    `exit 1` alone is a true statement that ends the investigation where it
    starts: the runner reports `exit <code>: <stderr tail>`, so the tail is the
    only diagnostic anyone gets.

    **`NoReturn` is load-bearing, not decoration.** Without it a checker reads
    every `out` below as possibly-unbound — which is the same shape as the bug
    that motivates this file's "delete nothing" rule: a helper that dies before
    binding a variable, and a caller that carries on with the empty value.
    """
    print(f"publish_workset: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    try:
        out = Path(os.environ["AGENT_SYS_OUTPUT_WORKSET"])
    except KeyError:
        _fail(
            "AGENT_SYS_OUTPUT_WORKSET is not set. It is exported per declared output "
            "kind; it is absent when a task names one kind twice in `outputs`, "
            "because then neither slot can be addressed."
        )

    name = os.environ.get("KFO_WORKSET", "").strip()
    if not name:
        _fail("KFO_WORKSET is empty; the agent spec's env block should default it")

    # An explicit directory wins over package data. This is the seam where
    # stage 3's output arrives once stage 3 exists.
    override = os.environ.get("KFO_WORKSET_DIR", "").strip()
    if override:
        source, origin = Path(override), f"`--var workset_dir={override}`"
    else:
        package = os.environ.get("AGENT_SYS_TASK_PACKAGE")
        if not package:
            _fail("neither KFO_WORKSET_DIR nor AGENT_SYS_TASK_PACKAGE is set; nothing to copy from")
        source = Path(package) / "assets" / "worksets" / name
        origin = f"package data at `assets/worksets/{name}/`"

    if not source.is_dir():
        _fail(f"workset source {source} is not a directory")

    # `code` requires exactly one top-level item, `codes`, and rejects anything
    # else before a reader gets to it. The workset keeps its own name inside.
    destination = out / "items" / "codes" / name
    if destination.exists():
        _fail(
            f"{destination} already exists. This version was already written; "
            "publishing again would mean overwriting a sealed artefact. "
            "Nothing was changed."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    # `copytree`, not `move`: the package's own copy is the source of truth and
    # stays exactly where it is.
    shutil.copytree(source, destination)

    (out / "README.md").write_text(_README.format(name=name, origin=origin), encoding="utf-8")

    files = sum(1 for p in destination.rglob("*") if p.is_file())
    print(f"publish_workset: {files} files -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
