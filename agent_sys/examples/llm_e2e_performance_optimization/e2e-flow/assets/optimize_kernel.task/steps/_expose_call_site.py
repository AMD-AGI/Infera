#!/usr/bin/env python3
"""Give a `call_site_fragment` operator one interface both sides can call.

**The gap this closes.** forge edits the engine module in place (`--kernel` is
`edit_target.source_file`), and m3's harness loads `--impl` as a file defining
the workset's function. For a fragment operator the stock module defines no such
function: the computation is a line inside `edit_target.entry_function`. So the
campaign's output and the oracle's input never met — measured on the first
real campaign, STEP 4 had nothing it could call.

**The interface is the signature m3 already declares.** Before the campaign,
on forge's branch, this script

* appends the Definition's `baseline` function named by
  `integration.signature` (e.g. `sampler_softmax`) to the module, with the
  baseline's own imports; and
* replaces `integration.call_site.fragment`, found exactly once inside
  `entry_function`, with `integration.call_site.call`.

The engine then calls the operator through that function, forge optimises it
there, and the harness calls the same function out of the edited module. The
change is committed by the caller as its own commit, so `engine.patch` (cut
against the stock baseline) carries it to m5 as part of one single-file diff.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import NoReturn


def _die(message: str) -> NoReturn:
    sys.exit(f"_expose_call_site: {message}")


def signature_name(signature: str) -> str:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature or "")
    if not match:
        _die(f"integration.signature {signature!r} does not start with a function name")
    return match.group(1)


def baseline_definition(baseline: str, name: str) -> str:
    """The baseline's imports and its `name` function, as source to append."""
    tree = ast.parse(baseline)
    lines = baseline.splitlines()
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)) or (
            isinstance(node, ast.FunctionDef) and node.name == name
        ):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            keep.append("\n".join(lines[start - 1 : node.end_lineno]))
    if not any(re.match(rf"(@.*\n)*def {re.escape(name)}\(", k) for k in keep):
        _die(f"the Definition's baseline defines no top-level `{name}`")
    return "\n".join(keep)


def entry_span(tree: ast.Module, entry_function: str) -> tuple[int, int]:
    nodes, node = tree.body, None
    for part in entry_function.split("."):
        node = next((n for n in nodes if getattr(n, "name", None) == part), None)
        if node is None:
            _die(f"`edit_target.entry_function` {entry_function!r} is not defined in the module")
        nodes = getattr(node, "body", [])
    assert node is not None and node.end_lineno is not None
    return node.lineno - 1, node.end_lineno


def expose(text: str, entry_function: str, fragment: str, call: str, definition: str) -> str:
    lines = text.splitlines(keepends=True)
    start, end = entry_span(ast.parse(text), entry_function)
    want = [line.strip() for line in fragment.strip().splitlines()]
    hits = [
        i
        for i in range(start, end - len(want) + 1)
        if [lines[i + k].strip() for k in range(len(want))] == want
    ]
    if len(hits) != 1:
        _die(
            f"integration.call_site.fragment occurs {len(hits)} times inside {entry_function}; "
            "it must name exactly one place"
        )
    at = hits[0]
    indent = lines[at][: len(lines[at]) - len(lines[at].lstrip(" \t"))]
    lines[at : at + len(want)] = [indent + call.strip() + "\n"]
    body = "".join(lines)
    if not body.endswith("\n"):
        body += "\n"
    banner = "# --- m4: the operator's interface (integration.signature), called above ---\n"
    return body + "\n\n" + banner + definition + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="STEP 1's pinned inputs")
    ap.add_argument("--file", required=True, help="the module in forge's workspace")
    a = ap.parse_args()
    pinned = json.loads(Path(a.inputs).read_text(encoding="utf-8"))
    integration = pinned.get("integration") or {}
    call_site = integration.get("call_site") or {}
    if not call_site.get("fragment") or not call_site.get("call"):
        _die(
            "the operator is `call_site_fragment` and declares no integration.call_site "
            "{fragment, call}; without it forge's edit and the harness have no common interface"
        )
    name = signature_name(str(integration.get("signature") or ""))
    root = Path(pinned["workset_root"])
    definition = json.loads((root / str(pinned["definition"])).read_text(encoding="utf-8"))
    entry = str((pinned.get("edit_target") or {}).get("entry_function") or "")
    path = Path(a.file)
    path.write_text(
        expose(
            path.read_text(encoding="utf-8"),
            entry,
            call_site["fragment"],
            call_site["call"],
            baseline_definition(str(definition.get("baseline") or ""), name),
        ),
        encoding="utf-8",
    )
    print(f"exposed {name} in {path.name}; {entry} now calls it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
