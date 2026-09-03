#!/usr/bin/env python3
"""`check_environment` — mission G5, on all fifteen kinds.

> 整个流程的 handoff 都需要传递 env

One document, one schema, three spellings of where it lives (CONTRACT.md §2).
This body **dispatches on what it finds** rather than being told the content
type, so a kind whose type changes does not silently stop being checked.

Three things it decides, in cost order:

1. **Present.** An `environment.yaml` exists somewhere this contract puts it.
2. **Valid.** It validates against `assets/schemas/environment.schema.json` —
   the same file the producer validated against, which is the whole of G2 — and
   carries the fields a reproduction fails without.
3. **Consistent.** Every handoff in this phase describes the *same machine and
   image*.

### What (3) deliberately does not check, and why

**`runtime.container` is reported, never failed on.** Module 5's two arms have
different containers by construction — a container holds one state for its life,
which is the entire reason the two-arm design exists, and CONTRACT.md §5 grants
it. A `strong` validator that failed on that would stop the graph on the one
stage the contract says must differ.

So `fixed` is compared strictly and `runtime` is only described. That is weaker
than "modules 1–4 shared one container", which is what I wanted to assert and
cannot from here: this body sees handoff **ids**, not kinds
(`zone.materials()` is keyed by id), so it cannot tell an m5 arm from an m2 line.
Stated rather than approximated — an assertion that is right four times out of
five and stops the graph the fifth is worse than one that reports.
"""
from __future__ import annotations

import os
import pathlib
import sys

_PKG = pathlib.Path(os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE", ""))
sys.path.insert(0, str(_PKG / "assets" / "lib"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402
import zone  # noqa: E402

#: Where a record may live, in the order the contract lists them. The glob is
#: last because a `code` handoff wraps its content in one named directory (a
#: packup, or a workset named after its operator), and only that shape needs it.
CANDIDATES = ("items/env/environment.yaml", "items/codes/environment.yaml", "items/codes/*/environment.yaml")


def find_record(content: pathlib.Path) -> tuple[pathlib.Path | None, str]:
    for rel in CANDIDATES:
        if "*" in rel:
            found = sorted(content.glob(rel))
            if len(found) == 1:
                return found[0], rel
            if len(found) > 1:
                return None, f"more than one environment.yaml under items/codes/*: {[str(p) for p in found]}"
        else:
            path = content / rel
            if path.is_file():
                return path, rel
    return None, f"no environment.yaml at any of {list(CANDIDATES)}"


def dotted(doc: dict, section: str, field: str):
    return (doc.get(section) or {}).get(field)


def main() -> int:
    args = zone.args()
    required_fixed = list(args.get("require_fixed", []))
    required_runtime = list(args.get("require_runtime", []))
    compare_fixed = list(args.get("compare_fixed_across_inputs", []))

    verdict: dict[str, bool] = {}
    findings: list[str] = []
    # hid -> the `fixed` fields we compare across the phase.
    seen: dict[str, dict] = {}

    for hid in zone.inputs():
        content = zone.content_of(hid)
        if content is None:
            findings.append(f"{hid}: nothing staged — treated as no content, never as a pass")
            verdict[hid] = False
            continue

        path, how = find_record(content)
        if path is None:
            findings.append(f"{hid}: {how}")
            verdict[hid] = False
            continue

        problems: list[str] = []
        try:
            import yaml

            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — the message is the finding
            findings.append(f"{hid}: {how} is unreadable: {type(exc).__name__}: {exc}")
            verdict[hid] = False
            continue

        if not isinstance(doc, dict):
            findings.append(f"{hid}: {how} is not a mapping")
            verdict[hid] = False
            continue

        # (2) valid — against the package's own schema, the same file the
        # producer used. `items_schema` does not do this job: for a file item it
        # validates the filename string and never reads the contents.
        try:
            schema_lib.validate(args.get("schema", "environment"), doc)
        except schema_lib.SchemaError as exc:
            problems.append(str(exc))

        for field in required_fixed:
            if dotted(doc, "fixed", field) in (None, ""):
                problems.append(f"fixed.{field} is missing or empty")
        for field in required_runtime:
            if dotted(doc, "runtime", field) in (None, ""):
                problems.append(f"runtime.{field} is missing or empty")

        seen[hid] = {k: dotted(doc, "fixed", k) for k in compare_fixed}
        container = dotted(doc, "runtime", "container")
        if container:
            findings.append(f"{hid}: runtime.container={container} (reported, not judged — see module 5's two arms)")

        verdict[hid] = not problems
        for line in problems:
            findings.append(f"{hid}: {line}")

    # (3) consistent — `fixed` only. A disagreement here means two handoffs in
    # one phase describe two different machines, and every number computed
    # across them is void.
    if len(seen) > 1 and compare_fixed:
        base_hid, base = next(iter(seen.items()))
        for hid, other in list(seen.items())[1:]:
            for field in compare_fixed:
                if base.get(field) != other.get(field):
                    findings.append(
                        f"{hid}: fixed.{field}={other.get(field)!r} disagrees with "
                        f"{base_hid}'s {base.get(field)!r} — these are two different machines"
                    )
                    verdict[hid] = False

    for line in findings:
        print(line, file=sys.stderr)
    zone.write_verdict(verdict)
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
