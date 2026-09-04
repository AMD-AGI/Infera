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


def deep(doc: dict, path: str):
    """`fixed.gpu_devices` -> the value, or None if any hop is missing."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def check_invariants(doc: dict, rules: list) -> list[str]:
    """Relations BETWEEN two fields of the environment record.

    Lifted from m1's `check_deploy_kit.check_invariant` (`53bc783`) rather than
    rewritten, keeping their `count_of` / `at_most` / `on_absent` spelling so the
    two read alike — theirs governs the kit's copy, this one governs the same
    record in the other fourteen kinds.

    **Why it is here and not in the schema.** JSON Schema relates a value to a
    *constant*: `maxItems` takes a literal, and bounding an array by a sibling
    property's value needs `$data`, an Ajv extension absent from draft 2020-12,
    while `spec_loader/validate.py` runs a stock `Draft202012Validator`. m1
    established this. So `len(fixed.gpu_devices) <= fixed.gpu_count` was not
    unexpressed because nobody looked — it was looked for, found inexpressible
    there, and expressed in the layout instead.

    **Why the layout was not enough.** The same `environment.yaml` travels in all
    fifteen kinds (CONTRACT section 2), and the layout governs `deploy_kit` only.
    A `profiling_evidence` could carry `gpu_count: 4` beside eight devices and
    validate cleanly. m4 found this by looking in the wrong place and being right
    about fourteen kinds for a reason they had not found; m1 found the reason.

    The reading that makes it worth fixing rather than noting: **a rule enforced
    at one carrier of a shared document is indistinguishable, from the artefact,
    from a rule enforced everywhere** — and this one has a name, a fault number
    and a gate, so a reader who finds it concludes the document is checked.

    Deliberately not an expression language, m1's reasoning and it holds here: a
    second relation adds a key, not a parser.
    """
    out: list[str] = []
    for rule in rules or []:
        name = rule.get("name", "unnamed")
        values = deep(doc, rule["count_of"])
        limit = deep(doc, rule["at_most"])
        if values is None or limit is None:
            # Absent is NOT a fault here. `gpu_devices` is optional in the
            # schema on purpose: a record written before the criterion existed
            # is still valid, and omitting it honestly says "this run did not
            # record which devices it took", where `[]` would falsely claim it
            # took none. m1 owns the decision to tighten it; this validator
            # grades fourteen kinds and must not tighten it for them.
            if rule.get("on_absent", "skip") == "fault":
                out.append(f"{name}: {rule['count_of']} or {rule['at_most']} is absent")
            continue
        # `isinstance(list)`, not `try: len()`. A `try/except TypeError` guard
        # was the first version and it let a STRING through: `gpu_devices: "0,1"`
        # has `len` 3, which is under any plausible `gpu_count`, so a malformed
        # record would have passed while being counted by characters. Found by
        # running the case rather than by reading — the arm reported `pass` where
        # the battery said it must refuse.
        if not isinstance(values, list):
            out.append(
                f"{name}: {rule['count_of']} is {type(values).__name__} {values!r}, not a list. "
                f"A string here would be counted by characters and pass."
            )
            continue
        n = len(values)
        if not isinstance(limit, int) or isinstance(limit, bool):
            out.append(f"{name}: {rule['at_most']} is {limit!r}, not an integer")
            continue
        if n > limit:
            out.append(
                f"{name}: {rule['count_of']} has {n} entr{'y' if n == 1 else 'ies'} "
                f"{values!r} but {rule['at_most']} is {limit} — the record claims to "
                f"have taken more cards than it says the node has"
            )
    return out


def main() -> int:
    args = zone.args()
    invariants = list(args.get("invariants", []))
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

        problems.extend(check_invariants(doc, invariants))

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
