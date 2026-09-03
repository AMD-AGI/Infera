#!/usr/bin/env python3
"""`check_optimization_shape` — is this a real campaign, or a description of one?

The cheap half of m4's output gate. It runs before `check_speedup_substantiated`
(`cost: seconds` against `cost: gpu_hours`, and a phase is ordered cheap-first),
so a handoff whose document does not parse fails in a second instead of after a
measurement that had nothing to measure.

Three jobs, in cost order:

1. **The document validates** against `assets/schemas/kernel_optimization.json`
   — the same file the producer was handed (mission G2, *该 schema 同时暴露给
   producer & validator*). Most of what a shape check used to hand-roll is a
   schema problem now, and the parts that stayed are the parts a schema cannot
   express.
2. **The document agrees with the workset it says it came from.** The workset
   travels inside the handoff as `results/workset.snapshot.yaml`, and every
   premise field, entrypoint, protocol figure and integration point in the
   document must be the snapshot's. This is where M5.1.1 is enforced: an `apply`
   written against a different file than the workset declared is caught here,
   before m5 tries to be a program about it.
3. **The packup is a packup** — the four documents a cold reader needs, the
   apparatus `check_speedup_substantiated` will re-measure from, and no
   placeholder text.

**What this cannot catch, stated so nobody assumes otherwise.**

It does not run anything, so no number here is checked for truth: a document
claiming `mean_case_speedup: 99.0` passes, and substantiating it is the next
validator's job.

And it cannot prove the snapshot is a faithful copy of the real workset. A
validator on an output phase is handed only the handoffs it declared in
`inputs`, over `layout.stage(task.outputs, …)`; declaring `operator_workset`
instead would bind this body to the *workset's* phases and fail an innocent
producer, which is measured — `kernel-opt-demo/assets/
check_speedup_substantiated.validator/check.py` carries the account. So the
comparison against the real workset is **opportunistic**: `_cross_check` fires in
any phase that happened to stage both, which m5's input validation does because
it stages every one of m5's inputs. When it does not fire it says so in a note.
A check that silently passes when it did not run is worse than one that is
honestly absent.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

try:
    import schema as schema_lib  # noqa: E402
except Exception:  # pragma: no cover - reported as a problem, never as a pass
    schema_lib = None

_FENCE = re.compile(r"^\s*```")
_PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "to be filled in")

#: Where the structured document lives inside the packup. Fixed rather than
#: discovered: two validators, the producer and m5 all open it, and a path each
#: of them derives is a path they can each derive differently.
_DOC = "results/kernel_optimization.json"
_SNAPSHOT = "results/workset.snapshot.yaml"

# **There is deliberately no `<...>` template-slot rule, and its absence was
# measured rather than assumed.** The sibling package's shape check has one, so
# this body had one too; run against a real, complete, honest packup it fired
# twice — on `<workset>` in a REPRODUCE.md command and on `<project_root>` in a
# sentence describing another tool's default path. Both are documentation
# metavariables and both are *correct writing*. A regex cannot separate "a slot
# the author forgot to fill" from "a metavariable the author meant"; given that,
# the choice is which error to make, and for a `strong` validator whose PASS is
# unqualified a false failure is the worse one — it teaches an author to write
# vaguer documentation to get past the gate.


def _lines(path: Path) -> tuple[int, int]:
    """(content lines, command lines inside fenced blocks)."""
    content = commands = 0
    fenced = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _FENCE.match(raw):
            fenced = not fenced
            continue
        line = raw.strip()
        if not line:
            continue
        if fenced:
            if not line.startswith("#"):  # a comment in a fence is documentation
                commands += 1
            continue
        if line.startswith("#"):
            continue
        content += 1
    return content, commands


def _load_yaml(path: Path):
    import yaml  # a declared agent_sys dependency

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _operator_of(snapshot: dict, operator_id: str) -> dict | None:
    for entry in snapshot.get("operators") or ():
        if isinstance(entry, dict) and entry.get("operator_id") == operator_id:
            return entry
    return None


def _same(label: str, expected, actual, problems: list[str]) -> None:
    if expected != actual:
        problems.append(f"{label}: workset says {expected!r}, the handoff says {actual!r}")


def _check_against_snapshot(doc: dict, snapshot: dict, problems: list[str]) -> None:
    """Every field the document copied from the workset must still be the workset's.

    This is the half of the gate a schema cannot do. A schema can say `apply`
    exists and is shaped right; only a comparison can say it points at the file
    the workset declared rather than at the file the optimiser happened to open.
    """
    operator_id = doc.get("operator")
    operator = _operator_of(snapshot, str(operator_id))
    if operator is None:
        have = [e.get("operator_id") for e in snapshot.get("operators") or () if isinstance(e, dict)]
        problems.append(f"operator {operator_id!r} is not in the workset (has: {have})")
        return

    ground = snapshot.get("ground_truth") or {}
    premise = doc.get("premise") or {}
    _same("premise.abort_on_mismatch", ground.get("abort_on_mismatch"), premise.get("abort_on_mismatch"), problems)
    _same("premise.warn_on_mismatch", ground.get("warn_on_mismatch"), premise.get("warn_on_mismatch"), problems)
    _same("premise.workset_environment", ground.get("environment"), premise.get("workset_environment"), problems)

    # M5.1.1 — the integration point is the workset's, not the optimiser's.
    declared = operator.get("edit_target") or {}
    point = (doc.get("apply") or {}).get("integration_point") or {}
    for field in ("source_file", "entry_function"):
        _same(f"apply.integration_point.{field}", declared.get(field), point.get(field), problems)

    performance = (doc.get("evidence") or {}).get("performance") or {}
    _same("evidence.performance.protocol", snapshot.get("protocol"), performance.get("protocol"), problems)

    # The entrypoints must be the workset's own, verbatim. A producer that
    # re-implements the correctness suite has measured a different thing, and
    # the difference is invisible in a report.
    entrypoints = operator.get("entrypoints") or snapshot.get("entrypoints") or {}
    correctness = (doc.get("evidence") or {}).get("correctness") or {}
    _same(
        "evidence.correctness.entrypoint",
        (entrypoints.get("correctness") or {}).get("cmd"),
        correctness.get("entrypoint"),
        problems,
    )
    _same(
        "evidence.performance.entrypoint",
        (entrypoints.get("performance") or {}).get("cmd"),
        performance.get("entrypoint"),
        problems,
    )

    # Every case the workset declares for performance must have been measured.
    declared_cases = {
        s.get("case_id")
        for s in operator.get("shapes") or ()
        if isinstance(s, dict) and s.get("role") in ("performance", "correctness-and-performance")
    }
    measured = set((performance.get("measured") or {}).get("per_case_ms") or {})
    missing = sorted(c for c in declared_cases if c and c not in measured)
    if missing:
        problems.append(f"the workset declares performance shapes that were never measured: {missing}")


def _check_arithmetic(doc: dict, problems: list[str]) -> None:
    """The ratios must follow from the two tables they were computed from.

    Recomputed rather than trusted, for the reason `workset_io.weighted_mean`
    exists: a stored figure that does not follow from the raw numbers cannot
    come from a different formula, so it can only come from the record having
    been edited after it was measured — which is exactly the finding worth
    making.
    """
    performance = (doc.get("evidence") or {}).get("performance") or {}
    claim = performance.get("claim")
    if not claim:
        return  # no claim is a legitimate outcome; the schema decides when

    baseline = (performance.get("baseline") or {}).get("per_case_ms") or {}
    measured = (performance.get("measured") or {}).get("per_case_ms") or {}
    stated = claim.get("speedup_per_case") or {}

    for case, ratio in stated.items():
        if case not in baseline or case not in measured:
            problems.append(f"claim.speedup_per_case names {case!r}, which is not in both tables")
            continue
        expected = baseline[case] / measured[case]
        if abs(expected - float(ratio)) > 0.005 * expected:
            problems.append(
                f"claim.speedup_per_case[{case}] is {ratio}, but "
                f"{baseline[case]}/{measured[case]} is {expected:.4f}"
            )
    if stated:
        expected_mean = sum(float(v) for v in stated.values()) / len(stated)
        got = float(claim.get("mean_case_speedup", 0.0))
        if abs(expected_mean - got) > 0.005 * max(expected_mean, 1e-9):
            problems.append(
                f"claim.mean_case_speedup is {got}, but the mean of speedup_per_case is {expected_mean:.4f}"
            )

    floor = float(claim.get("noise_floor", 0.0))
    mean = float(claim.get("mean_case_speedup", 0.0))
    if floor and mean < floor:
        problems.append(
            f"a claim of {mean:.4f}x is below the workset's own noise floor {floor:.3f}x — "
            "not distinguishable from measurement spread, and reporting it as an improvement is a false claim"
        )


def _cross_check(doc: dict, snapshot: dict, notes: list[str], problems: list[str]) -> None:
    """If some other handoff in this phase *is* the workset, hold the snapshot to it.

    Opportunistic on purpose — see the module docstring. In m4's output phase
    nothing else is staged and this records that it did not run; in m5's input
    phase the real `operator_workset` is staged beside this handoff and the
    snapshot stops being taken on trust.
    """
    ref = doc.get("workset_ref") or {}
    for hid, staged in zone.materials().items():
        candidate = Path(staged) / "items" / "codes" / "workset.yaml"
        if not candidate.is_file():
            continue
        try:
            real = _load_yaml(candidate)
        except Exception as exc:  # a workset that does not parse is m3's failure, not m4's
            notes.append(f"{hid}: workset.yaml did not parse ({exc}); snapshot not cross-checked")
            continue
        if real == snapshot:
            notes.append(f"snapshot cross-checked against the real workset staged as {hid}")
        else:
            problems.append(
                f"{_SNAPSHOT} is not the workset staged as {hid} — the premise and the baseline in "
                "this handoff were taken from a document that is not the one m3 published"
            )
        if ref.get("workset_id") and real.get("workset_id") != ref.get("workset_id"):
            problems.append(
                f"workset_ref.workset_id is {ref.get('workset_id')!r}, the staged workset is "
                f"{real.get('workset_id')!r}"
            )
        return
    notes.append(
        "no workset staged in this phase, so the snapshot is taken on trust here; "
        "it is cross-checked in m5's input validation, which stages both"
    )


def _check(content: Path, args: dict, problems: list[str], notes: list[str]) -> bool:
    packup, why = zone.find_packup(content)
    if packup is None:
        problems.append(why)
        return False

    # --- 1. the document, against the schema both sides read -----------------
    doc_path = packup / _DOC
    doc: dict | None = None
    if not doc_path.is_file():
        problems.append(f"missing {_DOC}; there is nothing structured to consume")
    elif schema_lib is None:
        problems.append("assets/lib/schema.py could not be imported; the document cannot be validated")
    else:
        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{_DOC} does not parse: {exc}")
        else:
            try:
                schema_lib.validate(str(args.get("schema") or "kernel_optimization"), doc)
            except schema_lib.SchemaError as exc:
                problems.extend(str(exc).splitlines())
                doc = None

    # --- 2. the snapshot, and the document's agreement with it ---------------
    snapshot_path = packup / _SNAPSHOT
    snapshot: dict | None = None
    if not snapshot_path.is_file():
        problems.append(f"missing {_SNAPSHOT}; the workset this claims to come from is not carried")
    else:
        try:
            snapshot = _load_yaml(snapshot_path)
        except Exception as exc:
            problems.append(f"{_SNAPSHOT} does not parse: {exc}")
        else:
            if schema_lib is not None:
                try:
                    schema_lib.validate("workset", snapshot)
                except schema_lib.SchemaError as exc:
                    problems.append("the carried workset snapshot is not a valid workset:")
                    problems.extend(str(exc).splitlines()[1:])
                    snapshot = None

    if doc is not None and snapshot is not None:
        _check_against_snapshot(doc, snapshot, problems)
        _cross_check(doc, snapshot, notes, problems)
    if doc is not None:
        _check_arithmetic(doc, problems)

    # --- 3. the packup a cold reader needs -----------------------------------
    floors: dict = args.get("min_content_lines") or {}
    for name, floor in floors.items():
        target = packup / name
        if not target.is_file():
            problems.append(f"missing {name}")
            continue
        content_lines, command_lines = _lines(target)
        if content_lines < int(floor):
            problems.append(f"{name} has {content_lines} content lines, needs >= {floor}")
        if name == "REPRODUCE.md":
            need = int(args.get("min_command_lines") or 0)
            if command_lines < need:
                problems.append(f"REPRODUCE.md has {command_lines} command lines, needs >= {need}")
        text = target.read_text(encoding="utf-8", errors="replace")
        for marker in _PLACEHOLDERS:
            if marker in text:
                problems.append(f"{name} still carries a {marker} placeholder")

    for name in ("scripts", "results"):
        directory = packup / name
        if not directory.is_dir():
            problems.append(f"missing {name}/")
        elif not any(p.is_file() for p in directory.rglob("*")):
            problems.append(f"{name}/ holds no files")

    readme = packup / "README.md"
    if readme.is_file():
        head = readme.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"^##\s+Result\b", head, re.M):
            problems.append("README.md has no `## Result` section")
        # A run that could be read as a complete campaign must be impossible to
        # read as one. The schema already forbids a mock or a degraded run from
        # carrying a claim; this is the half a schema cannot reach, because it
        # is about what the prose says.
        forge = ((doc or {}).get("evidence") or {}).get("forge") or {}
        if forge.get("mock") and "MOCK" not in head.upper():
            problems.append(
                "the document says forge.mock but README.md never says MOCK — "
                "a mock that is not visibly a mock reads as a success"
            )
        if forge.get("degraded") and "SMOKE" not in head.upper():
            problems.append(
                "the document says forge.degraded but README.md never says SMOKE — "
                "a degraded budget produces a campaign that reads like a full one"
            )

    environment = packup / "environment.md"
    if environment.is_file() and not re.search(r"\d", environment.read_text(encoding="utf-8", errors="replace")):
        problems.append("environment.md carries no numbers at all — nothing is pinned")

    # The apparatus the expensive validator re-measures from, and the files the
    # `apply` block names. A kit that reports a speedup and does not carry the
    # thing that measured it cannot be checked by anyone who does not already
    # have the workset, which is most readers.
    for rel in args.get("required_evidence") or []:
        target = packup / rel
        if not target.is_file():
            problems.append(f"missing evidence {rel}")
        elif target.stat().st_size == 0:
            problems.append(f"evidence {rel} is empty")

    for entry in ((doc or {}).get("apply") or {}).get("files") or ():
        source = packup / str(entry.get("source"))
        if not source.is_file():
            problems.append(f"apply names {entry.get('source')!r}, which is not in the packup")

    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        notes: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            # Staged nothing is *no content*, and it is never a pass.
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems, notes)
        for note in notes:
            print(f"{hid} note: {note}")
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_optimization_shape: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
