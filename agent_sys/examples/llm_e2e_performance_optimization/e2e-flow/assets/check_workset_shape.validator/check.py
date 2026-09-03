#!/usr/bin/env python3
"""`check_workset_shape` — completeness, strong.

The workset validates against the merged schema and carries at least three
shapes with runnable correctness and performance entrypoints.

**Shape, not quality — and the split is deliberate.** Whether the entrypoints
actually run, and whether their numbers are true, is `check_workset_runs`'s
question and costs GPU hours. This one costs seconds and runs first, so a
workset missing a file fails before a card is booked for it.

Every rule below is decided by opening a file that either is there or is not, or
by comparing two documents that either agree or do not. Nothing is judged. That
is why `strong` needs no qualification here: it cannot be *approximately* right
about whether `run_performance.sh` exists.

The schema (`assets/schemas/workset.schema.json`) carries the field-level half
and the producer validated against it too. What is left here is what a JSON
Schema cannot state, and on this kind it is most of the value:

1. **Every path the document names exists, is non-empty, and — for an
   entrypoint — is executable.** A `workset.yaml` naming a `run_forge.sh` that
   was never written is the single most likely way this artefact is wrong, and
   the schema can only check that the *string* looks like a path.
2. **`shapes` corresponds to `workload`, line for line.** The JSONL is the
   source of truth for shapes and `shapes` is an index of it; an index that has
   drifted means the `--shape CASE_ID` selector selects something other than
   what a reader reading `workset.yaml` expects.
3. **The Definition is a flashinfer-bench Definition.** `reference` and
   `baseline` are Python source strings, so they are parsed, and both must
   define `run`. A Definition whose reference does not parse is a correctness
   gate that will fail at the first call with a SyntaxError hours later.
4. **`reference` and `baseline` are not the same function.** Conflating them is
   the single most common way a speedup number becomes meaningless, and it is
   invisible in a document where both fields are merely present.
5. **The KernelForge add-on agrees with the base it was generated from**
   (M3.7.6). Same argument `analyze-demo` made for `invocation_spec.json`
   against `forge_task.yaml`: two files produced from one record, so a
   disagreement means one was edited on its own.
6. **No hard-coded host path in executable or generated content.** `analyze-demo`
   justifies this rule by saying the seal refuses a delivery over one. Measured
   against the framework rather than inherited: it does not.
   `handoff/store.py:447,494` decline to call `locality.check` — user-ruled
   2026-08-31 at a measured 97% false-positive rate — and the sealed
   `deploy_kit` in the mock set carries `/shared_nfs/...` in five files. The rule
   survives on its own merit, which was always the real one: a script carrying
   one host's directory does not run on the next host. It skips the environment
   record, whose absolute paths are schema-required and are the point of it.
7. **No template markers.** A workset that still says TODO is not a workset.

What it cannot catch, stated so nobody assumes otherwise: it does not run
anything. An entrypoint that is present, executable and measures the wrong
operator passes here.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as S  # noqa: E402
import workset_io as W  # noqa: E402
import zone  # noqa: E402

_PLACEHOLDERS = ("TODO", "TBD", "FIXME", "XXX", "to be filled in", "<fill", "REPLACE_ME")

#: flashinfer-bench's Definition keys, as `rank0/definitions/` carries them.
_DEFINITION_KEYS = ("name", "op_type", "axes", "inputs", "outputs", "reference", "baseline")


def _exists(root: Path, relative: str, label: str, problems: list[str], *, executable: bool = False) -> bool:
    target = root / relative
    if not target.is_file():
        problems.append(f"{label}: {relative} is absent")
        return False
    if target.stat().st_size == 0:
        problems.append(f"{label}: {relative} is empty")
        return False
    if executable and not os.access(target, os.X_OK):
        # `agent/gate.py:EXECUTABLE_ITEMS` refuses a seal for a non-executable
        # `script` item *after* the body has returned, with a follow-up message
        # that does not name the missing bit — an AI task once looped to its
        # silent timeout over exactly this. Naming it here costs nothing.
        problems.append(f"{label}: {relative} is not executable; run `chmod +x` on it")
        return False
    return True


def _check_entrypoints(root: Path, block: dict, label: str, required: list, problems: list[str]) -> None:
    for wanted in required:
        if wanted not in block:
            problems.append(f"{label}: no {wanted} entrypoint")
    for kind, entry in block.items():
        # The command may carry arguments; the script is the first word.
        script = entry["cmd"].split()[0]
        _exists(root, script, f"{label}.{kind}", problems, executable=True)


def _check_definition(root: Path, operator: dict, problems: list[str]) -> None:
    label = operator["operator_id"]
    if not _exists(root, operator["definition"], label, problems):
        return
    try:
        definition = W.load_definition(root.parent.parent, operator["definition"])
    except (json.JSONDecodeError, OSError) as error:
        problems.append(f"{label}: {operator['definition']} does not load: {error}")
        return

    for key in _DEFINITION_KEYS:
        if key not in definition:
            problems.append(f"{label}: the Definition has no {key!r}; flashinfer-bench requires it")
    if definition.get("op_type") != operator["op_type"]:
        problems.append(
            f"{label}: workset.yaml says op_type {operator['op_type']!r}, "
            f"the Definition says {definition.get('op_type')!r}"
        )

    sources = {}
    for key in ("reference", "baseline"):
        source = definition.get(key)
        if not isinstance(source, str) or not source.strip():
            problems.append(f"{label}: the Definition's {key!r} is not a source string")
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            problems.append(f"{label}: the Definition's {key!r} does not parse: line {error.lineno}: {error.msg}")
            continue
        if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body):
            problems.append(f"{label}: the Definition's {key!r} defines no top-level `run`; that is the entry point")
        sources[key] = source

    # Rule 4. A reference and a baseline that are the same function mean the
    # workset measures a speedup against the thing being checked for
    # correctness, which is a ratio of one.
    if len(sources) == 2 and sources["reference"].strip() == sources["baseline"].strip():
        problems.append(
            f"{label}: the Definition's reference and baseline are the same source. "
            f"reference is what correctness is judged against; baseline is the incumbent "
            f"fast implementation a speedup is judged against. Identical means the speedup is 1.0 by construction"
        )


def _check_shapes(content: Path, root: Path, operator: dict, args: dict, problems: list[str]) -> None:
    label = operator["operator_id"]
    shapes = operator["shapes"]
    floor = W.arg_num(args, "min_shapes", 3, int)
    if len(shapes) < floor:
        problems.append(f"{label}: {len(shapes)} shape(s), M3.7.4.1.2 requires {floor}")

    if not _exists(root, operator["workload"], label, problems):
        return
    try:
        lines = W.load_workload(content, operator["workload"])
    except (json.JSONDecodeError, OSError) as error:
        problems.append(f"{label}: {operator['workload']} does not load: {error}")
        return

    if len(lines) != len(shapes):
        problems.append(
            f"{label}: the workload has {len(lines)} line(s), workset.yaml indexes {len(shapes)} shape(s). "
            f"The JSONL is the source of truth; shapes is its index"
        )
    for i, (line, shape) in enumerate(zip(lines, shapes)):
        payload = line.get("workload") or {}
        if payload.get("uuid") != shape["uuid"]:
            problems.append(
                f"{label}: workload line {i} has uuid {payload.get('uuid')!r}, "
                f"shapes[{i}] ({shape['case_id']}) says {shape['uuid']!r}"
            )
        if payload.get("axes") != shape["axes"]:
            problems.append(f"{label}: workload line {i} axes {payload.get('axes')} != shapes[{i}].axes {shape['axes']}")
        if line.get("definition") != Path(operator["definition"]).stem:
            problems.append(
                f"{label}: workload line {i} names definition {line.get('definition')!r}, "
                f"the operator's definition file is {Path(operator['definition']).stem!r}"
            )
        for slot in ("solution", "evaluation"):
            if line.get(slot) is not None:
                problems.append(
                    f"{label}: workload line {i} pre-fills {slot!r}. Those slots are the consumer's; "
                    f"a workset that fills them asserts an answer it has not measured"
                )

    primaries = [s["case_id"] for s in shapes if s.get("is_primary")]
    if len(primaries) != 1:
        problems.append(f"{label}: {len(primaries)} primary shape(s) {primaries}, expected exactly 1")
    if not any("performance" in s["role"] for s in shapes):
        problems.append(f"{label}: no shape carries a performance role; nothing here can be timed")

    ids = [s["case_id"] for s in shapes]
    if len(set(ids)) != len(ids):
        problems.append(f"{label}: duplicate case_id(s) {sorted({i for i in ids if ids.count(i) > 1})}")


def _check_forge(root: Path, operator: dict, problems: list[str]) -> None:
    """The KernelForge add-on, M3.7.6. Optional as a block; internally consistent
    when present. It is *layered over* the base — a consumer that does not use
    KernelForge ignores it and loses nothing — so its absence is not a fault and
    its disagreement with the base is."""
    forge = operator.get("forge")
    if not forge:
        return
    label = operator["operator_id"]
    _exists(root, forge["one_line"], label, problems, executable=True)
    _exists(root, forge["driver"], label, problems)

    driver = root / forge["driver"]
    if driver.is_file():
        source = driver.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(source)
        except SyntaxError as error:
            problems.append(f"{label}: {forge['driver']} does not parse: line {error.lineno}")
        # forge-loop reads correctness and timing off this file's stdout and
        # treats the file as protected. A driver that never prints `case_ms:`
        # cannot be benchmarked, and the failure surfaces hours later as a
        # campaign that ran and measured nothing. Four substring checks buy that
        # back for free.
        for token in ("SNR", "allclose", "--bench-mode", "CASE_ID"):
            if token not in source:
                problems.append(f"{label}: {forge['driver']} never mentions {token!r}; forge-loop's preflight rejects it")

    cases_rel = forge.get("cases")
    if cases_rel and _exists(root, cases_rel, label, problems):
        try:
            document = json.loads((root / cases_rel).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{label}: {cases_rel} does not load: {error}")
            return
        entries = document if isinstance(document, list) else (document.get("cases") or [])
        exported = [e.get("case_id") for e in entries]
        declared = [s["case_id"] for s in operator["shapes"]]
        if exported != declared:
            problems.append(
                f"{label}: {cases_rel} exports cases {exported} but the workload declares {declared}. "
                f"The export is generated from the base format; a disagreement means one of them was edited alone"
            )

    for key in ("invocation_spec", "task"):
        if forge.get(key):
            _exists(root, forge[key], label, problems)


def _check(content: Path, args: dict, problems: list[str]) -> bool:
    root = W.workset_root(content)
    if not root.is_dir():
        problems.append("items/codes is not a directory")
        return False

    try:
        document = W.load_workset(content)
    except Exception as error:  # noqa: BLE001 — yaml raises several unrelated types
        problems.append(f"items/codes/workset.yaml does not load: {error}")
        return False

    name = args.get("schema") or "workset"
    try:
        S.validate(name, document)
    except S.SchemaError as error:
        problems.extend(str(error).splitlines()[1:])
        return False

    # CONTRACT 2: a `code` kind puts the environment record here. `check_environment`
    # grades its contents; this only refuses a workset that dropped it.
    _exists(root, "environment.yaml", "workset", problems)

    _check_entrypoints(root, document["entrypoints"], "workset", args.get("require_entrypoints") or [], problems)

    for operator in document["operators"]:
        label = operator["operator_id"]
        _check_entrypoints(root, operator["entrypoints"], label, args.get("require_entrypoints") or [], problems)
        _check_definition(root, operator, problems)
        _check_shapes(content, root, operator, args, problems)
        _check_forge(root, operator, problems)
        if operator["reference"]["kind"] == "written":
            _exists(root, operator["reference"]["path"], f"{label}.reference", problems)

        # The apparatus is the set a consumer copies. A named file that is not
        # there produces a copy that cannot run, an hour later, on the consumer's
        # side — so it is checked here, where it costs nothing.
        for relative in operator["apparatus"]:
            _exists(root, relative, f"{label}.apparatus", problems)

        # M5.1.1: the integration point has to be usable by a program. An empty
        # `public_symbol` or a sentinel invariant means m5 is back to reading a
        # report and deciding, which is what declaring it was for.
        integration = operator["integration"]
        if not integration["target_files"]:
            problems.append(f"{label}: integration.target_files is empty; m5 has nothing to replace")
        if not integration["public_symbol"].strip():
            problems.append(
                f"{label}: integration.public_symbol is empty. The file may be rewritten wholesale, "
                f"so the symbol is what a replacement must still provide"
            )

    # The evidence is optional in the schema — a workset may be shape-checked
    # before it has been measured — and required here, because a workset that
    # reaches a consumer unmeasured is the state M4.3.5 was reversed against.
    evidence = document.get("evidence")
    if not evidence:
        problems.append(
            "no evidence block. m4 takes its ground truth strictly from this artefact, "
            "so an unmeasured workset has nothing for it to take"
        )
    else:
        for key, schema_def in (("correctness_report", "correctness_report"),
                                ("performance_report", "performance_report")):
            if _exists(root, evidence[key], "evidence", problems):
                _validate_report(root / evidence[key], schema_def, f"evidence.{key}", problems)
        # `noise_floor` is a **transcription of a computed figure**, so it is
        # checked against the figure rather than merely required to be present.
        # A workset declaring 1.01 on a host whose measured spread implies 1.09
        # is one that will call noise a win, and it is the consumer who pays.
        report_path = root / evidence["performance_report"]
        if report_path.is_file():
            try:
                measured = json.loads(report_path.read_text(encoding="utf-8")).get("noise_floor")
            except json.JSONDecodeError:
                measured = None
            if measured is not None:
                for operator in document["operators"]:
                    if operator["noise_floor"] < measured - 1e-9:
                        problems.append(
                            f"{operator['operator_id']}: noise_floor is {operator['noise_floor']}, but the "
                            f"measured spread in {evidence['performance_report']} gives {measured}. "
                            f"A floor below what the host's own noise supports calls noise a win"
                        )

    # Rules 6 and 7.
    #
    # **The premise `analyze-demo` gave for rule 6 is no longer true and the
    # rule is kept anyway, rescoped.** That validator says an absolute path
    # "refuses the whole delivery" at the seal. Measured against the framework
    # rather than inherited: `handoff/store.py:447,494` do not call
    # `locality.check` at all — user-ruled 2026-08-31 after the shape heuristic
    # read an HTTP access-log line as a filesystem path and refused a correct
    # artefact, at a measured 97% false-positive rate on a real kit. The sealed
    # `deploy_kit` in the mock set carries `/shared_nfs/...` in five files and
    # sealed cleanly, which is the same finding from the other side.
    #
    # So this is not a seal rule. It is a **portability rule**, and it applies
    # to what a reproducer executes: a script that hard-codes one host's
    # directory does not run on the next host, which is the entire reason the
    # workset carries `${AITER_ROOT}` instead of a path.
    #
    # It therefore skips the environment record. `environment.yaml` and
    # `workset.yaml`'s inlined `ground_truth.environment` carry `model_path`,
    # which is absolute **by schema requirement** — recording where the weights
    # were is the document's job. Scanning them would reject every conforming
    # workset, which is how a rule this shape gets deleted rather than fixed.
    offenders: list[str] = []
    for path in sorted(p for p in W.workset_root(content).rglob("*") if p.is_file()):
        if path.suffix in (".py", ".sh", ".json", ".jsonl"):
            offenders.extend(
                f"{path.relative_to(content)}: {hit.split(': ', 1)[-1]}" for hit in W.absolute_paths_in(path)
            )
    if offenders:
        problems.append(
            f"{len(offenders)} hard-coded absolute path(s) in executable or generated content, "
            f"first: {offenders[0]}. A reproducer on another host cannot run it; use a "
            f"${{PLACEHOLDER}} resolved from the environment"
        )

    # **Only content this workset declares as its own.**
    #
    # Scanning every `.md` under `content/` produced a false positive of the
    # family that got `locality.check` disconnected: the sealed stage-3 operator
    # directories, carried along for their ranking provenance, contain the
    # sentence *"A `TODO:` line here is a gap that is stated rather than papered
    # over"* — prose **about** the marker, in an artefact this validator has no
    # business grading the wording of.
    #
    # So the scan follows the manifest: `workset.yaml`, the Definitions and
    # Workloads, the generated forge add-on, and whatever `apparatus` names.
    # That is exactly the set "a workset that is still a template" is about, and
    # it still catches the one case that matters — the `integration.invariants`
    # sentinel, because `workset.yaml` is always in `apparatus`.
    declared: set[Path] = {root / "workset.yaml"}
    for operator in document["operators"]:
        declared.add(root / operator["definition"])
        declared.add(root / operator["workload"])
        declared.update(root / rel for rel in operator["apparatus"])
        if operator["reference"]["kind"] == "written":
            declared.add(root / operator["reference"]["path"])
        for key, value in (operator.get("forge") or {}).items():
            if isinstance(value, str) and "/" in value:
                declared.add(root / value)
    for path in sorted(declared):
        if not path.is_file() or path.suffix not in (".md", ".yaml", ".yml", ".json"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in _PLACEHOLDERS:
            if marker in text:
                problems.append(f"{path.relative_to(content)} still carries a {marker} placeholder")

    return not problems


def _validate_report(path: Path, definition: str, label: str, problems: list[str]) -> None:
    """One `evidence/*.json` against `workset.schema.json#/$defs/<definition>`.

    Resolved through `schema.py`'s registry rather than by loading the file
    directly, so the `$ref` to `environment.schema.json` inside it still works.
    """
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    root = S.schema_path("workset").parent
    registry = Registry().with_resources(
        (p.name, Resource.from_contents(json.loads(p.read_text()), default_specification=DRAFT202012))
        for p in sorted(root.glob("*.schema.json"))
    )
    validator = Draft202012Validator(
        {"$ref": f"workset.schema.json#/$defs/{definition}"}, registry=registry
    )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"{label}: does not load: {error}")
        return
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path)):
        where = "$." + ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "$"
        problems.append(f"{label}: {where}: {error.message}")


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems)
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_workset_shape: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
