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

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable
import workset_io as W  # noqa: E402 — m3's shared helpers; `write_report` is the reason

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


def _load_json(path: Path, problems: list[str]) -> dict | None:
    """A JSON object, or `None` with the reason recorded. Never raises.

    A validator body that dies on a malformed input reports nothing at all —
    `PhaseRunner` sees a non-zero exit and no `verdict.json`, which is a
    different and much less useful failure than "this file does not parse".
    """
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{path.name} does not parse: {exc}")
        return None
    if not isinstance(loaded, dict):
        problems.append(f"{path.name} is a {type(loaded).__name__}, expected an object")
        return None
    return loaded


def _operator_of(snapshot: dict, operator_id: str) -> dict | None:
    for entry in snapshot.get("operators") or ():
        if isinstance(entry, dict) and entry.get("operator_id") == operator_id:
            return entry
    return None


def _same(label: str, expected, actual, problems: list[str]) -> None:
    if expected != actual:
        problems.append(f"{label}: workset says {expected!r}, the handoff says {actual!r}")


def _same_file(expected: str, actual: str) -> bool:
    """Do two paths name one file, allowing for the two frames in play?

    The workset writes repo-relative and the container frame is root-relative,
    one level deeper, so the shared part is a suffix of one and the whole of the
    other. `_check_apply` has always compared `container_path` this way; this is
    that rule, extracted so the two sites cannot drift — which is the fault
    being fixed here, not a new tolerance.

    Anchored on a segment boundary. A bare `endswith` would call
    `layers/my_sampler.py` the same file as `layers/sampler.py`, which is the
    obvious way to turn a strict check into a useless one.
    """
    left, right = str(expected or "").strip("/"), str(actual or "").strip("/")
    if not left or not right:
        return left == right
    if left == right:
        return True
    longer, shorter = (left, right) if len(left) >= len(right) else (right, left)
    return longer.endswith("/" + shorter)


#: **Written and deliberately switched off.** Flip to `True` to arm the check
#: below. See `_substitution_matches_apply_mode` for what it refuses and why it
#: is not armed yet.
_ENFORCE_SUBSTITUTION_PAIR = False


def _substitution_matches_apply_mode(packup, operator: dict, doc: dict,
                                     problems: list[str], notes: list[str]) -> None:
    """Refuse an `apply` block the workset's own two fields make impossible.

    **`substitution` and `apply_mode` are a pair with illegal combinations, and
    nothing anywhere knows they are a pair.** Each validates fine alone — both
    are strings from their enum — so a schema that checks them separately cannot
    see it, and m4 emits the impossible combination in silence.

    The case, measured on rung 0's own workset, 2026-09-04::

        substitution:   call_site_fragment      # the edit is INSIDE Sampler.forward
        apply_mode:     overlay_files           # replace the whole file
        public_symbol:  null                    # there is no module symbol to swap
        module_symbols: 9   (Sampler, create_sampler, …)
        replacement defines: sampler_softmax, run     -> intersection: ZERO

    Overlaying `sampler.py` with a standalone kernel module deletes every symbol
    the engine imports. **The seed cannot be fixed to satisfy both readers**: a
    file that could replace `sampler.py` is not a file m3's harness can `exec`
    and call `run` on. That is M5.1.1 as a proof rather than a design question.

    **Why this is off.** m5's `check_patch_live` and six sibling validators have
    never seen an artefact the graph produced. Refusing here would stop the flow
    at stage 4 and keep it that way — buying an earlier message at the cost of
    the only chance to exercise them on something nobody chose. A gate that has
    never fired is not a gate, and that argument applies to *theirs* before it
    applies to mine. Leader's ruling, 2026-09-04: let rung 0 reach m5, then arm
    this so the failure names the operator's own declaration instead of arriving
    two stages downstream.

    **Written now and left inert on purpose**, because *"once they have spoken,
    the gate belongs at m4"* is the kind of intention that dies when the day
    ends. Written-and-disabled survives a handover; a note does not.

    **To arm it:** set `_ENFORCE_SUBSTITUTION_PAIR = True`. Nothing else.
    """
    if not _ENFORCE_SUBSTITUTION_PAIR:
        return
    integration = operator.get("integration") or {}
    substitution = str(integration.get("substitution") or "")
    apply_block = doc.get("apply") or {}
    apply_mode = str(apply_block.get("apply_mode") or integration.get("apply_mode") or "")
    if substitution != "call_site_fragment" or apply_mode != "overlay_files":
        return

    declared_symbols = [str(s) for s in (integration.get("module_symbols") or [])]
    if not declared_symbols:
        return

    # What the replacement actually defines, read rather than assumed.
    defined: set[str] = set()
    for entry in apply_block.get("files") or []:
        relative = str((entry or {}).get("replacement") or "")
        if not relative:
            continue
        candidate = Path(packup) / relative
        if not candidate.is_file():
            continue
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            notes.append(f"could not read {relative} to compare its symbols: {error}")
            continue
        defined |= {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    if not defined:
        return

    kept = sorted(set(declared_symbols) & defined)
    if kept:
        return
    dropped = sorted(set(declared_symbols) - defined)
    problems.append(
        "the workset's own declaration is self-contradictory for this operator, and the "
        f"`apply` block cannot satisfy both halves: `integration.substitution` is "
        f"'{substitution}' — the edit lives INSIDE an existing function — while "
        f"`apply_mode` is '{apply_mode}', which replaces the whole file. The replacement "
        f"defines {sorted(defined)} and keeps NONE of the {len(declared_symbols)} symbols "
        f"`integration.module_symbols` says the file exports: {dropped}. This is not a bad "
        "replacement; a file that could stand in for the target is not a file the "
        "performance harness can exec and call `run` on, so no seed satisfies both. The "
        "workset is what has to change (M5.1.1)"
    )


def _same_path(label: str, expected, actual, problems: list[str]) -> None:
    if not _same_file(expected, actual):
        problems.append(
            f"{label}: workset says {expected!r}, the handoff says {actual!r}. "
            f"These are compared by the part the two path frames share; neither is a "
            f"suffix of the other, so they are different files"
        )


def _check_against_snapshot(doc: dict, snapshot: dict, problems: list[str],
                            packup: Path, notes: list[str]) -> None:
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
    #
    # **`source_file` is compared by the part the two frames share, not by
    # equality**, and this body already knew that 300 lines down: `_check_apply`
    # compares `apply.files[].container_path` against the same `source_file` with
    # exactly this rule, and says why — the workset writes repo-relative
    # (`python/sglang/srt/layers/sampler.py`) while the container frame is
    # root-relative (`@SGLANG_ROOT@/srt/...`, and `SGLANG_ROOT` is
    # `/sgl-workspace/sglang/python/sglang`). Comparing one of them strictly and
    # the other by suffix meant no spelling of the workset's
    # `integration.target_files` could satisfy both readers: the frame that let
    # the path resolve was refused here, and the frame that passed here resolved
    # to a doubled path that exists in no image. Measured both ways 2026-09-04.
    #
    # `entry_function` stays strict. It is a symbol, not a path; there are no two
    # frames for it, and a suffix rule on a name would accept `fwd_o` for
    # `chunk_fwd_o`.
    declared = operator.get("edit_target") or {}
    point = (doc.get("apply") or {}).get("integration_point") or {}
    _same("apply.integration_point.entry_function",
          declared.get("entry_function"), point.get("entry_function"), problems)
    _same_path("apply.integration_point.source_file",
               declared.get("source_file"), point.get("source_file"), problems)
    _substitution_matches_apply_mode(packup, operator, doc, problems, notes)

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
            except Exception as exc:  # noqa: BLE001 — see below
                # **Anything the loader raises is a FAIL, never an escape.**
                # `schema.validate` promises `SchemaError`; it can raise other
                # things, and one is live: with `referencing` unimportable it
                # falls back to a registry-less validator, and this schema
                # `$ref`s `environment.schema.json`, so `iter_errors` raises
                # `_WrappedReferencingError: Unresolvable`. Measured 2026-09-03.
                #
                # Uncaught, that kills the body — non-zero exit and **no
                # `verdict.json`**, which `PhaseRunner` sees as a broken
                # validator rather than a refused handoff. A body that cannot
                # validate must say so and fail, not disappear.
                problems.append(
                    f"the schema loader raised {type(exc).__name__}: {exc}. "
                    "The document was NOT validated"
                )
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
                except Exception as exc:  # noqa: BLE001 — same reason as above
                    problems.append(
                        f"the schema loader raised {type(exc).__name__} on the workset snapshot: "
                        f"{exc}. The snapshot was NOT validated"
                    )
                    snapshot = None

    if doc is not None and snapshot is not None:
        _check_against_snapshot(doc, snapshot, problems, packup, notes)
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

    if doc is not None:
        _check_apply(doc, packup, problems)

    return not problems


def _check_apply(doc: dict, packup: Path, problems: list[str]) -> None:
    """The apply block, `apply/manifest.json`, and the two agreeing.

    m5's `apply_patch` is a program because it *reads* the manifest rather than
    judging it, so three things have to hold and none of them is about taste:
    the manifest is where `apply_patch` globs for it, it says what the document
    says, and every file it names is one the workset declared and one the packup
    actually carries.
    """
    apply = doc.get("apply") or {}
    manifest_rel = str(apply.get("manifest") or "apply/manifest.json")
    manifest_path = packup / manifest_rel

    if not manifest_path.is_file():
        problems.append(
            f"missing {manifest_rel} — `apply_patch` globs `*/apply/manifest.json` and a manifest "
            "anywhere else is one it will not find (see the CONTRACT constant in "
            "assets/apply_patch.task/apply.py)"
        )
        manifest = None
    else:
        manifest = _load_json(manifest_path, problems)

    declared = str(((apply.get("integration_point") or {}).get("source_file")) or "")
    entries = apply.get("files") or []

    for index, entry in enumerate(entries):
        where = f"apply.files[{index}]"
        container_path = str(entry.get("container_path") or "")

        # The container path must be the file the WORKSET declared. The two are
        # written in different frames — the workset is repo-relative
        # (`python/sglang/srt/layers/sampler.py`) and the manifest is
        # root-relative (`@SGLANG_ROOT@/srt/layers/sampler.py`, and SGLANG_ROOT
        # is `/sgl-workspace/sglang/python/sglang`) — so they are compared by
        # the part they share rather than by a mapping this body would have to
        # hard-code and keep in step with `container_roots.yaml`.
        #
        # `_same_file` is the shared reader. It used to be this expression alone
        # while `integration_point.source_file` was compared strictly, and the
        # two disagreeing is the fault §4.3 names — so the rule lives in one
        # function now rather than in one function and one inline expression.
        if declared and container_path:
            tail = container_path.split("@", 2)[-1].lstrip("/")
            if not _same_file(declared, tail):
                problems.append(
                    f"{where}.container_path {container_path!r} is not the file the workset "
                    f"declared as its integration point ({declared!r}). M5.1.1: the apply is "
                    "written against the workset's edit_target, not against whatever file the "
                    "optimiser happened to open"
                )

        replacement = entry.get("replacement")
        if replacement and not (packup / str(replacement)).is_file():
            problems.append(f"{where}.replacement names {replacement!r}, which is not in the packup")
        patch = entry.get("patch")
        if patch and not (packup / str(patch)).is_file():
            problems.append(f"{where}.patch names {patch!r}, which is not in the packup")

    # The manifest is the copy m5 opens; the document is the copy every other
    # consumer reads. Two records of one fact is a drift waiting to happen, so
    # they are compared rather than trusted.
    if manifest is not None:
        if manifest.get("files") != entries:
            problems.append(
                f"{manifest_rel} and the document's apply.files disagree — m5 reads the manifest "
                "and everything else reads the document, so a difference here is two answers to "
                "one question"
            )
        for field, value in (
            ("apply_mode", apply.get("apply_mode")),
            ("operator_id", doc.get("operator")),
        ):
            if manifest.get(field) != value:
                problems.append(f"{manifest_rel} says {field}={manifest.get(field)!r}, the document says {value!r}")


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    # **The reasons, on disk, beside the verdict** — m3's `workset_io.write_report`,
    # reused rather than re-implemented. A validator's stdout is kept nowhere
    # (`temp/bugs/2026-09-03-a-validators-stdout-is-not-kept-anywhere.md`), so a
    # zone holds `args.json`, `inputs.json`, `materials.json`, `verdict.json` and
    # **not one word about why**.
    #
    # This stage supplied the sixth instance: on 2026-09-04 rung 0 reached
    # `optimize_kernel` for the first time, this validator refused, and the only
    # thing recoverable was `kernel_optimization: invalid`. The finding was real
    # and the reason was gone before anyone could read it.
    findings: dict[str, tuple[list[str], list[str]]] = {}
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
        findings[hid] = (problems, notes)
        for note in notes:
            print(f"{hid} note: {note}")
        for problem in problems:
            print(f"{hid}: {problem}")
    # **Before `write_verdict`, deliberately.** m3's ordering: a crash in the
    # report writer must not take the reasons with it, and a verdict that
    # arrives without them is the state this exists to end.
    W.write_report("check_optimization_shape", findings)
    zone.write_verdict(verdicts)
    print(f"check_optimization_shape: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
