#!/usr/bin/env python3
"""`check_identity_resolved` — trustworthiness, strong.

Every candidate carries a resolution level, and an unresolved one says why
rather than guessing.

**Read `min_resolve_ratio: 0.0` before reading anything else here.** It is
deliberate and it is not a disabled check. The evidence available to `identify`
is a device symbol; for a Tensile GEMM dispatched from C++ there is no Python
frame to find, and the symbol a generator emits frequently appears nowhere in
the checkout — the sealed artefact's own case is
`mfma_moe1_silu_mul_afp4_wfp4_bf16_g1u1`, whose generator never emits `_g1u1`
at all. A floor above zero would fail a correct analysis on exactly the kernels
the next stage excludes anyway, and the pressure it would create is to write a
confident `source_file_path` rather than an honest `resolution_hint`. That is
the opposite of what this validator is for.

So what is graded is that **uncertainty is modelled, not that it is absent**.
The schema carries most of that — an operator with no `source_file_path` must
carry a `resolution_hint` or an `excluded_reason`, and `agent_recovered` must
carry a hint — and this file carries the three rules a schema cannot state:

1. **Every `image_repo_path` is declared.** It must be a key of
   `container_root_placeholders`. A placeholder nobody defined expands to
   nothing and the consumer silently reads the wrong tree.
2. **`summary.resolve_ratio` follows from `operators`.** Recomputed, not
   trusted: a stored ratio that does not follow from the list is a record edited
   after it was produced. This is the same move `check_workset_runs` makes with
   `weighted_mean_ms` and `check_no_regression` makes with its verdict field.
3. **The floor, once the ratio is known to be true.** Reported always, enforced
   only when `min_resolve_ratio` is above zero.

And the schema-copy rule, CONTRACT.md §3.4, as in every `structured_text` kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as S  # noqa: E402
import zone  # noqa: E402

#: The two methods that put a file on disk. `agent_recovered` gave a direction
#: and `declared` was asserted a priori; neither is counted as resolved, and
#: `agent_recovered` is **not a failure** — it is the path two of Arena's own
#: samples took.
_RESOLVING_METHODS = frozenset({"trace_python_stack", "symbol_search"})


def _resolved(operator: dict) -> bool:
    """One operator counts as resolved when a file was found *by a method that
    finds files*. Both halves matter: a `source_file_path` filled in under
    `agent_recovered` is a guess wearing a fact's clothes."""
    return bool(operator.get("source_file_path")) and operator.get("source_resolution_method") in _RESOLVING_METHODS


def _check(content: Path, args: dict, problems: list[str], notes: list[str]) -> bool:
    document = content / "items" / "text.json"
    if not document.is_file():
        problems.append("items/text.json is absent")
        return False
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"items/text.json is not valid JSON: {error}")
        return False

    name = args.get("schema") or "operator_identity"
    try:
        S.validate(name, data)
    except S.SchemaError as error:
        problems.extend(str(error).splitlines()[1:])
        return False

    carried = content / "items" / "schema"
    if not carried.is_file():
        problems.append("items/schema is absent; a structured_text handoff carries its own schema (CONTRACT 3.4)")
    elif carried.read_bytes() != S.schema_path(name).read_bytes():
        problems.append(f"items/schema differs from assets/schemas/{name}.schema.json")

    operators = data["operators"]
    declared = set(data["container_root_placeholders"])

    # 1. Every placeholder used is a placeholder defined.
    for operator in operators:
        placeholder = operator["image_repo_path"]
        if placeholder not in declared:
            problems.append(
                f"{operator['kernel_id']}: image_repo_path {placeholder} is not in "
                f"container_root_placeholders {sorted(declared)}. An undeclared placeholder "
                f"expands to nothing and the consumer reads the wrong tree"
            )

    ids = [o["kernel_id"] for o in operators]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate kernel_id(s): {sorted({i for i in ids if ids.count(i) > 1})}")

    names = [o["logical_operator"] for o in operators]
    if len(set(names)) != len(names):
        problems.append(
            f"duplicate logical_operator(s): {sorted({n for n in names if names.count(n) > 1})}. "
            f"It becomes a directory name in the workset, so two of them collide silently"
        )

    # 2. The stored ratio follows from the list.
    resolved = [o for o in operators if _resolved(o)]
    actual = len(resolved) / len(operators)
    summary = data["summary"]
    if summary["operators"] != len(operators):
        problems.append(f"summary.operators is {summary['operators']}, the list holds {len(operators)}")
    if summary["resolved"] != len(resolved):
        problems.append(
            f"summary.resolved is {summary['resolved']}, the list gives {len(resolved)}. "
            f"An operator counts as resolved only when a file was found by a method that finds files"
        )
    if abs(summary["resolve_ratio"] - actual) > 0.005:
        problems.append(
            f"summary.resolve_ratio is {summary['resolve_ratio']:.3f} but the list gives {actual:.3f}; "
            f"the record disagrees with itself"
        )

    # 3. The floor, on the recomputed figure rather than the stored one.
    # `or 0.0` would read an explicit 0 as the default, which is harmless only
    # because the default is 0.0. Written explicitly so it stays harmless if
    # somebody changes the default.
    raw = args.get("min_resolve_ratio")
    floor = 0.0 if raw is None or raw == "" else float(raw)
    if actual < floor:
        problems.append(f"resolve_ratio {actual:.3f} is below the floor {floor}")

    # Reported, never graded: this is the number a reader wants and the number
    # that must not become a target.
    unresolved = [o for o in operators if not _resolved(o)]
    notes.append(f"{len(resolved)}/{len(operators)} resolved (ratio {actual:.2f}, floor {floor})")
    for operator in unresolved:
        hint = operator.get("resolution_hint") or operator.get("excluded_reason") or ""
        notes.append(
            f"  {operator['kernel_id']} {operator['logical_operator']}: "
            f"{operator['source_resolution_method']} — {hint[:100]}"
        )

    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        notes: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            verdicts[hid] = _check(content, args, problems, notes)
        for note in notes:
            print(f"{hid}: {note}")
        for problem in problems:
            print(f"{hid}: {problem}")
    zone.write_verdict(verdicts)
    print(f"check_identity_resolved: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
