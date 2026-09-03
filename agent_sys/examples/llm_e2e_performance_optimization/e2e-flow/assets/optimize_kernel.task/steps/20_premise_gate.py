#!/usr/bin/env python3
"""STEP 2 — the M4.3.5 premise gate. Run before anything is spent.

    优化任务的 ground truth 本身就应该严格的从 workset 中来，如果最基础的硬件、
    优化前提不一样，直接报错 abort。软件环境不太一样可以报 warning。

Compares the workset's `ground_truth.environment` — where the baseline this
stage divides by was measured — against the environment this run is actually
in, field by field, using **the workset's own** `abort_on_mismatch` and
`warn_on_mismatch` lists. The workset declares what its numbers depend on; m4
does not get to decide that its own difference was the harmless kind.

Exit codes are the interface, because the readme branches on them:

  0  the premise holds. Warnings may have been recorded; they are not failures.
  2  **ABORT.** An abort-level field differs. Do not run forge, do not measure.
     Go to STEP 6 and write the handoff with this verdict in it.
  1  the gate could not be run at all.

**Why an abort rather than a smaller pass.** Measured 2026-09-02: the sealed
stage-4 run timed `B8_V151936` at 50.18 µs on gfx950 against the workset's
55.40 µs on gfx942. Divide one by the other and 9.6% of newer silicon appears
as a speedup, in a comparison that reads as entirely legitimate and that nobody
downstream can detect. The previous package's answer was to silently
re-baseline, which makes the report internally consistent and still answers a
different question.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib as lib  # noqa: E402

#: A floor under whatever the workset declares. A workset that forgot
#: `fixed.gpu_arch` is not licence to skip it — the mission names these four.
_FLOOR_ABORT = ("fixed.gpu_arch",)


def _dig(doc: dict, dotted: str):
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _resolve(field: str, environment: dict) -> tuple[str, object]:
    """A premise field name as a dotted path into an environment record.

    The workset writes dotted paths (`fixed.gpu_arch`); the mission names bare
    leaves (`gpu_arch`). Both are accepted, and a bare name is looked up under
    `fixed` then `runtime`, so the two spellings cannot silently check different
    fields. `check_speedup_substantiated` resolves them identically — the two
    have to agree or a producer passes here and fails there.
    """
    if "." in field:
        return field, _dig(environment, field)
    for section in ("fixed", "runtime"):
        value = _dig(environment, f"{section}.{field}")
        if value is not None:
            return f"{section}.{field}", value
    return f"fixed.{field}", None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pinned = lib.load_json(Path(a.inputs))
    ground = pinned.get("ground_truth") or {}
    workset_env = ground.get("environment") or {}
    run_env = dict(pinned.get("run_environment") or {})
    if not workset_env:
        lib.die("the workset's ground_truth carries no environment; the premise cannot be compared")

    abort_fields = list(dict.fromkeys(list(ground.get("abort_on_mismatch") or []) + list(_FLOOR_ABORT)))
    warn_fields = list(ground.get("warn_on_mismatch") or [])

    # Deduplicated by **resolved path**, not by the name written down: the
    # workset spells `fixed.gpu_arch` and `_FLOOR_ABORT` spells the same thing,
    # so a set of the raw strings would report one mismatch twice.
    # `check_speedup_substantiated` dedupes identically — the two have to agree
    # or a producer sees a different list here than the grader uses.
    aborted, warnings = [], []
    seen: set[str] = set()
    for field in abort_fields:
        # `operator`, `shapes` and `dtype` are not environment fields. They are
        # compared against the workset itself, by STEP 1 (which refused a
        # workset that does not define the operator or declare the shapes) and
        # again by `check_speedup_substantiated`.
        if field in ("operator", "shapes", "dtype"):
            continue
        path, expected = _resolve(field, workset_env)
        if path in seen:
            continue
        seen.add(path)
        _, actual = _resolve(field, run_env)
        if expected != actual:
            aborted.append({"field": path, "expected": expected, "actual": actual, "stage": "m4"})

    seen = set()
    for field in warn_fields:
        path, expected = _resolve(field, workset_env)
        if path in seen:
            continue
        seen.add(path)
        _, actual = _resolve(field, run_env)
        if expected != actual:
            warnings.append({"field": path, "expected": expected, "actual": actual, "stage": "m4"})

    # The warnings travel forward in the environment record's own channel, which
    # `environment.schema.json` defines for exactly this, so m5 sees them
    # without knowing anything about m4's premise block.
    carried = list(run_env.get("warnings") or [])
    for warning in warnings:
        if warning not in carried:
            carried.append(warning)
    if carried:
        run_env["warnings"] = carried

    verdict = {"held": not aborted, "aborted_on": aborted, "warnings": warnings}
    premise = {
        "abort_on_mismatch": list(ground.get("abort_on_mismatch") or []),
        "warn_on_mismatch": warn_fields,
        "workset_environment": workset_env,
        "run_environment": run_env,
        "verdict": verdict,
    }
    lib.write_json(Path(a.out), premise)

    for warning in warnings:
        print(
            f"warning: {warning['field']}: the workset was measured with {warning['expected']!r}, "
            f"this run has {warning['actual']!r} — recorded and carried forward",
            file=sys.stderr,
        )
    if aborted:
        print(
            "ABORT: the optimisation premise does not hold. No ratio computed on this machine "
            "answers the question the workset asked (M4.3.5):",
            file=sys.stderr,
        )
        for entry in aborted:
            print(
                f"  {entry['field']}: workset {entry['expected']!r}, this run {entry['actual']!r}",
                file=sys.stderr,
            )
        print(
            "Do not run forge and do not measure. Go to STEP 6 and write the handoff with this "
            "verdict in it; the schema will not let you write a claim beside it.",
            file=sys.stderr,
        )
        return 2

    print(f"ok: premise holds across {len(abort_fields)} abort-level field(s), {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
