#!/usr/bin/env python3
"""`check_workset_runs` — trustworthiness, strong.

Every driver ran on the target GPU, met the correctness gate, and produced a
measurement with the shape mission 3.2.7 asks for.

This is the only validator in the package whose subject is a number rather than
a document, and it is the last line before KernelForge starts depending on these
drivers. forge-loop cannot check them: it treats `--driver` as a protected file
and takes what it prints on trust.

`bench_stats` is imported rather than reimplemented. The producer computes the
weighted average with the same module, so a disagreement between what was
written and what is accepted is impossible by construction — which is the point
of putting the arithmetic in `assets/lib/` instead of in either body.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import bench_stats  # noqa: E402
import store  # noqa: E402


def check(content: Path, args: dict) -> tuple[bool, str]:
    document = content / "items" / "text.json"
    if not document.is_file():
        return False, "items/text.json is absent"
    try:
        data = json.loads(document.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return False, f"items/text.json is not valid JSON: {error}"

    operators = data.get("operators")
    if not isinstance(operators, list) or not operators:
        return False, "operators is absent or empty"

    if not (data.get("environment") or {}).get("node"):
        return False, "environment.node is empty; the evidence does not say where it was measured"

    min_groups = int(args.get("min_groups") or bench_stats.MIN_GROUPS)
    min_iters = int(args.get("min_iters_per_group") or bench_stats.MIN_ITERS_PER_GROUP)
    max_rsd = float(args.get("max_rsd") or 0.1)

    clean = 0
    notes = []
    for operator in operators:
        label = operator.get("operator_id", "?")

        if not operator.get("ran"):
            notes.append(f"{label}: did not run ({operator.get('failure') or 'no reason recorded'})")
            continue
        if not operator.get("correct"):
            notes.append(f"{label}: ran but failed the gate ({operator.get('failure')})")
            continue

        bench = operator.get("bench")
        if not bench:
            notes.append(f"{label}: correct but produced no measurement")
            continue

        if int(bench.get("groups") or 0) < min_groups:
            return False, (
                f"{label}: {bench.get('groups')} measurement group(s), "
                f"mission 3.2.7 requires {min_groups}"
            )
        per_group = bench.get("per_group_ms") or []
        if len(per_group) != int(bench.get("groups") or 0):
            return False, f"{label}: per_group_ms has {len(per_group)} entries for {bench.get('groups')} groups"
        if int(bench.get("iters_total") or 0) < min_groups * min_iters:
            return False, (
                f"{label}: {bench.get('iters_total')} iterations in total, "
                f"below {min_groups}x{min_iters}"
            )

        # Recompute rather than trust. The producer and this share
        # `bench_stats`, so a stored figure that disagrees means the record was
        # edited after it was measured.
        groups = [{"iters": int(bench["iters_total"]) // len(per_group), "mean_ms": m} for m in per_group]
        recomputed = bench_stats.weighted_mean(groups)
        stored = float(bench.get("weighted_mean_ms") or 0.0)
        if stored <= 0:
            return False, f"{label}: weighted_mean_ms is {stored}, which is not a duration"
        if abs(recomputed - stored) / stored > 0.01:
            return False, (
                f"{label}: weighted_mean_ms is {stored:.6f} but the per-group figures "
                f"give {recomputed:.6f}; the record disagrees with itself"
            )

        rsd = float(bench.get("rsd") or 0.0)
        if rsd > max_rsd:
            return False, (
                f"{label}: run-to-run spread {rsd:.4f} exceeds {max_rsd}. The machine was "
                f"not quiet, and a baseline measured now would make forge-loop chase noise"
            )
        clean += 1

    ratio = clean / len(operators)
    floor = float(args.get("min_pass_ratio") or 0.5)
    if ratio < floor:
        return False, (
            f"{clean} of {len(operators)} workset(s) measured cleanly (ratio {ratio:.2f}, "
            f"floor {floor}). " + "; ".join(notes[:3])
        )

    return True, (
        f"{clean}/{len(operators)} measured cleanly on "
        f"{(data.get('environment') or {}).get('node')}"
        + (f"; {len(notes)} not usable yet" if notes else "")
    )


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_workset_runs: {hid}: no published content")
            continue
        ok, why = check(content, args)
        results[hid] = ok
        print(f"check_workset_runs: {hid}: {'PASS' if ok else 'FAIL'} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
