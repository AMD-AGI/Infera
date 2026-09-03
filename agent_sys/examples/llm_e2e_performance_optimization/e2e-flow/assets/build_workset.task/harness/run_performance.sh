#!/usr/bin/env python3
"""One-click performance over every operator and every shape (M3.7.4).

    ./run_performance.sh [--operator ID] [--shape CASE_ID] [--impl PATH] [--json OUT]

**This produces m4's denominator.** With no `--impl` it measures the workset's
own baseline and reproduces `evidence/performance.json`; with `--impl` it
measures a candidate under the same protocol on the same shapes. A speedup is a
ratio between two runs of this one script, which is the entire reason M4.3.5
could be reversed from "do not trust the workset's number" to "take the ground
truth strictly from the workset". Protected, for the reason in `_common.py`.

**Five groups of ten, each group a fresh measurement, and the per-group figures
are kept.** The reduction is recorded *and* recomputable: `check_workset_runs`
imports the same `weighted_mean` and rejects a stored figure that does not
follow from `per_group_ms`, so a record edited after it was measured cannot
survive. Fewer groups than five cannot distinguish a 5% win from the ~2%
round-to-round spread a steady node already has.
"""

import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import build_inputs, finish, load_definition, load_impl, setup  # noqa: E402


def _sync():
    """Device-side completion before the clock is read.

    Without it every timing is a launch-queue measurement: the host returns
    immediately and a kernel that got slower looks identical to one that got
    faster. Guarded rather than assumed, so the harness still runs on a host
    with no GPU — where it is timing CPU work and says so through
    `protocol.timing`.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _time_group(callable_, inputs, iters: int) -> float:
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        callable_(**inputs)
    _sync()
    return (time.perf_counter() - start) * 1000.0 / iters


def main() -> int:
    ctx = setup("performance")
    protocol = ctx.doc["protocol"]
    ctx.report["protocol"] = protocol
    evidence = ctx.doc.get("evidence") or {}
    ctx.report["correctness_report"] = evidence.get("correctness_report")

    groups = int(protocol["groups"])
    iters = int(protocol["iters_per_group"])
    warmup = int(protocol.get("warmup", 0))
    ok = True

    for operator in ctx.operators:
        definition = load_definition(operator)
        row = {"operator_id": operator["operator_id"], "ran": False, "failure": None,
               "weighted_mean_ms": None, "shapes": []}
        try:
            under_test = load_impl(operator, definition, ctx.args.impl, "baseline")
        except Exception as error:  # noqa: BLE001
            row["failure"] = f"could not load the implementation: {error!r}"
            ctx.report["operators"].append(row)
            ok = False
            continue

        row["ran"] = True
        for shape in ctx.shapes(operator):
            try:
                inputs = build_inputs(definition, shape)
                for _ in range(warmup):
                    under_test(**inputs)
                per_group = [_time_group(under_test, inputs, iters) for _ in range(groups)]
            except Exception as error:  # noqa: BLE001
                row["failure"] = repr(error)
                ok = False
                break
            mean = sum(per_group) / len(per_group)
            spread = (sum((m - mean) ** 2 for m in per_group) / len(per_group)) ** 0.5
            row["shapes"].append({
                "case_id": shape["case_id"], "uuid": shape["uuid"],
                "groups": groups, "iters_total": groups * iters,
                "per_group_ms": per_group, "weighted_mean_ms": mean,
                "median_ms": statistics.median(per_group),
                "rsd": spread / mean if mean else 0.0,
            })

        if row["shapes"]:
            # Weighted by how often each shape actually occurred in the capture.
            # An unweighted mean over three shapes lets a rare edge case move the
            # headline as much as the modal one.
            weights = {s["case_id"]: max(1, int(s.get("calls") or 1)) for s in operator["shapes"]}
            total = sum(weights[s["case_id"]] for s in row["shapes"])
            row["weighted_mean_ms"] = sum(
                s["weighted_mean_ms"] * weights[s["case_id"]] for s in row["shapes"]) / total
        ctx.report["operators"].append(row)

    finish(ctx, ok)


if __name__ == "__main__":
    main()
