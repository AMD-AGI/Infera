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
    # **`timing` is read here, and that is the whole point of this block.**
    # It was declared in `workset.yaml` as `event` and consumed by nothing —
    # while every measurement below is `perf_counter()` wrapped in syncs. m2
    # found it by reading both harnesses side by side: the word `timing`
    # appeared only in a docstring. A field nothing reads cannot be wrong, so
    # it stayed wrong, and a reader of the workset reasonably believed HIP
    # events had been used.
    #
    # Refusing an unimplemented value is the half that keeps it honest. Naming
    # the method truthfully would fix it once; making the harness *read* the
    # name is what stops it drifting again — the reader that could not be told
    # can now be told (CONTRACT 4.3).
    timing = str(protocol.get("timing", "wall_clock_sync"))
    if timing != "wall_clock_sync":
        sys.exit(
            f"protocol.timing is {timing!r} and this harness implements only 'wall_clock_sync': "
            f"time.perf_counter() around `iters` calls, with torch.cuda.synchronize() before the "
            f"clock starts and again before it stops. Refusing rather than measuring one thing "
            f"and labelling it another — every number here would carry the wrong method name."
        )
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

    # The noise floor, derived rather than declared. Two independent
    # measurements each with relative standard deviation `s` differ by more than
    # `sqrt(2) * z * s` by chance alone; at z = 2 that is 2.83 s. Computed from
    # the spread this run actually saw, so a noisy host correctly demands a
    # bigger win than a quiet one instead of both being handed 1.05.
    spreads = [s["rsd"] for o in ctx.report["operators"] for s in o["shapes"]]
    ctx.report["noise_floor"] = round(1.0 + 2.83 * max(spreads), 4) if spreads else 1.01

    finish(ctx, ok)


if __name__ == "__main__":
    main()
