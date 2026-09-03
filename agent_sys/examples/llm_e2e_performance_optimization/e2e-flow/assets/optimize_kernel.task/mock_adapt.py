#!/usr/bin/env python3
"""MOCK-MAP adaptation (G) — bring the sealed stage-4 artefact up to this contract.

`assets/lib/mock.sh` has just copied `stage4-kernel-opt/kernel_optimization`'s
bytes into the output handoff. Those bytes are a real run's and they stay a real
run's; what they predate is two fields, and the gap is structural:

* **`premise`** — M4.3.5 did not exist when that run was sealed. The run's own
  headline finding *was* a premise mismatch (a gfx942 workset against a gfx950
  host, 9.6% apart on `B8_V151936`), recorded in prose in `notes.md` and
  `verification.json` because there was no field for it.
* **`apply`** — M5.1.1 did not exist either, and the run was `KFO_MOCK=1`, so
  there was no optimised kernel and nothing to apply.
* **`results/workset.snapshot.yaml`** and the carried baseline report — the
  merged `operator_workset` kind did not exist, so there was no `workset.yaml`
  to snapshot.

This script renders those from the workset that is **actually staged as this
task's input**, the way adaptation (A) renders `environment.yaml` from a sealed
record plus the run's own `--var`s. Nothing is invented: every measurement stays
the sealed run's, and every rendered field is a copy of the staged workset's.

**Why the mock then passes where a verbatim copy would abort.** Both sides of
the premise comparison become the environment record m1 minted for *this* run —
m1 through m4 share one container on one node (CONTRACT §5) — so
`fixed.gpu_arch` matches itself and `verdict.held` is true. That is not the mock
being lenient; it is what a real run of this flow looks like, and it is the only
configuration in which m5 ever gets to run.

`--premise mismatched` reproduces the sealed run's abort instead, which is the
only cheap end-to-end test of the abort path this package has. Use it once,
deliberately, the way MOCK-MAP (E) uses the refused `integration_report`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "steps"))

import _lib as lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", required=True, help="$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION")
    ap.add_argument(
        "--premise",
        default=os.environ.get("E2E_MOCK_PREMISE", "held"),
        choices=("held", "mismatched"),
        help="`mismatched` reproduces the sealed run's gfx942-vs-gfx950 abort on purpose",
    )
    a = ap.parse_args()

    content = Path(a.handoff)
    codes = content / "items" / "codes"
    packups = sorted(p for p in codes.iterdir() if p.is_dir()) if codes.is_dir() else []
    if len(packups) != 1:
        lib.die(f"expected exactly one packup under {codes}, found {[p.name for p in packups]}")
    packup = packups[0]

    workset = lib.load_workset()
    operator = lib.pick_operator(workset, os.environ.get("KFO_WORKSET_OPERATOR") or None)
    operator_id = str(operator.get("operator_id"))
    workset_root = lib.workset_root()
    run_env = lib.load_environment()
    ground = workset.get("ground_truth") or {}

    # --- carry the workset with the handoff ---------------------------------
    apparatus = packup / lib.APPARATUS
    if apparatus.exists():
        shutil.rmtree(apparatus)
    apparatus.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workset_root, apparatus)
    (packup / "results").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workset_root / "workset.yaml", packup / lib.SNAPSHOT)

    baseline_rel = (workset.get("evidence") or {}).get("performance_report")
    baseline: dict[str, float] = {}
    if baseline_rel and (workset_root / str(baseline_rel)).is_file():
        shutil.copyfile(workset_root / str(baseline_rel), packup / lib.BASELINE_REPORT)
        baseline = lib.report_medians(lib.load_json(packup / lib.BASELINE_REPORT), operator_id)

    # --- the premise --------------------------------------------------------
    workset_env = json.loads(json.dumps(ground.get("environment") or run_env))
    aborted: list[dict] = []
    if a.premise == "mismatched":
        # The sealed run's own finding, reproduced rather than invented: its
        # workset was measured on gfx942 and it ran on gfx950.
        workset_env.setdefault("fixed", {})["gpu_arch"] = "gfx942"
        aborted = [{
            "field": "fixed.gpu_arch",
            "expected": "gfx942",
            "actual": (run_env.get("fixed") or {}).get("gpu_arch"),
            "stage": "m4",
        }]

    premise = {
        "abort_on_mismatch": list(ground.get("abort_on_mismatch") or ["fixed.gpu_arch"]),
        "warn_on_mismatch": list(ground.get("warn_on_mismatch") or []),
        "workset_environment": workset_env,
        "run_environment": run_env,
        "verdict": {"held": not aborted, "aborted_on": aborted, "warnings": []},
    }

    # --- the document -------------------------------------------------------
    #
    # The sealed run's measurements, read out of the artefact it already
    # carries. `verification.json` is that run's own record and every figure
    # below is copied from it; nothing here is recomputed and nothing is guessed.
    verification = {}
    for name in ("verification.json",):
        candidate = packup / "results" / name
        if candidate.is_file():
            verification = lib.load_json(candidate)
    forge_result = {}
    if (packup / "results" / "forge_result.json").is_file():
        forge_result = lib.load_json(packup / "results" / "forge_result.json")

    measured = {k: float(v) for k, v in (verification.get("baseline_median_ms") or {}).items()}
    target = operator.get("edit_target") or {}
    kernel = packup / "results" / "optimized_kernel.py"

    # --- the apply block ----------------------------------------------------
    #
    # A real run reads `base_sha256` off the engine tree in its own container
    # (`60_write_handoff.py`). A mock may be running on a login node with no
    # engine tree at all, so it hashes what it can and **says which**: the real
    # stock file when it is reachable, and otherwise the sealed replacement,
    # marked in `notes`. A mock that quietly writes a well-formed hash of the
    # wrong file is one m5 would refuse two stages later with no way to tell a
    # stale patch from a fabricated one.
    container_path = lib.container_path_for(str(target.get("source_file") or "")) or ""
    stock = lib.expand_container_path(container_path) if container_path else None
    if stock is not None and stock.is_file():
        base_sha256, sha_source = lib.sha256_of(stock), "the engine tree in this container"
    elif kernel.is_file():
        base_sha256, sha_source = lib.sha256_of(kernel), "the sealed replacement (NO ENGINE TREE REACHABLE)"
    else:
        base_sha256, sha_source = "0" * 64, "nothing (NO ENGINE TREE AND NO KERNEL)"

    apply_block = {
        "apply_mode": "overlay_files",
        "manifest": lib.APPLY_MANIFEST,
        "image": ((run_env.get("fixed") or {}).get("image")),
        "logical_operator": operator_id,
        "integration_point": {
            k: v for k, v in target.items()
            if k in ("source_file", "entry_function", "entry_function_line", "repo_root_var")
        },
        "files": (
            [{
                "container_path": container_path,
                "base_sha256": base_sha256,
                "change": "modify",
                "replacement": "results/optimized_kernel.py",
            }]
            if kernel.is_file() and container_path else []
        ),
        "revert": "Remove the overlay and restart the engine.",
    }
    if target.get("entry_function"):
        apply_block["runtime_marker"] = {
            "first_call": re.escape(str(target["entry_function"])) + r"\s*\("
        }

    document = {
        "schema_version": 1,
        "operator": operator_id,
        "workset_ref": {
            "handoff_id": str(Path(lib.input_content("operator_workset")).parent.parent.name),
            "version": str(Path(lib.input_content("operator_workset")).parent.name),
            "digest": None,
            "workset_id": workset.get("workset_id"),
            "snapshot": lib.SNAPSHOT,
        },
        "premise": premise,
        "apply": apply_block,
        "evidence": {
            "correctness": {
                "entrypoint": ((lib.entrypoints(workset, operator).get("correctness")) or {}).get("cmd"),
                "report": "results/correctness_report.json",
                "passed": verification.get("correctness_passed") is True,
                "shapes": [
                    {"case_id": case, "passed": True, "snr_db": snr, "allclose": True}
                    for case, snr in (verification.get("snr_db_per_case") or {}).items()
                ],
            },
            "performance": {
                "entrypoint": ((lib.entrypoints(workset, operator).get("performance")) or {}).get("cmd"),
                "protocol": workset.get("protocol"),
                "baseline": {
                    "source": "workset",
                    "report": str(baseline_rel),
                    "per_case_ms": baseline,
                },
                "measured": {
                    "report": "results/verification.json",
                    "per_case_ms": measured or baseline,
                },
            },
            # `mock: true` regardless of `--premise`: the sealed run was one,
            # and the schema then forbids a claim on that ground alone. Saying
            # otherwise to make the mock look like a campaign is exactly the
            # artefact `check_optimization_shape`'s mock-consistency rule exists
            # to refuse.
            "forge": {
                "ran": False,
                "mock": True,
                "degraded": False,
                "result_json": "results/forge_result.json" if forge_result else None,
                "mean_case_speedup": forge_result.get("mean_case_speedup"),
                "improved": forge_result.get("improved"),
                "snr_db": verification.get("snr_db"),
                "iteration_count": forge_result.get("iteration_count", 0),
            },
        },
        "notes": (
            "MOCK. The bytes of this handoff are the sealed 2026-09-02 stage-4 run's "
            "(KFO_MOCK=1, no campaign, no optimised kernel, nothing claimed). MOCK-MAP "
            "adaptation (G) rendered `premise`, `apply`, the workset snapshot and the carried "
            "baseline report from the workset staged as this task's input, because the sealed "
            "artefact predates all four. Every measurement is the sealed run's; every rendered "
            "field is a copy of the staged workset's. apply.files[].base_sha256 was hashed from "
            f"{sha_source}."
            + (
                " `--premise mismatched`: the workset environment's gpu_arch was set to gfx942 to "
                "reproduce the sealed run's own abort, which is the only cheap test of the abort "
                "path this package has. Expect m4 to refuse and the graph to stop here."
                if aborted else ""
            )
        ),
    }
    lib.write_json(packup / lib.APPLY_MANIFEST, {
        "schema_version": 1,
        "operator_id": operator_id,
        "logical_operator": apply_block["logical_operator"],
        "image": apply_block["image"],
        "apply_mode": apply_block["apply_mode"],
        "files": apply_block["files"],
        **({"runtime_marker": apply_block["runtime_marker"]} if "runtime_marker" in apply_block else {}),
    })
    lib.write_json(packup / lib.DOC, document)

    problems = lib.validate("kernel_optimization", document)
    if problems:
        print("the adapted mock does not validate against kernel_optimization.schema.json:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(
        f"mock-adapt: rendered premise ({'held' if not aborted else 'ABORTED'}), apply, "
        f"snapshot and baseline report into {packup.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
