#!/usr/bin/env python3
"""STEP 6 — assemble the packup and write the document.

**The ratios are computed here and not by the agent**, from the workset's own
baseline and STEP 5's measurement. That is the whole of M4.3.5 made mechanical:
there is no point in the flow at which a model chooses a denominator.

It refuses to write `evidence.performance.claim` when
  * the premise aborted,
  * forge was mocked, or
  * correctness did not pass.
The first two are the schema's rule and this script would fail validation if it
tried; the third is this script's, and it is the same rule STEP 5 enforces one
step earlier.

What it does **not** write: `README.md`, `REPRODUCE.md`, `environment.md` and
`notes.md` beyond a skeleton. Those are the part a cold reader needs, and no
script knows what surprised you.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib as lib  # noqa: E402

_SKELETON = {
    "README.md": """# Kernel optimisation — {operator}

Packup date {today}. Operator `{operator}`, from workset `{workset_id}`.

## Result

<!-- FIRST LINE MUST SAY WHAT HAPPENED, in words a reader cannot mistake.
     - mock            -> `MOCK RUN — no optimization was performed`
     - degraded budget -> `SMOKE TEST — degraded budget`
     - aborted premise -> say the premise did not hold and name the field
     Then: what was run, what was measured, and what was not. -->

## What this was

## Navigation

| file | what is in it |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable commands, and the expected output |
| `environment.md` | host, GPU, image, versions, and how they differ from the workset's |
| `notes.md` | the traps, including the ones that cost this run time |
| `results/kernel_optimization.json` | **the document every consumer reads** |
| `results/workset.snapshot.yaml` | the workset this was optimised against, verbatim |
| `results/optimized_kernel.py` | the kernel |
| `scripts/workset/` | the workset's own test apparatus, copied unmodified |
""",
    "REPRODUCE.md": """# Reproduce

Ordered and copy-pasteable. Every command was run.

```sh
```

## Expected output

""",
    "environment.md": """# Environment

A rendering of `results/kernel_optimization.json`'s `premise.run_environment`,
not a second source of truth. If the two disagree, the JSON is right.

## How this differs from the workset's environment

""",
    "notes.md": """# Notes

The traps, the wrong turns, and anything a later run should not repeat.

""",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--state", required=True, help="the directory STEPs 2-5 wrote into")
    ap.add_argument("--forge", default=None, help="STEP 3's workdir; defaults to <state>/../forge")
    ap.add_argument("--out", required=True, help="$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION")
    a = ap.parse_args()

    pinned = lib.load_json(Path(a.inputs))
    state = Path(a.state)
    forge_dir = Path(a.forge) if a.forge else state.parent / "forge"
    operator_id = str(pinned["operator_id"])

    premise = lib.load_json(state / "premise.json")
    held = bool((premise.get("verdict") or {}).get("held"))

    forge_result = {}
    if (forge_dir / "forge_result.json").is_file():
        forge_result = lib.load_json(forge_dir / "forge_result.json")
    mocked = bool(forge_result.get("mock"))
    degraded = (forge_dir / "degraded").is_file() and (forge_dir / "degraded").read_text().strip() == "true"

    correctness = lib.load_json(state / "correctness.json") if (state / "correctness.json").is_file() else {}
    performance = lib.load_json(state / "performance.json") if (state / "performance.json").is_file() else {}
    correctness_passed = correctness.get("passed") is True

    # --- the packup ---------------------------------------------------------
    #
    # `items/codes/` is required by the `code` content type: a file placed
    # directly under `items/` is rejected before anyone reads it. Exactly one
    # packup directory, `<name>.packup_<YYYYMMDD>` with a real eight-digit date.
    #
    # No explicit mode on any mkdir. Measured 2026-09-01: a run created a
    # directory at 0644, wrote seven files into it and could not read them
    # back — a directory without its execute bit cannot be traversed by anyone,
    # including its owner.
    today = date.today().strftime("%Y%m%d")
    packup = Path(a.out) / "items" / "codes" / f"{operator_id}.packup_{today}"
    (packup / "results").mkdir(parents=True, exist_ok=True)
    (packup / "scripts").mkdir(parents=True, exist_ok=True)

    workset_root = Path(pinned["workset_root"])
    apparatus = packup / lib.APPARATUS
    if apparatus.exists():
        shutil.rmtree(apparatus)
    # The whole runnable workset, copied unmodified. `check_speedup_
    # substantiated` re-measures from *this* copy, because a validator on an
    # output phase is handed only the handoffs it declared and cannot reach the
    # workset. A kit that reports a speedup and does not carry the thing that
    # measured it cannot be checked by anyone who does not already have the
    # workset, which is most readers.
    shutil.copytree(workset_root, apparatus)
    shutil.copyfile(workset_root / "workset.yaml", packup / lib.SNAPSHOT)
    baseline_rel = pinned.get("baseline_report")
    if baseline_rel:
        shutil.copyfile(workset_root / str(baseline_rel), packup / lib.BASELINE_REPORT)

    kernel = forge_dir / "optimized_kernel.py"
    if kernel.is_file():
        shutil.copyfile(kernel, packup / "results" / "optimized_kernel.py")
    if forge_result:
        lib.write_json(packup / "results" / "forge_result.json", forge_result)
    if correctness:
        lib.write_json(packup / "results" / "correctness_report.json", correctness)
    if performance:
        lib.write_json(packup / "results" / "performance_measured.json", performance)

    for name, template in _SKELETON.items():
        target = packup / name
        if not target.exists():
            target.write_text(
                template.format(operator=operator_id, today=today, workset_id=pinned.get("workset_id")),
                encoding="utf-8",
            )

    # --- the document -------------------------------------------------------
    baseline = {k: float(v) for k, v in (pinned.get("baseline_per_case_ms") or {}).items()}
    measured = lib.report_medians(performance, operator_id) if performance else {}

    target = pinned.get("edit_target") or {}
    kernel_path = packup / "results" / "optimized_kernel.py"
    document = {
        "schema_version": 1,
        "operator": operator_id,
        "workset_ref": {
            "handoff_id": str(Path(lib.input_content("operator_workset")).parent.parent.name),
            "version": str(Path(lib.input_content("operator_workset")).parent.name),
            "digest": None,
            "workset_id": pinned.get("workset_id"),
            "snapshot": lib.SNAPSHOT,
        },
        "premise": premise,
        "apply": {
            "mode": "overlay_files",
            "integration_point": {
                k: v for k, v in target.items()
                if k in ("source_file", "entry_function", "entry_function_line", "repo_root_var")
            },
            "files": (
                [{
                    "source": "results/optimized_kernel.py",
                    "target": str(target.get("source_file")),
                    "sha256": _sha256(kernel_path),
                    "action": "replace",
                }]
                if kernel_path.is_file() else []
            ),
            "revert": "Remove the overlay and restart the engine; nothing in the stock tree is modified in place.",
        },
        "evidence": {
            "correctness": {
                "entrypoint": (pinned["entrypoints"]["correctness"] or {}).get("cmd"),
                "report": "results/correctness_report.json",
                "passed": correctness_passed,
                "shapes": [
                    {
                        "case_id": s.get("case_id"),
                        "passed": s.get("passed") is True,
                        "snr_db": s.get("snr_db"),
                        "allclose": s.get("allclose"),
                        **({"extra": s["extra"]} if isinstance(s.get("extra"), dict) else {}),
                    }
                    for e in correctness.get("operators") or ()
                    if e.get("operator_id") == operator_id
                    for s in e.get("shapes") or ()
                ],
            },
            "performance": {
                "entrypoint": (pinned["entrypoints"]["performance"] or {}).get("cmd"),
                "protocol": pinned.get("protocol"),
                "baseline": {
                    "source": "workset",
                    "report": str(baseline_rel),
                    "per_case_ms": baseline,
                },
                "measured": {
                    "report": "results/performance_measured.json",
                    "per_case_ms": measured,
                },
            },
            "forge": {
                "ran": bool(forge_result.get("ran", not mocked)),
                "mock": mocked,
                "degraded": degraded,
                "result_json": "results/forge_result.json" if forge_result else None,
                "mean_case_speedup": forge_result.get("mean_case_speedup"),
                "improved": forge_result.get("improved"),
                "snr_db": forge_result.get("snr_db"),
                "iteration_count": forge_result.get("iteration_count"),
            },
        },
    }

    # rsd travels with the measurement: the two sides are not alike, and a
    # reader comparing a ~2% baseline against a ~8% optimised side should be
    # able to see that without re-deriving it.
    rsd = {
        str(s.get("case_id")): float(s["rsd"])
        for e in performance.get("operators") or ()
        if e.get("operator_id") == operator_id
        for s in e.get("shapes") or ()
        if isinstance(s, dict) and isinstance(s.get("rsd"), (int, float))
    }
    if rsd:
        document["evidence"]["performance"]["measured"]["rsd_per_case"] = rsd

    # --- the claim, if one may be made --------------------------------------
    refusals = []
    if not held:
        refusals.append("the premise aborted")
    if mocked:
        refusals.append("forge was mocked, so no kernel was optimised")
    if not correctness_passed:
        refusals.append("correctness did not pass")
    shared = sorted(set(baseline) & set(measured))
    if not shared:
        refusals.append("no case was measured on both sides")

    if refusals:
        print("no claim is written: " + "; ".join(refusals), file=sys.stderr)
    else:
        per_case = {c: round(baseline[c] / measured[c], 4) for c in shared if measured[c] > 0}
        noise_floor = float(
            ((pinned.get("ground_truth") or {}).get("noise_floor"))
            or (pinned.get("gates") or {}).get("noise_floor")
            or 1.05
        )
        document["evidence"]["performance"]["claim"] = {
            "speedup_per_case": per_case,
            "mean_case_speedup": round(sum(per_case.values()) / len(per_case), 4),
            "noise_floor": noise_floor,
        }

    lib.write_json(packup / lib.DOC, document)

    problems = lib.validate("kernel_optimization", document)
    if problems:
        print("the document does not validate against its own schema:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"ok: {packup}")
    print("Now write README.md, REPRODUCE.md, environment.md and notes.md — the skeletons are there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
