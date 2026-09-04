#!/usr/bin/env python3
"""Drive `check_speedup_substantiated`'s re-measurement path. See README.md.

Every case builds a whole `kernel_optimization` handoff and a whole stub
workset on disk, lays out a validation zone the way `validator/phase.py` lays
one out (`args.json`, `inputs.json`, `materials.json` in the cwd, `verdict.json`
owed back), and runs the **real validator body** in it. Nothing is imported and
monkeypatched: the thing under test is the file that will grade a real handoff.

One exception, and it is declared rather than hidden: `KFO_PYTHON` is pointed at
this interpreter so that `_interpreter()` finds one. On a host without torch it
would otherwise find none and every case would fail identically, testing the
probe rather than the path. The probe's own trap stays untested here — README.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent.parent
VALIDATOR = PACKAGE / "assets" / "check_speedup_substantiated.validator" / "check.py"

OPERATOR = "sampler_vocab_softmax"
CASES = ("B1_V151936", "B8_V151936", "B32_V151936")

#: The workset's recorded baseline, in ms. The numbers are the sealed stage-4
#: run's own gfx950 measurements, so that a reader comparing this fixture
#: against the real artefact sees the same figures rather than round numbers
#: that look invented.
BASELINE = {"B1_V151936": 0.048960, "B8_V151936": 0.050201, "B32_V151936": 0.058201}
NOISE_FLOOR = 1.057

#: `run_performance.sh`, in the shape m3 shipped:
#:   ./run_performance.sh [--operator ID] [--impl PATH] [--shape CASE_ID] [--json OUT]
#: No `--impl` is the baseline; `--impl PATH` is the candidate. The stub reads
#: what to print out of `plan.json` beside it, so a case can ask for a
#: disagreement, a crash, or a malformed report without editing this script.
_ENTRYPOINT = r'''#!/bin/sh
set -eu
DIR=$(cd "$(dirname "$0")" && pwd)
IMPL=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --impl) IMPL="$2"; shift 2 ;;
    --json) OUT="$2"; shift 2 ;;
    # **Recorded, not discarded.** `*) shift` below tolerates any flag in
    # silence, which is what let this kit report 8/8 over a `--environment`
    # that was never passed: the fixture accepted its absence exactly as
    # happily as its presence. m3's real harness compares this record against
    # `ground_truth.environment`, so a kit that cannot see whether it arrived
    # cannot test the gate that depends on it.
    --environment) echo "$2" > "$DIR/last_environment.txt"; shift 2 ;;
    --operator|--shape) shift 2 ;;
    *) shift ;;
  esac
done
# **A bare `python3`, deliberately.** An earlier version said
# `${KFO_PYTHON:-python3}` and masked a real bug: it honoured a variable the
# validator was not setting, so the kit could not see that the chosen
# interpreter never reached the entrypoint. A real workset script says
# `python3`, so this one does too.
exec python3 "$DIR/emit.py" --plan "$DIR/plan.json" \
     --side "$([ -n "$IMPL" ] && echo candidate || echo baseline)" --out "$OUT"
'''

_EMIT = r'''#!/usr/bin/env python3
"""Print a performance report the case asked for. No torch, no GPU, no timing."""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--side", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

plan = json.load(open(a.plan))
side = plan[a.side]

if side.get("exit_nonzero"):
    print(side.get("stderr", "the harness died"), file=sys.stderr)
    raise SystemExit(int(side["exit_nonzero"]))
if side.get("write_nothing"):
    raise SystemExit(0)

shapes = []
for case_id, ms in side["per_case_ms"].items():
    shape = {"case_id": case_id, "groups": 5, "iters_total": 150,
             "per_group_ms": [ms] * 5, "rsd": 0.02}
    if case_id not in side.get("omit_weighted_mean_for", []):
        shape["weighted_mean_ms"] = ms
    shapes.append(shape)

json.dump({"schema_version": 1, "generated_by": "stubkit",
           "impl": "candidate" if a.side == "candidate" else "baseline",
           "environment": {"node": "stub", "gpu_arch": "gfx950",
                           "container": "stub", "image_id": "sha256:" + "0" * 12},
           "started_at": "2026-09-03T00:00:00Z",
           "protocol": {"groups": 5, "iters_per_group": 30, "warmup": 10,
                        "timing": "hip_graph_replay"},
           "operators": [{"operator_id": plan["operator_id"], "ran": True, "shapes": shapes}]},
          open(a.out, "w"), indent=1)
'''


def _environment(node: str = "crsuse2-m2m-061") -> dict:
    return {
        "schema_version": 1,
        "fixed": {
            "node": node, "gpu_arch": "gfx950", "gpu_count": 8,
            "image": "infera/engine-sglang:gfx950-local", "image_id": "sha256:" + "a" * 12,
            "dockerfile": None, "rocm": "7.2.4", "torch": "2.10.0", "driver": "6.14.14",
            "model_name": "Qwen/Qwen3-0.6B", "model_path": "/shared_nfs/models/Qwen3-0.6B",
            "tp_size": 1,
        },
        "runtime": {"container": "stub", "endpoint": "http://127.0.0.1:30000",
                    "started_at": "2026-09-03T00:00:00Z", "transport": "local"},
    }


def _workset() -> dict:
    return {
        "schema_version": 1, "workset_id": "stub.workset",
        "produced_by": "stubkit",
        "ground_truth": {"abort_on_mismatch": ["gpu_arch", "gpu_count", "dtype"],
                         "warn_on_mismatch": ["rocm", "torch"],
                         "environment": _environment(),
                         "dtypes": {OPERATOR: "fp32"},
                         "noise_floor": NOISE_FLOOR},
        "protocol": {"groups": 5, "iters_per_group": 30, "warmup": 10,
                     "timing": "hip_graph_replay", "reduction": "weighted_mean"},
        # The flag spelling as data, which is what the validator now reads
        # instead of its own `args` -- one declared source beats two agreeing
        # copies.
        "entrypoints": {"correctness": {"cmd": "./run_correctness.sh", "report": "evidence/correctness.json",
                                        "flags": {"operator": "--operator", "shape": "--shape",
                                                  "impl": "--impl", "report": "--json"}},
                        "performance": {"cmd": "./run_performance.sh", "report": "evidence/performance.json",
                                        "flags": {"operator": "--operator", "shape": "--shape",
                                                  "impl": "--impl", "report": "--json"}}},
        "evidence": {"correctness_report": "evidence/correctness.json",
                     "performance_report": "evidence/performance.json",
                     "measured_on": {"node": "stub", "gpu_arch": "gfx950",
                                     "container": "stub", "at": "2026-09-03T00:00:00Z"}},
        "operators": [{
            "operator_id": OPERATOR, "op_type": "softmax", "status": "complete",
            "missing_fields": [], "definition": "definitions/softmax/x.json",
            "workload": "workloads/softmax/x.jsonl",
            "shapes": [{"case_id": c, "uuid": f"u{i}", "axes": {"batch": 1 << (3 * i)},
                        "role": "correctness-and-performance", "observed": True}
                       for i, c in enumerate(CASES)],
            "entrypoints": {"correctness": {"cmd": "./run_correctness.sh", "report": "evidence/correctness.json"},
                            "performance": {"cmd": "./run_performance.sh", "report": "evidence/performance.json"}},
            "reference": {"kind": "written"}, "baseline": {"kind": "imported"},
            "edit_target": {"source_file": "python/sglang/srt/layers/sampler.py",
                            "entry_function": "sampler_softmax", "repo_root_var": "@SGLANG_ROOT@"},
            "integration": {"target_files": ["python/sglang/srt/layers/sampler.py"],
                            "public_symbol": "sampler_softmax",
                            "invariants": ["writes in place into out"],
                            "apply_mode": "overlay_files"},
            "gates": {"snr_db": 30.0}, "noise_floor": NOISE_FLOOR,
            "apparatus": ["run_correctness.sh", "run_performance.sh", "emit.py", "plan.json"],
            "provenance": {"source": "stub", "magpie_row": {
                "Name": "stub", "Calls": 1, "Avg time (us)": 55.59,
                "% Total": 14.5, "Input Shapes": "[[8, 151936]]"}},
        }],
    }


def _document(claim_speedup: float | None, measured: dict, *, noise_floor=NOISE_FLOOR,
              baseline: dict | None = None) -> dict:
    baseline = baseline if baseline is not None else BASELINE
    doc = {
        "schema_version": 1, "operator": OPERATOR,
        "workset_ref": {"handoff_id": "stub", "version": "v0", "digest": None,
                        "workset_id": "stub.workset", "snapshot": "results/workset.snapshot.yaml"},
        "premise": {"abort_on_mismatch": ["gpu_arch", "gpu_count"],
                    "warn_on_mismatch": ["rocm", "torch"],
                    "workset_environment": _environment(),
                    "run_environment": _environment(),
                    "dtypes": {OPERATOR: "fp32"},
                    "verdict": {"held": True, "aborted_on": [], "warnings": []}},
        "apply": {"apply_mode": "overlay_files", "manifest": "apply/manifest.json",
                  "integration_point": {"source_file": "python/sglang/srt/layers/sampler.py",
                                        "entry_function": "sampler_softmax"},
                  "files": [{"container_path": "@SGLANG_ROOT@/srt/layers/sampler.py",
                             "base_sha256": "0" * 64, "change": "modify",
                             "replacement": "results/optimized_kernel.py"}]},
        "evidence": {
            "correctness": {"entrypoint": "./run_correctness.sh", "report": "results/correctness_report.json",
                            "passed": True,
                            "shapes": [{"case_id": c, "passed": True} for c in CASES]},
            "performance": {
                "entrypoint": "./run_performance.sh",
                "protocol": {"groups": 5, "iters_per_group": 30, "warmup": 10, "timing": "hip_graph_replay"},
                "baseline": {"source": "workset", "report": "evidence/performance.json",
                             "per_case_ms": baseline},
                "measured": {"report": "results/performance_measured.json", "per_case_ms": measured},
            },
            "forge": {"ran": True, "mock": False, "degraded": False,
                      "result_json": "results/forge_result.json", "mean_case_speedup": claim_speedup},
        },
    }
    if claim_speedup is not None:
        per_case = {c: round(baseline[c] / measured[c], 4) for c in measured}
        claim = {"speedup_per_case": per_case,
                 "mean_case_speedup": round(sum(per_case.values()) / len(per_case), 4)}
        if noise_floor is not None:
            claim["noise_floor"] = noise_floor
        doc["evidence"]["performance"]["claim"] = claim
    return doc


def _build(root: Path, *, document: dict, plan: dict) -> Path:
    """A handoff `content/` with a stub workset inside it as its apparatus."""
    packup = root / "handoff" / "items" / "codes" / f"{OPERATOR}.packup_20260903"
    (packup / "results").mkdir(parents=True, exist_ok=True)
    (packup / "apply").mkdir(parents=True, exist_ok=True)
    apparatus = packup / "scripts" / "workset"
    (apparatus / "evidence").mkdir(parents=True, exist_ok=True)

    (apparatus / "run_performance.sh").write_text(_ENTRYPOINT, encoding="utf-8")
    (apparatus / "run_correctness.sh").write_text(_ENTRYPOINT, encoding="utf-8")
    (apparatus / "emit.py").write_text(_EMIT, encoding="utf-8")
    for name in ("run_performance.sh", "run_correctness.sh", "emit.py"):
        (apparatus / name).chmod(0o755)
    (apparatus / "plan.json").write_text(json.dumps({"operator_id": OPERATOR, **plan}, indent=1))

    (packup / "results" / "kernel_optimization.json").write_text(json.dumps(document, indent=2))
    (packup / "results" / "workset.snapshot.yaml").write_text(
        __import__("yaml").safe_dump(_workset()), encoding="utf-8")
    (packup / "results" / "workset.baseline_report.json").write_text(json.dumps({
        "schema_version": 1, "impl": "baseline", "generated_by": "stubkit",
        "operators": [{"operator_id": OPERATOR, "ran": True, "shapes": [
            {"case_id": c, "groups": 5, "iters_total": 150, "per_group_ms": [BASELINE[c]] * 5,
             "weighted_mean_ms": BASELINE[c], "rsd": 0.02} for c in CASES]}],
    }, indent=1))
    (packup / "results" / "optimized_kernel.py").write_text("# stub candidate\n")

    # **This run's record, beside the packup** — CONTRACT §2's place for it in a
    # `code` handoff, and the document `_remeasure` hands the entrypoint as
    # `--environment`. Without it here the validator resolves `record = None`,
    # appends no flag, and every case below passes without the new path
    # executing at all: measured, 8/8 green over code that never ran.
    (packup.parent / "environment.yaml").write_text(
        __import__("yaml").safe_dump(_environment()), encoding="utf-8")
    return root / "handoff"


def _run(root: Path, handoff: Path) -> tuple[str, bool | None]:
    zone = root / "zone"
    zone.mkdir(parents=True, exist_ok=True)
    (zone / "args.json").write_text(json.dumps({
        "schema": "kernel_optimization",
        "abort_on_premise_mismatch": ["gpu_arch"], "warn_on_mismatch": ["rocm"],
        "require_correctness_pass": True, "min_shapes_measured": 3,
        "baseline_agreement_tolerance": 0.05, "tolerance": 0.15, "timeout_seconds": 120,
        "report_flag": "--json", "impl_flag": "--impl",
    }))
    (zone / "inputs.json").write_text('["stub"]')
    (zone / "materials.json").write_text(json.dumps({"stub": os.path.relpath(handoff, zone)}))

    environment = dict(os.environ)
    environment["AGENT_SYS_TASK_PACKAGE"] = str(PACKAGE)
    # **Declared, not hidden.** `_interpreter()` does not take an interpreter on
    # trust — it runs `import torch` in each candidate and rejects one that
    # cannot. That is correct and it is why the first run of this kit failed all
    # seven cases identically on a torch-less login node, testing the probe
    # rather than the path.
    #
    # So the kit puts a two-line `torch.py` on PYTHONPATH. The probe then
    # selects this interpreter, which is the state a real node is in for real
    # reasons. It proves nothing about tensor math and the stub entrypoints do
    # no arithmetic, so nothing here silently depends on it.
    shim = root / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "torch.py").write_text('__version__ = "0.0.0+stubkit"\n')
    environment["PYTHONPATH"] = str(shim) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["KFO_PYTHON"] = sys.executable
    environment["TMPDIR"] = str(root / "tmp")
    (root / "tmp").mkdir(exist_ok=True)
    proc = subprocess.run([sys.executable, str(VALIDATOR)], cwd=zone, env=environment,
                          capture_output=True, text=True, timeout=300)
    output = proc.stdout + proc.stderr
    verdict = None
    if (zone / "verdict.json").is_file():
        verdict = json.loads((zone / "verdict.json").read_text()).get("stub")
    return output, verdict


def main() -> int:
    faster = {c: BASELINE[c] / 2.8 for c in CASES}
    barely = {c: BASELINE[c] / 1.01 for c in CASES}

    cases: list[tuple[str, dict, dict, bool, str]] = [
        (
            "an honest claim: 2.8x claimed, 2.8x re-measured",
            _document(2.8, faster),
            {"baseline": {"per_case_ms": BASELINE}, "candidate": {"per_case_ms": faster}},
            True, "",
        ),
        (
            "the seed disagrees with the workset's recorded baseline — the premise "
            "did not hold empirically, and under the deleted rule this was a silent re-baseline",
            _document(2.8, faster),
            {"baseline": {"per_case_ms": {c: v * 1.30 for c, v in BASELINE.items()}},
             "candidate": {"per_case_ms": faster}},
            False, "ABORT",
        ),
        (
            "run_performance.sh exits non-zero on the candidate — a crashed measurement "
            "must not read as a zero-speedup measurement",
            _document(2.8, faster),
            {"baseline": {"per_case_ms": BASELINE},
             "candidate": {"exit_nonzero": 1, "stderr": "HIP error: invalid device function"}},
            False, "re-measurement failed",
        ),
        (
            "the report omits weighted_mean_ms for one case — a case silently dropped "
            "from the mean is a candidate averaged over the shapes it happened to win",
            _document(2.8, faster),
            {"baseline": {"per_case_ms": BASELINE},
             "candidate": {"per_case_ms": faster, "omit_weighted_mean_for": ["B32_V151936"]}},
            False, "B32_V151936",
        ),
        (
            "the producer over-claims: 2.8x claimed, 1.1x re-measured",
            _document(2.8, faster),
            {"baseline": {"per_case_ms": BASELINE},
             "candidate": {"per_case_ms": {c: BASELINE[c] / 1.1 for c in CASES}}},
            False, "below the claimed",
        ),
        (
            "1.01x against a 1.057 noise floor — noise reported as a win",
            _document(1.01, barely),
            {"baseline": {"per_case_ms": BASELINE}, "candidate": {"per_case_ms": barely}},
            False, "noise floor",
        ),
        (
            "the claim carries no noise_floor — the consumer would be picking its own "
            "significance threshold",
            _document(2.8, faster, noise_floor=None),
            {"baseline": {"per_case_ms": BASELINE}, "candidate": {"per_case_ms": faster}},
            False, "noise_floor",
        ),
    ]

    wrong_dtype = _document(2.8, faster)
    wrong_dtype["premise"]["dtypes"] = {OPERATOR: "bf16"}
    cases.append((
        "the handoff optimised bf16 where the workset's ground truth says fp32 — "
        "a speedup at a different precision is a different question",
        wrong_dtype,
        {"baseline": {"per_case_ms": BASELINE}, "candidate": {"per_case_ms": faster}},
        False, "dtype",
    ))

    failures: list[str] = []
    landed_any = False
    for index, (label, document, plan, expect_pass, expect_text) in enumerate(cases, start=1):
        # **A case that expects a refusal must name what the refusal is about.**
        # Without this, a case passes on ANY failure — and two of these did
        # exactly that on the first run of this kit, on a missing interpreter
        # that had nothing to do with what they claim to test. Checked before
        # anything is built, because it is a fault in the case and not in the
        # validator.
        if not expect_pass and not expect_text:
            failures.append(f"case {index}: expects a refusal but names no reason to look for")
            continue
        root = Path(tempfile.mkdtemp(prefix=f"stubkit-{index}-"))
        try:
            handoff = _build(root, document=document, plan=plan)
            output, verdict = _run(root, handoff)
            if verdict is not expect_pass:
                failures.append(
                    f"case {index}: expected verdict {expect_pass}, got {verdict}\n"
                    f"  {label}\n" + "\n".join("    " + l for l in output.splitlines()[:12]))
            elif expect_text and expect_text not in output:
                failures.append(
                    f"case {index}: verdict right but never said {expect_text!r}\n"
                    f"  {label}\n" + "\n".join("    " + l for l in output.splitlines()[:12]))
            else:
                print(f"ok   case {index}: {label}")

            # **Did the re-measurement hand over this run's record, and the
            # right one?** Only checked where the entrypoint actually ran —
            # several cases refuse before `_remeasure` is reached, and demanding
            # the flag there would fail them for not doing something they never
            # got to. The whole-kit assertion below is what keeps this able to
            # fail: if NO case lands the record, the flag is not being passed at
            # all and the gate it feeds is untested.
            landed = sorted(root.rglob("last_environment.txt"))
            if landed:
                landed_any = True
                got = landed[0].read_text().strip()
                expected = str((handoff / "items" / "codes" / "environment.yaml").resolve())
                if got != expected:
                    failures.append(
                        f"case {index}: --environment pointed at {got!r}, expected the handoff's "
                        f"own record {expected!r}. A relative path is the live form of this: the "
                        f"entrypoint runs under <scratch>/seed, not the validation zone\n  {label}")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # **The kit as a whole must have exercised the flag at least once.** Without
    # this the per-case check above is vacuous: with `--environment` removed
    # entirely, every case simply stops landing a file and nothing complains.
    # Measured — that is exactly how this kit reported 8/8 over code that never
    # ran, before the record was added to the fixture.
    if not landed_any:
        failures.append(
            "no case handed the entrypoint --environment, so m3's M4.3.5 gate was never "
            "exercised by this kit. Either _remeasure stopped passing it or the fixture "
            "stopped carrying items/codes/environment.yaml")

    if failures:
        print()
        for failure in failures:
            print("FAIL " + failure)
        print(f"\n{len(failures)} of {len(cases)} failed")
        return 1
    print(f"\n{len(cases)} cases passed")
    print("NOTE: this is a harness test. Every number above was one the stub was told "
          "to print; nothing here measures a kernel. See README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
