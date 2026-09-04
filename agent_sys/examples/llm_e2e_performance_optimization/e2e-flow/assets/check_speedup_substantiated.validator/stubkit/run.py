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
# **Where did this actually run?** The kit's real-transport mode passes 8/8 the
# same as the local mode does, so a passing run is no evidence at all that the
# entrypoint crossed into a container — and an identical pass on both sides is
# precisely the shape of a mode that silently did nothing. `/.dockerenv` is
# written by docker into every container it starts and is absent on the host,
# so this one line is the difference the exit status cannot show.
{ [ -f /.dockerenv ] && echo IN_CONTAINER || echo ON_HOST; } > "$DIR/last_where.txt"
exec python3 "$DIR/emit.py" --plan "$DIR/plan.json" --impl-path "$IMPL" \
     --side "$([ -n "$IMPL" ] && echo candidate || echo baseline)" --out "$OUT"
'''

_EMIT = r'''#!/usr/bin/env python3
"""Print a performance report the case asked for. No torch, no GPU, no timing."""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--side", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--impl-path", dest="impl_path", default="")
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
           # **`impl_path` beside `impl`, because the real harness always sets
           # both** (`_common.py:255-256`, `"impl_path": args.impl`). This stub
           # omitted it entirely, so the kit exercised a report shape production
           # cannot emit -- and once m3 bound the pair (`8ae9094`: candidate
           # REQUIRES a non-empty `impl_path`), a stub candidate report became
           # one the schema refuses. Nothing validates these reports today, so
           # it was silent; the fixture was simply wrong. m3 predicted exactly
           # this when they landed the binding -- *a second producer would not
           # have been caught, and you have one.*
           "impl_path": (a.impl_path or None) if a.side == "candidate" else None,
           "environment": {"node": "stub", "gpu_arch": "gfx950",
                           "container": "stub", "image_id": "sha256:" + "0" * 12},
           "started_at": "2026-09-03T00:00:00Z",
           "protocol": {"groups": 5, "iters_per_group": 10, "warmup": 3,
                        "timing": "wall_clock_sync"},
           "operators": [{"operator_id": plan["operator_id"], "ran": True, "shapes": shapes}]},
          open(a.out, "w"), indent=1)
'''


# **Opt-in real transport, because the leg this kit exists to cover was the one
# leg it could not reach.**
#
# Every case above stubs the entrypoint, which is right — the kit grades the
# validator's logic, not a kernel. But `runtime.transport` was pinned to
# `local`, so `_remeasure` always took the in-process branch and the code the
# leader asked to be threaded through — `_transport_env`, the wrapper routing,
# `--environment` crossing a container boundary — **ran in no test at all**.
# CONTRACT §4.4 in its plainest form: a fixture more convenient than production
# tests the fixture.
#
# Set these and the same cases run their entrypoints through
# `run_in_container.sh` into a real container on a real node:
#
#     KFO_STUBKIT_CONTAINER=<name>  KFO_STUBKIT_NODE=<host>  KFO_STUBKIT_JOBID=<id>
#     KFO_STUBKIT_ROOT=<path both hosts mount>
#
# Unset — CI, a login node, anyone without a node — nothing changes. The stub
# payload is unchanged in both modes, so a difference in outcome is a difference
# in the transport and nothing else, which is the only reason this is worth
# having as a mode rather than as a second kit.
_REAL = {
    "container": os.environ.get("KFO_STUBKIT_CONTAINER", ""),
    "node": os.environ.get("KFO_STUBKIT_NODE", ""),
    "jobid": os.environ.get("KFO_STUBKIT_JOBID", ""),
    "root": os.environ.get("KFO_STUBKIT_ROOT", ""),
}
REAL_TRANSPORT = all(_REAL[k] for k in ("container", "node", "jobid", "root"))


def _environment(node: str = "crsuse2-m2m-061") -> dict:
    runtime = {"container": "stub", "endpoint": "http://127.0.0.1:30000",
               "started_at": "2026-09-03T00:00:00Z", "transport": "local"}
    if REAL_TRANSPORT:
        node = _REAL["node"]
        # `slurm_jobid` too: `run_in_container.sh` takes it off the record when
        # the ambient is empty, and a validation zone's ambient IS empty.
        runtime.update(container=_REAL["container"], transport="spur",
                       slurm_jobid=_REAL["jobid"])
    return {
        "schema_version": 1,
        "fixed": {
            "node": node, "gpu_arch": "gfx950", "gpu_count": 8,
            "image": "infera/engine-sglang:gfx950-local", "image_id": "sha256:" + "a" * 12,
            "dockerfile": None, "rocm": "7.2.4", "torch": "2.10.0", "driver": "6.14.14",
            "model_name": "Qwen/Qwen3-0.6B", "model_path": "/shared_nfs/models/Qwen3-0.6B",
            "tp_size": 1,
        },
        "runtime": runtime,
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
        "protocol": {"groups": 5, "iters_per_group": 10, "warmup": 3,
                     "timing": "wall_clock_sync", "reduction": "weighted_mean"},
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
                # **Inert here, and a third spelling in the package.** The
                # validator this kit drives never reads `protocol` as data —
                # only `entrypoint` — so these numbers exercise nothing, and
                # `check_optimization_shape` is the sole reader anywhere,
                # comparing a copy against its own source. Recorded because a
                # reader cannot otherwise tell whether the values matter.
                #
                # **Aligned to the real workset now that m3 has landed the
                # wording** (`8ae9094` narrows the enum to `wall_clock_sync`,
                # the one thing `run_performance.sh:44-55` actually does). m2
                # found the three-way split and I held this back deliberately
                # until the owner decided, so it changed once rather than
                # twice. The counts follow for the same reason the value does:
                # a fixture whose protocol no real workset can emit is a
                # fixture testing itself.
                "protocol": {"groups": 5, "iters_per_group": 10, "warmup": 3, "timing": "wall_clock_sync"},
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
        # `baseline` requires `impl_path` to be null, not absent -- see the
        # entrypoint's note above.
        "schema_version": 1, "impl": "baseline", "impl_path": None, "generated_by": "stubkit",
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
        # In stub mode these are absent and `_transport_env` is never called.
        # In real mode they are the three `--var`s a run would supply, spelled
        # with m1's parameter names so one `--var transport_env` drives all
        # three validators.
        **({"transport_path": "/usr/local/bin",
            "transport_env": f"SPUR_CONTROLLER_ADDR={os.environ.get('SPUR_CONTROLLER_ADDR', '')}",
            "measure_gpu": "0",
            "scratch_dir": str(root / "scratch"),
            # The container has real torch, so the shim below is not needed and
            # not used: in real mode `_interpreter()` is not the thing being
            # exercised, the wrapper is.
            "timeout_seconds": 600} if REAL_TRANSPORT else {}),
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
    if REAL_TRANSPORT:
        # **Model the closed zone rather than inherit the shell.** A validator
        # declares no agent, so it gets no `SPUR_CONTROLLER_ADDR` and a `PATH`
        # without `/usr/local/bin`. m3 lost three non-reproductions to their own
        # login shell having the variable; this kit would inherit it from mine
        # for exactly the same reason. Stripped here so the only way the
        # transport can work is the `transport_path` / `transport_env` args
        # above — which is the mechanism under test.
        environment.pop("SPUR_CONTROLLER_ADDR", None)
        environment["PATH"] = "/usr/bin:/bin"
        # No torch shim and no `KFO_PYTHON`: the container carries a real torch,
        # and the wrapper supplies the interpreter inside it. The kit's declared
        # exception at the top of this file does not apply in this mode.
    else:
        shim = root / "shim"
        shim.mkdir(exist_ok=True)
        (shim / "torch.py").write_text('__version__ = "0.0.0+stubkit"\n')
        environment["PYTHONPATH"] = str(shim) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["KFO_PYTHON"] = sys.executable
    environment["TMPDIR"] = str(root / "tmp")
    # **The zone declares a card, because a real `cost: gpu_hours` zone has
    # one.** `_remeasure` now refuses when `HIP_VISIBLE_DEVICES` is unset —
    # unset does not mean "the caller chose", it means torch takes card 0, which
    # on a shared host is a co-tenant's. This kit stubs the entrypoint entirely
    # so no card is touched; the variable is here to reproduce the zone's
    # *precondition*, not to reserve anything.
    #
    # Declared with the other exception at the top of this file rather than
    # slipped in: a fixture that satisfies a guard it is not modelling is how a
    # kit stops testing the thing it was built for.
    environment["HIP_VISIBLE_DEVICES"] = "0"
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
    ran_in_container = False
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
        # In real mode every path here has to resolve **inside the container**
        # too: `_remeasure` copies the apparatus on the host and the entrypoint
        # then runs on the far side of a `docker exec`. The login node's `/tmp`
        # is not the node's, so `KFO_STUBKIT_ROOT` must name something both
        # hosts mount and the container has bind-mounted.
        root = Path(tempfile.mkdtemp(prefix=f"stubkit-{index}-",
                                     dir=_REAL["root"] if REAL_TRANSPORT else None))
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

            # Real mode only, and it has to be checked here because `root` is
            # removed below. See `last_where.txt` in `_ENTRYPOINT`.
            for where in sorted(root.rglob("last_where.txt")):
                text = where.read_text().strip()
                ran_in_container = ran_in_container or text == "IN_CONTAINER"
                if REAL_TRANSPORT and text != "IN_CONTAINER":
                    failures.append(
                        f"case {index}: the entrypoint ran {text}, not in "
                        f"{_REAL['container']}. Real-transport mode asked for the container "
                        f"path and got the local one\n  {label}")
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

    # **The same rule for the mode itself.** Real mode passed 8/8 on its first
    # run and so did local mode, which is no evidence whatsoever that anything
    # crossed a container boundary — the two modes are indistinguishable by exit
    # status alone. It took pointing the mode at a container that does not exist
    # to learn that the path was in fact being taken. That falsification should
    # not have to be run by hand every time.
    if REAL_TRANSPORT and not ran_in_container:
        failures.append(
            f"real-transport mode was requested for {_REAL['container']} on {_REAL['node']} "
            "but no case ever reached it, so every case above graded the LOCAL path under a "
            "name that claims otherwise")

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
