#!/usr/bin/env python3
"""Assemble the `sampler_vocab_softmax` operator_workset fixture (gfx942).

    build_fixture.py assemble --e2e <e2e-flow dir> --out <content dir> --facts <facts.json>
    build_fixture.py transcribe --out <content dir> --container <name>

`assemble` writes everything that is NOT a measurement: the byte-identical
harness copy, the Definition, the Workload, the environment record,
`workset.yaml`, the README, and the KernelForge add-on (by running e2e-flow's
`forge_export.py`, never by hand). Its shape follows e2e-flow's
`build_workset.task/mock_adapt.py`; the inputs mock_adapt reads out of a sealed
corpus are supplied here from `--facts`, a JSON file of facts probed inside the
engine image (see the l3-fixture notes for the probe commands).

`transcribe` runs after `measure_in_container.sh` has written `evidence/`. It
copies the measured `noise_floor` into every operator and into
`ground_truth.noise_floor`, and writes the `evidence` block. That is STEP 8 of
`build_workset.task/readme.md`.

Nothing here invents a number: every figure comes from `--facts` or from
`evidence/`, and the script refuses when one is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

OPERATOR = "sampler_vocab_softmax"
TARGET_FILE = "python/sglang/srt/layers/sampler.py"
#: (case_id suffix batch). B32 is the primary shape: a busy decode batch, chosen
#: here, not observed in a trace -- there is no in-service profile behind this fixture.
#: **No B1.** At one row the naive incumbent is ~0.16 ms and partly launch-bound;
#: re-measured, it came back 9.5 % off the recorded figure in two of the level-3
#: runs, and `check_speedup_substantiated` aborts beyond 5 %. 16-64 rows keep
#: every shape GPU-bound.
BATCHES = (16, 32, 64)
PRIMARY_BATCH = 32

REFERENCE = '''from __future__ import annotations
import torch


def _reference(logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax in float64. Slow and unambiguous."""
    out.copy_(torch.softmax(logits.double(), dim=-1).to(out.dtype))
    return out


# ----- entry point -----
def run(*args, **kwargs):
    return _reference(*args, **kwargs)
'''

#: **The demo's incumbent is deliberately naive**, so that a one-hour campaign
#: has something to win. The engine's own `torch.softmax` is a single well-tuned
#: pass and measured as such it left forge nothing (4 iterations, 0.990x-0.9997x,
#: all reverted). This one takes each row's max by **sorting the row**, one row at
#: a time -- a full sort where a reduction would do.
#:
#: **Slow on the GPU, not on the host.** A first naive version walked the row in
#: 8192-wide chunks; it was launch-bound, and its eager timing moved 5-10 %
#: between runs, so `check_speedup_substantiated` refused it for not reproducing
#: the workset's own baseline within 5 %. Measured for this one: run-to-run
#: within about 3 % at B8-B32 across fresh containers.
BASELINE = '''from __future__ import annotations
import torch


def sampler_softmax(logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """A deliberately naive softmax: one row at a time, the max found by sorting.

    Writes in place into `out`, which may be `logits` itself (the engine's call
    site is `sampler_softmax(logits, logits)`): a row is read in full before it
    is written.
    """
    for b in range(logits.shape[0]):
        row = logits[b]
        m = torch.sort(row).values[-1]
        e = torch.exp(row - m)
        out[b] = e / e.sum()
    return out


# ----- entry point -----
def run(*args, **kwargs):
    return sampler_softmax(*args, **kwargs)
'''

README = """# operator_workset: sampler_vocab_softmax on gfx942

## Purpose

A runnable optimisation workset for one operator, `sampler_vocab_softmax`: the
row-wise fp32 softmax over the vocabulary that sglang's `Sampler.forward` runs on
every decode step (`logits[:] = torch.softmax(logits, dim=-1)`), at Qwen3.6-27B's
vocabulary of {vocab}. It is the input to e2e-flow's m4 `optimize_kernel.task`
and is used by the level-3 demo (`examples/03_kernelforge_one_kernel`).

It was assembled by hand (`build_fixture.py`, after e2e-flow's
`build_workset.task/mock_adapt.py`) rather than produced by m3's `build_workset`,
and its evidence was measured once on GPU {gpu} of {node} (8x MI300X, gfx942) in
`{image}`.

## Interface

`items/codes/workset.yaml` is the one document a consumer reads. It indexes the
operator and, for it, the flashinfer-bench `definition` and `workload` paths,
the `shapes` list, the two `entrypoints`, the `gates`, the measured
`noise_floor`, the `integration` point and the `apparatus` list.

The two entrypoints are run from `items/codes/`:

    ./run_correctness.sh --operator sampler_vocab_softmax [--shape CASE_ID] [--impl PATH] --json <report>
    ./run_performance.sh --operator sampler_vocab_softmax [--shape CASE_ID] [--impl PATH] --json <report>

Both are protected: `run_correctness.sh`, `run_performance.sh` and `_common.py`
are byte-identical copies of e2e-flow's `build_workset.task/harness/`. `evidence/`
holds the reports this workset's own run produced, and `check_workset_runs`
re-runs the primary shape and compares. `operators/sampler_vocab_softmax/` is the
KernelForge add-on generated by e2e-flow's `assets/lib/forge_export.py`.

## Boundary

This is a fixture, not an m3 output, and several fields say less than a real
workset's would.

- **No in-service profile.** No profile of a served Qwen3.6-27B exists on this
  host. `provenance.magpie_row` comes from a standalone `torch.profiler` capture
  of the baseline call at the primary shape, so its `% Total` is the softmax
  kernel's share of that isolated capture, not of a decode step. Every shape is
  `observed: false`; B32 is primary by choice, not because a trace showed it.
- **No identity.** `edit_target.from_identity` is null: no `operator_identity`
  resolved this operator, so `check_workset_shape` reports the source file as
  UNVERIFIED. The file and line were read out of the image by hand.
- **Baseline only.** `evidence/` measures the incumbent against the reference;
  there is no candidate and no speedup here.
- **A naive incumbent, on purpose.** The baseline is not the engine's
  `torch.softmax` but a row-at-a-time softmax that sorts each row to find its
  max, written for the demo, so
  that a short campaign has something to win. m4 installs it at the call site
  (`integration.call_site`) before forge starts.
- **One machine.** The numbers are for gfx942 in the image named by
  `environment.yaml`. `ground_truth.abort_on_mismatch` stops a measurement whose
  environment record disagrees on gpu_arch, gpu_count, tp_size or dtype.
- **No server.** The environment record's `runtime.endpoint` names no running
  engine; the operator was measured standalone in a fresh container.
- **Fellow chosen by hand.** The Definition's tags name `triton` as the single
  KernelForge backend; that is a choice for this demo, not something derived.
"""


def _die(message: str) -> None:
    sys.exit(f"build_fixture: {message}")


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def assemble(e2e: Path, content: Path, facts: dict) -> None:
    sys.path.insert(0, str(e2e / "assets" / "lib"))
    import workset_io as W  # noqa: PLC0415

    for key in (
        "vocab",
        "image",
        "image_id",
        "node",
        "gpu_count",
        "sampler_sha256",
        "module_symbols",
        "rocm",
        "torch",
        "sglang",
        "measure_container",
        "measure_gpu",
        "model_name",
        "model_path",
        "profile_primary",
    ):
        if key not in facts:
            _die(f"--facts carries no {key!r}")
    vocab = int(facts["vocab"])
    root = content / "items" / "codes"
    root.mkdir(parents=True, exist_ok=True)

    # 1. The harness, copied byte for byte (scaffold.py's rule; never authored).
    harness = e2e / "assets" / "build_workset.task" / "harness"
    for name in ("_common.py", "run_correctness.sh", "run_performance.sh"):
        shutil.copy2(harness / name, root / name)
        if name.endswith(".sh"):
            (root / name).chmod(0o755)

    # 2. The Definition.
    _write_json(
        root / f"definitions/softmax/{OPERATOR}.json",
        {
            "name": OPERATOR,
            "op_type": "softmax",
            "axes": {
                "batch": {
                    "type": "var",
                    "description": "Concurrent decode requests; rows of the logits tensor.",
                },
                "vocab": {
                    "type": "const",
                    "value": vocab,
                    "description": "Vocabulary size of Qwen3.6-27B (config.json vocab_size). Fixed by the model.",
                },
            },
            "inputs": {
                "logits": {
                    "shape": ["batch", "vocab"],
                    "dtype": "float32",
                    "description": "Raw logits.",
                },
                "out": {
                    "shape": ["batch", "vocab"],
                    "dtype": "float32",
                    "description": "Pre-allocated output. The production call site is `logits[:] = ...`, so a replacement that allocates is not substitutable.",
                },
            },
            "outputs": {
                "out": {
                    "shape": ["batch", "vocab"],
                    "dtype": "float32",
                    "description": "Probabilities.",
                }
            },
            "reference": REFERENCE,
            "baseline": BASELINE,
            "tags": ["softmax", "sglang", "triton"],
            "description": "Row-wise softmax over the vocabulary, written in place into a caller-provided buffer.",
        },
    )

    # 3. The Workload and the shapes index, in the same order.
    shapes, lines = [], []
    for index, batch in enumerate(BATCHES):
        uuid = f"{index:016x}"
        case_id = f"B{batch}_V{vocab}"
        shapes.append(
            {
                "case_id": case_id,
                "uuid": uuid,
                "axes": {"batch": batch},
                "role": None,
                "is_primary": batch == PRIMARY_BATCH,
                "observed": False,
                "note": (
                    "primary by choice: a typical decode batch. No in-service trace exists for this fixture"
                    if batch == PRIMARY_BATCH
                    else "chosen to bracket the primary batch; not observed in a trace"
                ),
            }
        )
        lines.append(
            json.dumps(
                {
                    "definition": OPERATOR,
                    "workload": {
                        "axes": {"batch": batch},
                        "inputs": {"logits": {"type": "random"}, "out": {"type": "random"}},
                        "uuid": uuid,
                    },
                    "solution": None,
                    "evaluation": None,
                }
            )
        )
    (root / "workloads/softmax").mkdir(parents=True, exist_ok=True)
    (root / f"workloads/softmax/{OPERATOR}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    for shape, role in zip(shapes, W.assign_roles(shapes)):
        shape["role"] = role

    # 4. The environment record, through e2e-flow's own renderer (validates first).
    env = dict(os.environ)
    env.update(
        {
            "E2E_NODE": facts["node"],
            "E2E_IMAGE": facts["image"],
            "E2E_MODEL_NAME": facts["model_name"],
            "E2E_MODEL_PATH": facts["model_path"],
            "E2E_CONTAINER": facts["measure_container"],
            "E2E_TRANSPORT": "local",
            "E2E_TP": "1",
        }
    )
    for var in (
        "E2E_NODE_IP",
        "E2E_SERVED_NAME",
        "E2E_DEPLOY_MODE",
        "E2E_JOBID",
        "E2E_CTX",
        "E2E_PORT_ROUTER",
        "E2E_PORT_WORKER",
        "E2E_PORT_ETCD",
    ):
        env.pop(var, None)
    if (root / "environment.yaml").exists():
        (root / "environment.yaml").unlink()
    rendered = subprocess.run(
        [
            sys.executable,
            str(e2e / "assets/lib/env_render.py"),
            "--new",
            "--content-type",
            "code",
            "--out",
            str(content),
            "--set",
            "fixed.gpu_arch=gfx942",
            "--set",
            f"fixed.gpu_count={facts['gpu_count']}",
            "--set",
            f"fixed.image_id={facts['image_id']}",
            "--set",
            f"fixed.rocm={facts['rocm']}",
            "--set",
            f"fixed.torch={facts['torch']}",
            "--set",
            f"fixed.sglang={facts['sglang']}",
            "--set",
            "runtime.endpoint=http://127.0.0.1:30000",
            "--set",
            (
                "runtime.notes=Fixture record for a standalone operator measurement. No engine was "
                "served; endpoint names no running server. container is the ephemeral measurement "
                f"container (GPU {facts['measure_gpu']}, --rm)."
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if rendered.returncode != 0:
        _die(f"env_render failed:\n{rendered.stderr[-1500:]}")
    environment = yaml.safe_load((root / "environment.yaml").read_text(encoding="utf-8"))

    # 5. workset.yaml.
    def entrypoints(operator_id=None):
        suffix = f" --operator {operator_id}" if operator_id else ""
        tag = operator_id or "all"
        return {
            "correctness": {
                "cmd": f"./run_correctness.sh{suffix}",
                "report": f"evidence/correctness.{tag}.json",
                "protected": True,
            },
            "performance": {
                "cmd": f"./run_performance.sh{suffix}",
                "report": f"evidence/performance.{tag}.json",
                "protected": True,
                "timeout_s": 1800,
                "impl_contract": W.IMPL_CONTRACT,
            },
        }

    prof = facts["profile_primary"]
    document = {
        "schema_version": 1,
        "workset_id": "fixture.sampler-vocab-softmax.gfx942",
        "produced_by": {
            "package": "agent_sys/examples/03_kernelforge_one_kernel",
            "commit": facts.get("commit", "unknown"),
            "step": "hand-assembled fixture (assets/fixture/build_fixture.py), measured by e2e-flow build_workset.task/measure_in_container.sh",
        },
        "ground_truth": {
            "abort_on_mismatch": ["gpu_arch", "gpu_count", "tp_size", "dtype"],
            "warn_on_mismatch": ["image_id", "rocm", "torch"],
            "dtypes": {OPERATOR: "float32"},
            "noise_floor": None,  # transcribed from evidence/ by `transcribe`
            "environment": environment,
        },
        "entrypoints": entrypoints(),
        "protocol": {"groups": 5, "iters_per_group": 50, "warmup": 50, "timing": "wall_clock_sync"},
        "operators": [
            {
                "operator_id": OPERATOR,
                "device_symbol": prof["name"],
                "op_type": "softmax",
                "status": "complete",
                "missing_fields": [],
                "definition": f"definitions/softmax/{OPERATOR}.json",
                "workload": f"workloads/softmax/{OPERATOR}.jsonl",
                "shapes": shapes,
                "entrypoints": entrypoints(OPERATOR),
                "reference": {
                    "kind": "imported",
                    "module": "torch",
                    "symbol": "softmax",
                    "source_file": "torch/functional.py",
                    "rationale": "torch.softmax evaluated at float64. Imported rather than written: a hand-written float64 softmax must reproduce the max-subtraction trick or overflow at this vocabulary, and if it reproduces it then it is the same algorithm under test. The wrapper casts and writes into `out`; the algorithm is torch's.",
                },
                "baseline": {
                    "kind": "written",
                    "path": f"definitions/softmax/{OPERATOR}.json",
                    "rationale": "A deliberately naive softmax written for the demo (the Definition's `baseline`): one row at a time, its max found by a full sort. The engine's own `torch.softmax` gave a one-hour campaign nothing to win; this gives it a slow incumbent with an obvious fix. m4 installs it at the call site before the campaign (`integration.call_site`).",
                },
                "edit_target": {
                    "from_identity": None,
                    "source_owner": "sglang",
                    "repo_root_var": "@SGLANG_ROOT@",
                    "source_file": TARGET_FILE,
                    "editable_sources": [TARGET_FILE],
                    "entry_function": "Sampler.forward",
                    "source_resolution_method": "declared",
                    "resolution_evidence": f"Read inside {facts['image']}: {TARGET_FILE} defines class Sampler with forward(), whose standard sampling path runs `logits[:] = torch.softmax(logits, dim=-1)`. The demo replaces that call with the naive baseline (`integration.call_site`); device_symbol is the largest kernel of a standalone torch.profiler capture of the naive baseline at [{PRIMARY_BATCH}, {vocab}] fp32.",
                    "resolution_caveats": [
                        "No operator_identity and no in-service trace back this; the location was read by hand from the image."
                    ],
                },
                "integration": {
                    "target_files": [TARGET_FILE],
                    "base_sha256": {TARGET_FILE: facts["sampler_sha256"]},
                    "module_symbols": facts["module_symbols"],
                    "substitution": "call_site_fragment",
                    "call_site": {
                        "fragment": "logits[:] = torch.softmax(logits, dim=-1)",
                        "call": "sampler_softmax(logits, logits)",
                    },
                    "public_symbol": None,
                    "signature": "sampler_softmax(logits: Tensor[B, V] fp32, out: Tensor[B, V] fp32) -> Tensor",
                    "invariants": [
                        "writes in place into the caller-provided `out`; the call site is `logits[:] = ...`, so a replacement that allocates is not substitutable there",
                        "fp32 in and fp32 out",
                        "every output row sums to 1 within 1e-4; these feed torch.multinomial, and a row that does not is a sampler drawing from the wrong distribution",
                        "the reduced dimension is the last one and is contiguous; a replacement that relies on that must say so, because a non-contiguous `logits` then gives wrong results rather than an error",
                        "no build step may be added; a replacement needing compilation cannot be delivered as an overlay",
                    ],
                    "apply_mode": W.apply_mode_for("call_site_fragment"),
                    "requires_restart": True,
                    "build_step": None,
                },
                "gates": {
                    "snr_db": 30.0,
                    "allclose": {"atol": 1e-6, "rtol": 1e-3},
                    "extra": [
                        {
                            "name": "rows_sum_to_one",
                            "tolerance": 1e-4,
                            "brief": "Every output row sums to 1. No SNR threshold catches a wrong distribution.",
                        },
                        {
                            "name": "writes_in_place",
                            "tolerance": None,
                            "brief": "The returned tensor IS the caller's `out` and `out` was written. The call site is `logits[:] = ...`; a replacement that allocates passes every isolated correctness case and breaks there.",
                        },
                    ],
                },
                "noise_floor": None,  # transcribed from evidence/ by `transcribe`
                "apparatus": [
                    "_common.py",
                    "run_correctness.sh",
                    "run_performance.sh",
                    "workset.yaml",
                    f"definitions/softmax/{OPERATOR}.json",
                    f"workloads/softmax/{OPERATOR}.jsonl",
                ],
                "provenance": {
                    "source": (
                        f"Standalone torch.profiler capture of the naive baseline call at [{PRIMARY_BATCH}, {vocab}] fp32 "
                        f"on gfx942 (GPU {facts['measure_gpu']}, {facts['image']}), {prof['calls']} calls. NOT an "
                        "in-service decode profile: '% Total' is the softmax kernel's share of device time in that "
                        "isolated capture of the naive baseline; this row is its largest kernel."
                    ),
                    "pct_total": round(prof["pct"], 2),
                    "in_service_avg_us": None,
                    "magpie_row": {
                        "Name": prof["name"],
                        "Calls": prof["calls"],
                        "Self CUDA total (us)": round(prof["self_us"], 3),
                        "Avg time (us)": round(prof["avg_us"], 3),
                        "% Total": round(prof["pct"], 2),
                        "Input Shapes": prof["input_shapes"],
                        "in_service_avg_us": None,
                    },
                },
            }
        ],
    }
    (root / "workset.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    # 6. The README the `code` content type requires.
    (content / "README.md").write_text(
        README.format(
            vocab=vocab, gpu=facts["measure_gpu"], node=facts["node"], image=facts["image"]
        ),
        encoding="utf-8",
    )

    # 7. The forge add-on, by the generator a real run uses.
    export = subprocess.run(
        [sys.executable, str(e2e / "assets/lib/forge_export.py"), "--workset", str(root)],
        capture_output=True,
        text=True,
    )
    if export.returncode != 0:
        _die(f"forge_export failed:\n{export.stderr[-1500:]}")
    print(export.stdout.strip())
    print(
        f"build_fixture: assembled {root}; evidence/ is NOT written here -- run measure_in_container.sh"
    )


def transcribe(content: Path, container: str, gpu_arch: str = "gfx942") -> None:
    root = content / "items" / "codes"
    # measure_in_container.sh's default pair writes these two names.
    perf_rel, corr_rel = "evidence/performance.json", "evidence/correctness.json"
    perf_path, corr_path = root / perf_rel, root / corr_rel
    if not perf_path.is_file() or not corr_path.is_file():
        _die(f"evidence missing: {perf_path.is_file()=} {corr_path.is_file()=}")
    perf = json.loads(perf_path.read_text(encoding="utf-8"))
    corr = json.loads(corr_path.read_text(encoding="utf-8"))
    if not corr.get("passed"):
        _die("the correctness report did not pass; refusing to transcribe")
    floor = perf.get("noise_floor")
    if not isinstance(floor, (int, float)):
        _die("the performance report carries no numeric noise_floor")
    doc = yaml.safe_load((root / "workset.yaml").read_text(encoding="utf-8"))
    doc["ground_truth"]["noise_floor"] = floor
    for operator in doc["operators"]:
        operator["noise_floor"] = floor
    doc["evidence"] = {
        "correctness_report": corr_rel,
        "performance_report": perf_rel,
        "measured_on": {
            "node": perf["environment"]["node"],
            "gpu_arch": gpu_arch,
            "container": container,
            "at": perf["started_at"],
        },
    }
    # One operator, one measured run: every declared report path names the file
    # that run wrote, rather than a per-scope name no run produced.
    for block in [doc["entrypoints"]] + [o["entrypoints"] for o in doc["operators"]]:
        for key, rel in (("correctness", corr_rel), ("performance", perf_rel)):
            block[key]["report"] = rel
    (root / "workset.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(
        f"build_fixture: noise_floor {floor} transcribed; evidence block written "
        f"({datetime.now(timezone.utc).isoformat(timespec='seconds')}, {socket.gethostname()})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("assemble")
    a1.add_argument("--e2e", required=True, type=Path)
    a1.add_argument("--out", required=True, type=Path)
    a1.add_argument("--facts", required=True, type=Path)
    a2 = sub.add_parser("transcribe")
    a2.add_argument("--out", required=True, type=Path)
    a2.add_argument("--container", required=True)
    a = ap.parse_args()
    if a.cmd == "assemble":
        assemble(a.e2e.resolve(), a.out, json.loads(a.facts.read_text(encoding="utf-8")))
    else:
        transcribe(a.out, a.container)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
