#!/usr/bin/env python3
"""Both directions of `kernel_optimization.schema.json`, in one second.

`assets/schemas/README.md` rule 1: *write it against a real artefact, then prove
both directions — the real document validates, and a document missing what
matters is rejected with the fields named.* This file is that proof for m4's
schema, and it is runnable rather than asserted in prose.

    python3 assets/schemas/samples/selftest_kernel_optimization.py

**The rejections are mutations of the accepted sample, not separate files.** A
hand-written bad document drifts from the good one the first time the schema
changes, and then tests a shape nobody produces any more. Every case below
starts from `kernel_optimization.01_sealed_run.json` and breaks exactly one
thing, so a case can only fail for the reason it names.

**Why `expect` lists paths and not messages.** `jsonschema`'s wording is its own
and changes between versions; the *location* of the complaint is what a producer
acts on. A case asserts that the schema complained about `$.apply`, not that it
phrased it a particular way.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "lib"))

import schema  # noqa: E402 — the path insert above is what makes it importable

#: Both accept-direction samples. Two, because they exercise disjoint branches:
#: 01 is the sealed mock whose premise did not hold and which therefore carries
#: **no** `claim`, and 02 is a campaign whose premise held and which therefore
#: carries one. A schema branch only ever exercised in the reject direction is a
#: branch nobody has confirmed can be satisfied at all.
ACCEPTED = (
    _HERE / "kernel_optimization.01_sealed_run.json",
    _HERE / "kernel_optimization.02_real_campaign.json",
)
#: Mutations are applied to the sample that has a claim to break.
BASE = ACCEPTED[1]


def _drop(doc: dict, dotted: str) -> dict:
    out = copy.deepcopy(doc)
    node = out
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node[part]
    del node[leaf]
    return out


def _set(doc: dict, dotted: str, value) -> dict:
    out = copy.deepcopy(doc)
    node = out
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node[part]
    node[leaf] = value
    return out


def _problems(doc: dict) -> list[str]:
    try:
        schema.validate("kernel_optimization", doc)
    except schema.SchemaError as exc:
        return str(exc).splitlines()[1:]
    return []


def main() -> int:
    good = json.loads(BASE.read_text(encoding="utf-8"))

    failures: list[str] = []
    checks = 0

    # --- direction 1: the real documents validate ---------------------------
    for sample in ACCEPTED:
        checks += 1
        problems = _problems(json.loads(sample.read_text(encoding="utf-8")))
        if problems:
            failures.append(f"{sample.name} does not validate:\n  " + "\n  ".join(problems))
        else:
            print(f"ok   {sample.name} validates")

    # --- direction 2: a document missing what matters is rejected -----------
    #
    # One case per field the skeleton's `require_fields` names, plus the three
    # rules that are the reason this schema exists rather than a shape check.
    cases: list[tuple[str, dict, str]] = [
        (
            "no apply block at all — m5 cannot be a program without one (M5.1.1)",
            _drop(good, "apply"),
            "$",
        ),
        (
            "no premise block — the M4.3.5 comparison cannot be made",
            _drop(good, "premise"),
            "$",
        ),
        (
            "no workset_ref — a speedup whose denominator has no provenance",
            _drop(good, "workset_ref"),
            "$",
        ),
        (
            "no evidence",
            _drop(good, "evidence"),
            "$",
        ),
        (
            "no operator",
            _drop(good, "operator"),
            "$",
        ),
        (
            "the baseline was re-measured here instead of taken from the workset "
            "— the exact rule M4.3.5 reversed",
            _set(good, "evidence.performance.baseline.source", "remeasured_here"),
            "$.evidence.performance.baseline.source",
        ),
        (
            "a mock claiming a speedup — the most misleading artefact this package can produce",
            _set(_set(good, "evidence.forge.mock", True), "evidence.forge.ran", False),
            "$.evidence.performance",
        ),
        (
            "an aborted premise claiming a speedup — a different question, not a smaller result",
            _set(good, "premise.verdict.held", False),
            "$.evidence.performance",
        ),
        (
            "a claim with no measurement to have computed it from",
            _drop(good, "evidence.performance.measured"),
            "$.evidence.performance",
        ),
        (
            "correctness reported as a ratio instead of a boolean — correctness is not a percentage",
            _set(good, "evidence.correctness.passed", 0.67),
            "$.evidence.correctness.passed",
        ),
        (
            "apply mode the deferred registry hook, which nothing implements (todo.md T5)",
            _set(good, "apply.apply_mode", "registry_hook"),
            "$.apply.apply_mode",
        ),
        (
            "a bare absolute container_path — the handoff seal refuses to publish it at all",
            _set(
                good,
                "apply.files",
                [{
                    "container_path": "/sgl-workspace/sglang/python/sglang/srt/layers/sampler.py",
                    "base_sha256": "0" * 64,
                    "change": "modify",
                    "replacement": "results/optimized_kernel.py",
                }],
            ),
            "$.apply.files.0.container_path",
        ),
        (
            "a files[] entry carrying both `patch` and `replacement` — patchkit takes exactly one",
            _set(
                good,
                "apply.files",
                [{
                    "container_path": "@SGLANG_ROOT@/srt/layers/sampler.py",
                    "base_sha256": "0" * 64,
                    "change": "modify",
                    "replacement": "results/optimized_kernel.py",
                    "patch": "apply/patches/0001-sampler.patch",
                }],
            ),
            "$.apply.files.0",
        ),
        (
            "a files[] entry carrying neither, so there is nothing to apply",
            _set(
                good,
                "apply.files",
                [{
                    "container_path": "@SGLANG_ROOT@/srt/layers/sampler.py",
                    "base_sha256": "0" * 64,
                    "change": "modify",
                }],
            ),
            "$.apply.files.0",
        ),
        (
            "the manifest at a path of the producer's choosing — apply_patch globs for the fixed one",
            _set(good, "apply.manifest", "apply/my_manifest.json"),
            "$.apply.manifest",
        ),
        (
            "a workset snapshot at a path of the producer's choosing",
            _set(good, "workset_ref.snapshot", "results/my_copy.yaml"),
            "$.workset_ref.snapshot",
        ),
        (
            "a stray top-level field — a typo that a permissive schema would carry to m5",
            _set(good, "speedup", 2.83),
            "$",
        ),
    ]

    for label, doc, expect in cases:
        checks += 1
        problems = _problems(doc)
        if not problems:
            failures.append(f"NOT REJECTED: {label}")
            continue
        if not any(p.strip().startswith(expect + ":") for p in problems):
            failures.append(
                f"rejected for the wrong reason: {label}\n"
                f"  expected a complaint at {expect}, got:\n    " + "\n    ".join(problems)
            )
            continue
        print(f"ok   rejected: {label}")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"\n{len(failures)} failure(s) of {checks}")
        return 1
    print(f"\n{checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
