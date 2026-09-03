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
import re
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


#: Container roots whose contents **cannot** be delivered as an overlay.
#: `container_roots.yaml` says it of `SGL_KERNEL_ROOT` in as many words: those
#: are compiled sources, a change needs the image rebuilt, *and `apply_mode`
#: must say so*. `overlay_files` over one of them bind-mounts a `.cpp` nothing
#: will recompile — the patch is provably on disk, `check_patch_live` can even
#: confirm the bytes, and the running engine executes the stock binary.
_NOT_OVERLAYABLE = ("@SGL_KERNEL_ROOT@",)


def _apply_block(pinned: dict, packup: Path, kernel: Path, premise: dict) -> dict:
    """patchkit's manifest, in patchkit's vocabulary, plus the M5.1.1 cross-check.

    **Written against `integration`, not against `edit_target`**, and the
    distinction is m3's rather than a preference. `edit_target` says where an
    *optimiser* edits; `integration` says where a *replacement* is installed and
    what it may not change. They are usually the same file and are never the
    same statement, and collapsing them is what forces m5 to read an
    optimisation report and decide.

    m5's `apply_patch` is a program because it reads this rather than judging
    it. The full contract is the `CONTRACT` constant in
    `assets/apply_patch.task/apply.py`.

    Three refusals rather than a plausible-looking manifest, because each
    produces a patch that *applies* and does nothing:

    * no configured container root matches a `target_file`, so the `@ROOT@/...`
      form would have to be invented — and m5 would apply the invention to a
      real image;
    * a file under a root whose contents need a rebuild, where `overlay_files`
      is the wrong mode by the root's own description;
    * the stock file is unreachable, so `base_sha256` would be a placeholder.
    """
    integration = pinned.get("integration") or {}
    target = pinned.get("edit_target") or {}
    repo_root_var = str(target.get("repo_root_var") or "")

    target_files = list(integration.get("target_files") or [])
    if not target_files:
        lib.die(
            "the workset's operator declares no integration.target_files, so there is nothing "
            "to install a replacement over. That field is required by workset.schema.json; a "
            "workset without it did not come from `build_workset`"
        )
    # The workset names what may be replaced precisely so that m4 cannot widen
    # the blast radius by editing a sixth file and m5 cannot be surprised by one.
    if len(target_files) > 1:
        lib.die(
            f"the workset declares {len(target_files)} target_files {target_files} and this task "
            "delivers one optimised kernel. Optimising a multi-file integration point is not "
            "something to improvise at handoff-writing time"
        )

    declared_mode = str(integration.get("apply_mode") or "overlay_files")
    if declared_mode != "overlay_files":
        lib.die(
            f"the workset declares apply_mode={declared_mode!r}; this stage delivers "
            "overlay_files only (todo.md T5 defers the registry hook)"
        )

    files = []
    for target_file in target_files:
        container_path = lib.container_path_for(str(target_file), repo_root_var)
        if container_path is None:
            lib.die(
                f"integration.target_files names {target_file!r} under repo_root_var "
                f"{repo_root_var!r}, which is under no root in assets/lib/container_roots.yaml, "
                "so its @ROOT@/... form cannot be derived. Add the root there rather than "
                "writing an absolute path: the seal refuses one, and an invented path is one "
                "m5 would apply to a real image"
            )
        if container_path.startswith(_NOT_OVERLAYABLE):
            lib.die(
                f"{container_path} needs the image rebuilt and cannot be delivered as an overlay "
                "(see its description in assets/lib/container_roots.yaml). Bind-mounting a "
                "compiled source produces a patch that is provably on disk and provably not "
                "running"
            )
        if not kernel.is_file():
            continue
        # The hash of the **stock** file, not of the replacement. m5 pulls the
        # file out of the image, hashes it, and refuses on a mismatch — that is
        # what makes "this patch belongs to this image" checkable rather than
        # asserted, and a wrong value here turns into a refusal two stages later
        # with no way to tell a stale patch from a typo.
        #
        # **The workset's is preferred and m4's is the cross-check**, which is
        # m3's argument and it is the better one: their hash is pinned at the
        # moment the operator was identified, while one taken here is a hash of
        # whatever the file had become by then — so a file that changed in
        # between is *detectable* rather than silently blessed. m4 can hash it
        # at all only because m1–m4 share one container (CONTRACT §5).
        declared = (integration.get("base_sha256") or {}).get(str(target_file))
        stock = lib.expand_container_path(container_path)
        measured = lib.sha256_of(stock) if stock is not None and stock.is_file() else None

        if declared and measured and declared != measured:
            lib.die(
                f"the workset recorded base_sha256 {declared[:12]}… for {target_file} at identify "
                f"time; it hashes {measured[:12]}… in this container now. The file changed "
                "underneath the analysis, so the operator that was identified is not the operator "
                "about to be patched"
            )
        base_sha256 = declared or measured
        if not base_sha256:
            lib.die(
                f"no base_sha256 for {container_path}: the workset recorded none and the stock "
                f"file is not readable here (resolved to {stock}). It is what lets m5 prove this "
                "patch belongs to this image, and a placeholder is a refusal two stages "
                "downstream with no way to tell a stale patch from a typo"
            )
        if not declared:
            print(
                f"note: the workset recorded no base_sha256 for {target_file}; using m4's own "
                "hash of the file in this container, which is pinned later than the analysis",
                file=sys.stderr,
            )
        files.append({
            "container_path": container_path,
            "base_sha256": base_sha256,
            "change": "modify",
            # `replacement`, never a hand-rolled diff: m4's artefact is a whole
            # file, and `apply_patch` generates the diff from stock->replacement
            # so `patch_overlay` keeps one shape downstream.
            "replacement": "results/optimized_kernel.py",
        })

    block = {
        "apply_mode": declared_mode,
        "manifest": lib.APPLY_MANIFEST,
        "image": ((premise.get("run_environment") or {}).get("fixed") or {}).get("image"),
        "logical_operator": str(pinned["operator_id"]),
        # Copied from the workset's `integration`, verbatim, so that
        # `check_optimization_shape` can compare the two and refuse a divergence.
        "integration_point": {
            "source_file": str(target_files[0]),
            "entry_function": str(integration.get("public_symbol") or ""),
            **({"repo_root_var": repo_root_var} if repo_root_var else {}),
            **({"entry_function_line": target["entry_function_line"]}
               if isinstance(target.get("entry_function_line"), int) else {}),
        },
        "files": files,
        "revert": "Remove the overlay and restart the engine; nothing in the stock tree is modified in place.",
    }

    # What a replacement may not change. Carried from the workset rather than
    # restated, because none of it is inferable from the signature: for
    # `sampler_vocab_softmax` the call site is `logits[:] = torch.softmax(...)`,
    # so a replacement that *allocates* is not substitutable there — and it
    # passes every correctness gate.
    must_preserve = {}
    if integration.get("signature"):
        must_preserve["signature"] = integration["signature"]
    if integration.get("invariants"):
        must_preserve["invariants"] = list(integration["invariants"])
    for key in ("requires_restart", "build_step"):
        if key in integration:
            must_preserve[key] = integration[key]
    if must_preserve:
        block["must_preserve"] = must_preserve

    public_symbol = str(integration.get("public_symbol") or "")
    if public_symbol:
        # Declaring this upgrades m5's `check_patch_live` from "the patched
        # bytes were on disk" to "the patched code was entered". Without it the
        # validator passes and says what it could not show, which is honest and
        # is strictly less proof that this kernel ever ran. The symbol rather
        # than `edit_target.entry_function`, because the *file* may be rewritten
        # wholesale and this is the thing that must survive it.
        block["runtime_marker"] = {"first_call": re.escape(public_symbol) + r"\s*\("}
    return block


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
    # **The files the workset declares, not a directory guess.** m3's
    # `operators[].apparatus` exists for this consumer specifically:
    # `check_speedup_substantiated` re-measures from *this* copy, because a
    # validator on an output phase is handed only the handoffs it declared and
    # cannot reach the workset. A kit that reports a speedup and does not carry
    # the thing that measured it cannot be checked by anyone who does not
    # already have the workset, which is most readers -- and a consumer that
    # guesses the file set is one that ships a broken copy and finds out an hour
    # later.
    for relative in pinned.get("apparatus") or []:
        source = workset_root / str(relative)
        if not source.is_file():
            lib.die(f"apparatus names {relative!r}, which is not in the workset")
        destination = apparatus / str(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        shutil.copymode(source, destination)  # the entrypoints are executable
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
    measured = lib.report_per_case_ms(performance, operator_id) if performance else {}

    target = pinned.get("edit_target") or {}
    kernel_path = packup / "results" / "optimized_kernel.py"
    apply_block = _apply_block(pinned, packup, kernel_path, premise)
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
        "apply": apply_block,
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
        # Declared by the workset, derived there from the measured spread as
        # `1 + 2.83 x rsd_max` -- the two-sample 2-sigma separation, so a
        # noisier host correctly demands a bigger win. **No default**: STEP 1
        # already refused a workset without it, and a fallback here would be m4
        # choosing when to call its own result significant.
        noise_floor = pinned.get("noise_floor")
        if not isinstance(noise_floor, (int, float)):
            lib.die("the pinned inputs carry no numeric noise_floor; re-run STEP 1")
        document["evidence"]["performance"]["claim"] = {
            "speedup_per_case": per_case,
            "mean_case_speedup": round(sum(per_case.values()) / len(per_case), 4),
            "noise_floor": float(noise_floor),
        }

    lib.write_json(packup / lib.DOC, document)

    # The same facts, in the file m5 actually opens. Written from `apply_block`
    # rather than assembled a second time: two records of one fact that are
    # built twice are two records that drift, and `check_optimization_shape`
    # compares them precisely because it has seen that happen.
    lib.write_json(packup / lib.APPLY_MANIFEST, {
        "schema_version": 1,
        "operator_id": operator_id,
        "logical_operator": apply_block["logical_operator"],
        "image": apply_block["image"],
        "apply_mode": apply_block["apply_mode"],
        "files": apply_block["files"],
        **({"runtime_marker": apply_block["runtime_marker"]} if "runtime_marker" in apply_block else {}),
    })

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
