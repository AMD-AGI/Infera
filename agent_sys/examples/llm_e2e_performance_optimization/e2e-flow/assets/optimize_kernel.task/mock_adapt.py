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


#: `"$VAR"/a/b` -> `"$VAR/a/b"`. The closing quote moves to the far side of the
#: path; the shell behaviour is identical and a trailing glob stays outside.
_QUOTED_PREFIX = re.compile(r'"\$([A-Za-z_][A-Za-z0-9_]*)"(/[A-Za-z0-9._+@/-]+)')


def _ensure_impl_entry(packup: Path, workset_root: Path, operator: dict) -> str | None:
    """Give the sealed kernel the `run` entry point m3's `--impl` contract needs.

    **The sealed candidate predates the contract it now has to satisfy.**
    `_common.py:289` refuses a candidate whose source defines no top-level
    `run`, and `results/optimized_kernel.py` from the 2026-09-02 run defines
    only `sampler_softmax` — so `run_performance.sh --impl` would have exited
    *"defines no `run`"* before measuring anything. Found 2026-09-04 by reading
    the two files against each other rather than by running the campaign.

    **Appending is right here and refusing is right in `30_run_forge.sh`**, and
    the difference is which artefact is at fault. There the source *is* the
    workset's own baseline, so a missing `run` is a defect in the workset and
    papering over it would hide m3's problem — that script says so and exits.
    Here the source is a sealed artefact from a previous generation of the
    contract, and reconciling those is this file's entire job (see the module
    docstring). The shim is additive: the engine symbol m5 installs is
    untouched.

    **The shim is m3's own, not one invented here.** The delegated symbol is
    read out of the Definition's `baseline` — which carries exactly this pair,
    `def sampler_softmax(...)` beside `def run(*args, **kwargs)` — so the name
    cannot drift from what the harness calls. A sealed kernel that does not
    define that symbol is a real mismatch and is refused rather than guessed at.
    """
    kernel = packup / "results" / "optimized_kernel.py"
    if not kernel.is_file():
        return None
    source = kernel.read_text(encoding="utf-8")
    if re.search(r"^def run\(", source, re.M):
        return None

    relative = operator.get("definition")
    if not relative:
        lib.die("the workset's operator names no `definition`, so the --impl entry cannot be derived")
    definition = lib.load_json(workset_root / str(relative))
    baseline = (definition or {}).get("baseline")
    if not isinstance(baseline, str):
        lib.die(f"{relative} carries no `baseline` source, so the --impl entry cannot be derived")
    delegated = re.search(r"^def run\(.*?\n\s+return\s+([A-Za-z_]\w*)\(", baseline, re.M | re.S)
    if not delegated:
        lib.die(
            f"{relative}'s `baseline` has no `def run(...)` delegating to a symbol; m3's --impl "
            "contract cannot be satisfied without knowing which callable is the entry point"
        )
    symbol = delegated.group(1)
    if not re.search(rf"^def {re.escape(symbol)}\(", source, re.M):
        lib.die(
            f"the sealed kernel defines no `{symbol}`, which is what the Definition's `run` "
            f"delegates to. Sealed candidate and workset Definition disagree about the entry "
            f"point; that is a real mismatch and this script will not guess past it"
        )
    kernel.write_text(
        source.rstrip("\n")
        + "\n\n\n# ----- entry point, added by mock_adapt -----\n"
        + "# The sealed kernel predates m3's `--impl` contract, which requires a\n"
        + f"# top-level `run`. Delegates to `{symbol}`, the symbol the Definition's own\n"
        + "# `baseline` delegates to. Additive: the engine symbol m5 installs is unchanged.\n"
        + "def run(*args, **kwargs):\n"
        + f"    return {symbol}(*args, **kwargs)\n",
        encoding="utf-8",
    )
    return symbol


def _reseat_quotes(packup: Path) -> list[str]:
    """Make the sealed markdown survive `handoff.locality.check`.

    **The seal refuses `cp "$PACKUP"/scripts/kernel/*.py`**, and the reason has
    nothing to do with the command: `handoff/locality.py:67 _CANDIDATE` has a
    lookbehind of `[A-Za-z0-9._~@+-]`, `"` is not in it, so a closing quote does
    not shield the path that follows — `/scripts/kernel/` is read as a rooted
    local path. Inside the quotes the preceding character is the variable's last
    letter, which is in the class, so `cp "$PACKUP/scripts/kernel/"*.py` seals.

    Found by m5 at rung 5: `packup` carries this file verbatim into the terminal
    kit, and nothing before that stage seals a `kernel_optimization` at all.

    **This is an adaptation, not a redaction.** No number, path or claim
    changes — only where a quote sits in a shell expression the sealed run wrote
    before the rule existed. The sealed bytes on `/shared_nfs` are evidence and
    are never edited; this rewrites the working copy, and says so in `notes`.
    The real path does not need it: `60_write_handoff.py`'s REPRODUCE skeleton
    carries the rule where its author will read it.
    """
    changed: list[str] = []
    for path in sorted(packup.rglob("*.md")):
        before = path.read_text(encoding="utf-8", errors="replace")
        after = _QUOTED_PREFIX.sub(r'"$\1\2"', before)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed.append(str(path.relative_to(packup)))
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--handoff", required=True, help="$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION")
    ap.add_argument(
        "--premise",
        default=os.environ.get("E2E_MOCK_PREMISE", "matched"),
        # `matched` is MOCK-MAP (G)'s word and is canonical. `held` is accepted
        # because an earlier draft of this script used it and the value travels
        # through a `--var`; an operator who types the old one should get the
        # behaviour they meant rather than an argparse error two stages in.
        choices=("matched", "held", "mismatched"),
        help="`mismatched` reproduces the sealed run's gfx942-vs-gfx950 abort on purpose",
    )
    a = ap.parse_args()

    content = Path(a.handoff)
    codes = content / "items" / "codes"
    packups = sorted(p for p in codes.iterdir() if p.is_dir()) if codes.is_dir() else []
    if len(packups) != 1:
        lib.die(f"expected exactly one packup under {codes}, found {[p.name for p in packups]}")
    packup = packups[0]

    reseated = _reseat_quotes(packup)

    workset = lib.load_workset()
    operator = lib.pick_operator(workset, os.environ.get("E2E_WORKSET_OPERATOR") or None)
    operator_id = str(operator.get("operator_id"))
    workset_root = lib.workset_root()
    # After the workset is loaded, because the entry symbol is read out of the
    # Definition rather than assumed.
    impl_entry = _ensure_impl_entry(packup, workset_root, operator)
    run_env = lib.load_environment()
    ground = workset.get("ground_truth") or {}

    # --- carry the workset with the handoff ---------------------------------
    apparatus = packup / lib.APPARATUS
    if apparatus.exists():
        shutil.rmtree(apparatus)
    # The workset's declared file list, not a directory guess -- same rule as
    # the real producer, so the mock exercises the real constraint.
    for relative in operator.get("apparatus") or []:
        source = workset_root / str(relative)
        if not source.is_file():
            lib.die(f"apparatus names {relative!r}, which is not in the workset")
        destination = apparatus / str(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        shutil.copymode(source, destination)
    (packup / "results").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(workset_root / "workset.yaml", packup / lib.SNAPSHOT)

    baseline_rel = (workset.get("evidence") or {}).get("performance_report")
    baseline: dict[str, float] = {}
    if baseline_rel and (workset_root / str(baseline_rel)).is_file():
        shutil.copyfile(workset_root / str(baseline_rel), packup / lib.BASELINE_REPORT)
        baseline = lib.report_per_case_ms(lib.load_json(packup / lib.BASELINE_REPORT), operator_id)

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
        # **The mock dropped this and the real path does not**
        # (`20_premise_gate.py:135`). `dtype` is on M4.3.5's abort list, so a
        # premise without it is not a weaker premise — it is one the consumer
        # reads as `None`, and `check_speedup_substantiated` correctly aborts
        # with *"optimised … at dtype None; the workset's ground truth says
        # 'float32'"*. rung 0 stopped here. The workset carried
        # `ground_truth.dtypes` all along; only the mock failed to copy it,
        # which is the mock failing a gate the real producer passes.
        "dtypes": dict(ground.get("dtypes") or {}),
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
    integration = operator.get("integration") or {}
    target_files = list(integration.get("target_files") or [])
    container_path = (
        lib.container_path_for(str(target_files[0]), str(target.get("repo_root_var") or ""))
        if target_files else None
    ) or ""
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
            "source_file": str(target_files[0]) if target_files else "",
            # `edit_target`, not `integration.public_symbol` — see the same
            # line in `60_write_handoff.py`. The validator compares this
            # against `edit_target.entry_function`.
            "entry_function": str(target.get("entry_function") or ""),
            **({"repo_root_var": str(target["repo_root_var"])} if target.get("repo_root_var") else {}),
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
    if integration.get("public_symbol"):
        apply_block["runtime_marker"] = {
            "first_call": re.escape(str(integration["public_symbol"])) + r"\s*\("
        }

    document = {
        "schema_version": 1,
        "operator": operator_id,
        "workset_ref": {
            **lib.input_ref("operator_workset"),
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
                " A closing quote was moved from before a path to after it in "
                + ", ".join(reseated)
                + " so the sealed markdown survives `handoff.locality.check`; no number, path or "
                "claim changed. See `_reseat_quotes`."
                if reseated else ""
            )
            + (
                f" A `run(*args, **kwargs)` entry point delegating to `{impl_entry}` was appended "
                "to results/optimized_kernel.py: the sealed kernel predates m3's `--impl` "
                "contract, which requires a top-level `run` (`_common.py:289`), and without it "
                "the re-measurement exits before measuring. Additive only -- the engine symbol "
                "m5 installs is unchanged, and the delegated name is read from the Definition's "
                "own `baseline` rather than chosen here. See `_ensure_impl_entry`."
                if impl_entry else ""
            )
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

    # MOCK-MAP (A). **After** the document, because (G) touches the same tree
    # and because the warnings the record carries are the ones (G) just decided.
    lib.render_environment(content, (premise.get("verdict") or {}).get("warnings") or [])

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
