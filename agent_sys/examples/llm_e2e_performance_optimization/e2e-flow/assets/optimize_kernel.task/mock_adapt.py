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
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "steps"))

import _lib as lib  # noqa: E402
import _fragment_patch as fragpatch  # noqa: E402


#: `"$VAR"/a/b` -> `"$VAR/a/b"`. The closing quote moves to the far side of the
#: path; the shell behaviour is identical and a trailing glob stays outside.
_QUOTED_PREFIX = re.compile(r'"\$([A-Za-z_][A-Za-z0-9_]*)"(/[A-Za-z0-9._+@/-]+)')


def _hash_from_image(container_path: str, image: str) -> str | None:
    """`base_sha256` of the stock file, taken out of the image itself.

    **A mock may obtain a real fact by a route the producer does not use; it may
    not assert a fact the producer does not have.** Leader's ruling,
    2026-09-04, drawing the line between this and synthesising a
    `public_symbol`: `60_write_handoff.py` writes this same field hashed from
    the stock file, so extracting it from the image is *the same fact by another
    route*. A `public_symbol` the real workset records as `null` would be a
    *different fact*, and both m5 and I refused that one.

    **Why it is needed.** `mock_adapt` runs on a login node with no engine tree,
    so the fallback below hashes the *replacement* and says so. m5's
    `apply.py` then refuses — correctly — with *"the patch was cut against
    fcd3c924e48d…"*, and the run stops two gates before their compile check and
    both surface refusals ever see input. Measured 2026-09-04 by driving
    `apply.py` standalone against rung 0's own artefact.

    **The idiom is m5's, copied rather than re-derived** (`apply.py:510-524`,
    CONTRACT §4.1): `docker create` starts no process and touches no GPU, the
    `trap` removes the handle, and `patchkit.expand` turns the `@ROOT@` form
    into the path the image actually has — a lesson their comment already paid
    for once, when a doubled `python/sglang` made `docker cp` refuse a path no
    image contains.

    **Returns `None` rather than raising.** No allocation, no reachable node and
    no such path in the image are all ordinary on a login node, and a mock that
    cannot run without a node would be worse than one with an honestly degraded
    hash. The caller falls back and records which route it took.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
    try:
        import nodecall
        import patchkit
    except ImportError:
        return None
    inside = patchkit.expand(container_path)
    # **Hash on the node; never copy the file here.** The first version wrote
    # `docker cp "$CID:<inside>" '<local tempdir>'` — and `docker cp` runs on
    # the **node**, where a `tempfile.TemporaryDirectory()` created on the login
    # node does not exist: *invalid output path: directory "/tmp/tmp…"*. That is
    # T42 in a helper whose docstring already cites the sibling lesson — a path
    # computed here handed to a shell over there.
    #
    # Streaming to `sha256sum` is the better fix rather than a repaired
    # destination: **the file is not wanted, only its digest**, so there is no
    # shared filesystem to arrange, nothing to clean up, and no second locality
    # question later. `docker cp <src> -` emits a tar; `tar -xO` writes the
    # member's bytes to stdout.
    script = "\n".join([
        "set -e",
        f"CID=$(docker create '{image}' true)",
        "trap 'docker rm -f $CID >/dev/null 2>&1' EXIT",
        f'docker cp "$CID:{inside}" - | tar -xO | sha256sum | cut -d" " -f1',
    ])
    try:
        out = nodecall.on(script)
    except Exception:
        return None
    digest = (out or "").strip().splitlines()[-1].strip() if (out or "").strip() else ""
    return digest if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) else None


def _stock_from_image(container_path: str, image: str, scratch: Path) -> str | None:
    """The stock file's **text**, out of the image. `None` if it cannot be had.

    **A shared destination, not a stream, and not a login-node tempdir.**
    `_hash_from_image` streams through `sha256sum` because a digest is all it
    needs and streaming removes the locality question entirely. A diff needs the
    *bytes*, so there is something to transport and the question comes back —
    and the first version of `_hash_from_image` got it wrong by writing to a
    `tempfile.TemporaryDirectory()` created here while `docker cp` runs on the
    **node** (*invalid output path*). So `scratch` must be a path both hosts
    mount; the caller passes one under the run root, which is NFS.

    Streaming the content back through `nodecall.on` was the alternative and is
    worse: the payload crosses `spur exec bash -lc` and a shell re-parse, and a
    30 KB Python file is exactly the kind of thing that survives that in testing
    and mangles on the one line that matters.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
    try:
        import nodecall
        import patchkit
    except ImportError:
        return None
    inside = patchkit.expand(container_path)
    scratch.mkdir(parents=True, exist_ok=True)
    dest = scratch / "stock.src"
    script = "\n".join([
        "set -e",
        f"CID=$(docker create '{image}' true)",
        "trap 'docker rm -f $CID >/dev/null 2>&1' EXIT",
        f"docker cp \"$CID:{inside}\" '{dest}'",
    ])
    try:
        nodecall.on(script)
    except Exception:
        return None
    if not dest.is_file():
        return None
    try:
        text = dest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    # **Remove it: everything under the handoff ships.** `docker cp` has to land
    # somewhere both hosts mount, and the handoff tree is the one such place
    # this script has a handle on -- so the scratch went inside the packup and
    # rung 0 sealed a 30 KB copy of the stock engine file into the artefact as
    # `apply/.stock/stock.src`. Harmless to the validators and wrong in the
    # deliverable: a handoff carries what it claims to carry.
    #
    # Deleting is permitted here and the check is not rhetorical -- the standing
    # rule is that nothing is removed whose path lacks `yihou` or `/tmp`, it
    # follows identity mounts into containers, and it admits no judgement. This
    # path is under the run root, which is `/home/yihou/...`; the guard makes
    # that a precondition rather than an assumption.
    if "yihou" in str(scratch) or str(scratch).startswith("/tmp"):
        shutil.rmtree(scratch, ignore_errors=True)
    return text


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

    # **No baseline report means no honest value for `evidence.performance.
    # measured`.** The alternative is the sealed run's numbers, which is the
    # answer rung 0 already refused, so this stops here instead.
    if not baseline:
        lib.die(
            "the workset carries no usable performance report for "
            f"{operator_id}, so there is no measurement made on THIS host to put in "
            "evidence.performance.measured. Falling back to the sealed 2026-09-02 numbers is "
            "what rung 0 refused at -17.5% across three cases; a mock that cannot state an "
            "honest measurement should not state one"
        )

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

    # Kept for the notes and for `correctness`, NOT for `evidence.performance.
    # measured` -- see the comment there. These are 2026-09-02 numbers.
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
    image = str((run_env.get("fixed") or {}).get("image") or "")
    if stock is not None and stock.is_file():
        base_sha256, sha_source = lib.sha256_of(stock), "the engine tree in this container"
        sha_from = {"method": "engine_tree", "image": None}
    elif container_path and image and (extracted := _hash_from_image(container_path, image)):
        base_sha256 = extracted
        sha_source = f"the stock file extracted from image {image} (docker create + docker cp)"
        sha_from = {"method": "image_extract", "image": image}
    elif kernel.is_file():
        base_sha256, sha_source = lib.sha256_of(kernel), "the sealed replacement (NO ENGINE TREE REACHABLE)"
        sha_from = {"method": "replacement_fallback", "image": None}
    else:
        base_sha256, sha_source = "0" * 64, "nothing (NO ENGINE TREE AND NO KERNEL)"
        sha_from = {"method": "replacement_fallback", "image": None}

    # **A `call_site_fragment` operator is installed as a diff, not as a file.**
    # `apply.py:643-659` has run `patch -p1` all along and the producer could
    # never ask for it: the enum had one value, so the only shape m4 could emit
    # was a whole-file overlay, which drops the target's entire public surface
    # and is what m5's applier refuses by name. Leader's ruling, 2026-09-04.
    #
    # **The mock's diff inserts only the marker.** There is no campaign here and
    # the sealed kernel is a standalone module rather than an edited
    # `sampler.py`, so there is no fragment edit to carry — and a marker-only
    # diff is applyable, non-empty, exercises the whole path, and is honest:
    # the mock's optimisation *is* a no-op. The token says so
    # (`M4_MARKER_ONLY_NO_OPTIMISATION`) because a green `check_patch_live` on a
    # patch that optimises nothing must not read as evidence of one.
    patch_rel = None
    runtime_marker = None
    apply_mode = "overlay_files"
    fragment = str(integration.get("substitution") or "") == "call_site_fragment"
    if fragment and container_path and image:
        entry_function = str((operator.get("edit_target") or {}).get("entry_function") or "")
        if not entry_function:
            lib.die("this operator is `call_site_fragment` but declares no "
                    "`edit_target.entry_function`, so there is nowhere to anchor the marker")
        stock_text = _stock_from_image(container_path, image, packup / "apply" / ".stock")
        if stock_text is None:
            lib.die(
                f"this operator is `call_site_fragment`, so it must be installed as a diff, and "
                f"the stock {target_files[0]} could not be read out of {image}. Without it there "
                "is nothing to cut a patch against -- and the whole-file overlay that used to be "
                "emitted here is the shape m5's applier refuses by name"
            )
        # **The diff is cut in the CONTAINER frame, not the workset's.**
        # `apply.py:637` does `split_placeholder(container_path)` and extracts
        # into `tree/<rel>`, so `rel` is `srt/layers/sampler.py` — while the
        # workset's `target_files[0]` is `python/sglang/srt/layers/sampler.py`,
        # repo-relative. A diff headed with the workset's path made `patch -p1`
        # say **"No file to patch. Skipping patch."** and exit 1 having changed
        # nothing. Same two-frame problem `check_optimization_shape._same_file`
        # exists for, arriving in a third place.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
        import patchkit  # noqa: E402 — the path insert above is what makes it importable

        _root, patch_rel_path = patchkit.split_placeholder(container_path)
        diff_text, runtime_marker = fragpatch.build(
            stock_text, patch_rel_path, entry_function, operator_id, "mock0",
            no_optimisation=True,
        )
        patch_rel = f"{operator_id}.patch"
        patches = packup / "apply" / "patches"
        patches.mkdir(parents=True, exist_ok=True)
        (patches / patch_rel).write_text(diff_text, encoding="utf-8")
        apply_mode = "patch_in_place"

    apply_block = {
        "apply_mode": apply_mode,
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
                # **Structured, at m5's request, because they branch on it.**
                # The prose in `notes` stays -- it is what made the 2026-09-04
                # failure diagnosable in one read -- but a consumer should not
                # have to grep a sentence to learn the hash came from a
                # fallback. m5 prints this in `apply.py`'s mismatch refusal.
                "base_sha256_from": sha_from,
                "change": "modify",
                # One of the two, never both: `patchkit` requires exactly one.
                **({"patch": patch_rel} if patch_rel else
                   {"replacement": "results/optimized_kernel.py"}),
            }]
            if (patch_rel or kernel.is_file()) and container_path else []
        ),
        "revert": "Remove the overlay and restart the engine.",
    }
    if runtime_marker is not None:
        # **The marker the diff installed, not one derived from a symbol.** A
        # `call_site_fragment` operator has no `public_symbol` -- that is the
        # definition of the case -- so the marker cannot be derived, and
        # synthesising a symbol to get one was refused by m4 and m5 both. A diff
        # that edits a line can add a line, so the thing that proves execution
        # is the thing that was installed.
        apply_block["runtime_marker"] = runtime_marker
    elif integration.get("public_symbol"):
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
                # **`measured` means "on this host, now" — so it cannot be the
                # sealed run's numbers, and it was.**
                #
                # This read `"per_case_ms": measured or baseline` with `measured`
                # taken from the 2026-09-02 `verification.json`, i.e. a different
                # machine, image and container on a different day. rung 0 refused
                # it on 2026-09-04, correctly and by a margin that gave the game
                # away: **-17.7 %, -17.1 %, -17.5 % across three cases**, three
                # digits of agreement between them. Noise does not do that; two
                # different worlds do.
                #
                # The schema is explicit — *"What `results/optimized_kernel.py`
                # timed at, on this host, now"* — so this slot may not be filled
                # from an artefact, and the check is not too strict: as
                # constructed the mock could never have passed it, because the
                # artefact was asserting a measurement nobody made here.
                #
                # **The workset's own baseline report is the honest value**, and
                # it was sitting beside the wrong one the whole time. It was
                # measured by m3 in *this* run on *this* node minutes earlier,
                # under the same protocol, and in mock mode
                # `results/optimized_kernel.py` is the workset's baseline source
                # **verbatim** (`30_run_forge.sh` seeds it from the Definition's
                # `baseline`, refusing if it is not there). So it is a real
                # measurement of exactly this source on this host in this run.
                #
                # **What it is not is a measurement m4 made**, and the note below
                # says so rather than letting the report path imply a campaign.
                # Empty is refused rather than silently falling back: no baseline
                # report means there is no honest number for this field, and the
                # sealed one is the answer that has already been wrong once.
                "measured": {
                    "report": lib.BASELINE_REPORT,
                    "per_case_ms": baseline,
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
                " evidence.performance.measured carries the WORKSET's own baseline report, "
                "measured by m3 in this run on this node, because in mock mode "
                "results/optimized_kernel.py IS that baseline source verbatim and no campaign "
                "was run. It is a real measurement of exactly this source on this host; it is "
                "NOT a measurement m4 made. It used to carry the sealed 2026-09-02 numbers, "
                "which rung 0 refused at -17.5% across three cases on 2026-09-04 -- different "
                "machine, different day."
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
