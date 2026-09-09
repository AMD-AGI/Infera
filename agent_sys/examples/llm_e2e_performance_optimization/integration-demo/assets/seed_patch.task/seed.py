#!/usr/bin/env python3
"""Cut a semantics-preserving patch against one file, and package it as `kernel_patch`.

Runs on the login node, on a copy of the file `entry.sh` pulled out of the image.

**What "semantics-preserving" buys.** The mock adds two log lines and changes no
arithmetic, so the expected verdict is knowable in advance: correctness deltas
exactly zero, performance deltas inside the noise. That is what makes the
validators downstream checkable -- a comparison that reports a regression on this
input has a fault in the judgement, not in the thing judged. A mock that made the
model faster would test nothing except itself.

**Two markers, not one.** The import marker proves the interpreter compiled the
mounted bytes. The first-call marker proves the patched function was entered by a
real request. Only the second answers the question the integration stage exists
to ask, and it is the one a patch can most easily fail to satisfy -- a kernel
mounted into a code path the model never takes gives two identical arms and a
green report.

The first-call marker is guarded by a module-level boolean, so the steady-state
cost is one attribute load and one branch per call. That matters: this file is
also the performance baseline, and a marker that logged on every call would move
the number it is supposed to leave alone.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import difflib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import patchkit  # noqa: E402

REV = "mock-rev1"
IMPORT_MARKER = "INTEGRATION_PATCH_IMPORT"
FIRST_CALL_MARKER = "INTEGRATION_PATCH_FIRST_CALL"
FLAG = "_ITG_PATCH_FIRST_CALL_DONE"

#: Where the import marker goes. sglang modules almost all have this line, and
#: it is the one place in a module guaranteed to be after the imports the marker
#: needs and before any class that might use it.
LOGGER_ANCHOR = "logger = logging.getLogger(__name__)"


def _emit(indent: str, text: str, use_logger: bool) -> list[str]:
    call = f'logger.warning("{text}")' if use_logger else f'print("{text}", flush=True)'
    return [f"{indent}{call}\n"]


def insert_import_marker(lines: list[str], operator: str) -> tuple[list[str], bool]:
    """Put the import marker and the first-call flag at module level.

    Returns (lines, used_logger). Falls back to `print` when the module has no
    logger, because a marker that does not appear is worse than an ugly one.
    """
    for i, line in enumerate(lines):
        if line.strip() == LOGGER_ANCHOR:
            block = [
                "\n",
                "# --- integration-demo mock patch -----------------------------------------\n",
                f"# Semantics-preserving. Two markers so `check_patch_live` can tell "
                f"'the bytes are mounted'\n",
                "# from 'the code ran'. The flag makes the per-call cost one branch.\n",
                f"{FLAG} = False\n",
                *_emit("", f"{IMPORT_MARKER} {operator} {REV}", True),
                "# -------------------------------------------------------------------------\n",
            ]
            return lines[: i + 1] + block + lines[i + 1 :], True

    # No logger in this module. Put the block after the last top-level import.
    last = 0
    tree = ast.parse("".join(lines))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno or node.lineno)
    block = [
        "\n",
        "# --- integration-demo mock patch -----------------------------------------\n",
        f"{FLAG} = False\n",
        *_emit("", f"{IMPORT_MARKER} {operator} {REV}", False),
        "# -------------------------------------------------------------------------\n",
    ]
    return lines[:last] + block + lines[last:], False


def locate_body_start(source: str, symbol: str) -> tuple[int, str]:
    """First statement of `Class.method` or `function`, as (0-based line, indent).

    Parsed with `ast` rather than matched with a regex, because the signature of
    the function this targets spans nine lines and "the line after `def`" is
    somewhere in the middle of its arguments.

    A docstring is stepped over. Inserting before it would leave the string as an
    ordinary expression statement and silently delete the docstring.
    """
    tree = ast.parse(source)
    parts = symbol.split(".")
    if len(parts) == 2:
        cls, fname = parts
        holder = next(
            (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None
        )
        if holder is None:
            raise SystemExit(f"seed: no class {cls!r} in the target file")
        scope = holder.body
    elif len(parts) == 1:
        fname, scope = parts[0], tree.body
    else:
        raise SystemExit(f"seed: --symbol {symbol!r} is neither 'name' nor 'Class.name'")

    fn = next(
        (
            n
            for n in scope
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fname
        ),
        None,
    )
    if fn is None:
        raise SystemExit(f"seed: no function {fname!r} in {symbol!r}")

    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
        if not body:
            raise SystemExit(f"seed: {symbol} has only a docstring, nothing to instrument")
    first = body[0]
    return first.lineno - 1, " " * first.col_offset


def insert_first_call_marker(lines: list[str], symbol: str, operator: str,
                             use_logger: bool) -> list[str]:
    at, indent = locate_body_start("".join(lines), symbol)
    block = [
        f"{indent}global {FLAG}\n",
        f"{indent}if not {FLAG}:\n",
        f"{indent}    {FLAG} = True\n",
        *_emit(indent + "    ", f"{FIRST_CALL_MARKER} {operator} {REV}", use_logger),
    ]
    return lines[:at] + block + lines[at:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True)
    ap.add_argument("--container-path", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--package", required=True)
    args = ap.parse_args()

    stock = Path(args.stock)
    out = Path(args.out)
    operator = args.symbol.split(".")[-1] + "_marker"
    # Placeholder form from here on. A manifest naming
    # /sgl-workspace/... cannot be published: the seal refuses absolute paths
    # outside its allow-list and scans every file, not only the README.
    placeholder_path = patchkit.contract(args.container_path)
    _, rel = patchkit.split_placeholder(placeholder_path)

    original = stock.read_text(encoding="utf-8").splitlines(keepends=True)
    patched, used_logger = insert_import_marker(list(original), operator)
    patched = insert_first_call_marker(patched, args.symbol, operator, used_logger)

    if patched == original:
        print("seed: the patch changed nothing", file=sys.stderr)
        return 1
    # Compiling here rather than letting the engine find out: a syntax error in a
    # mounted file takes the worker down during model import, fifteen minutes
    # later, where it reads as a model-loading failure.
    compile("".join(patched), rel, "exec")

    diff = "".join(
        difflib.unified_diff(original, patched, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3)
    )

    codes = out / "items" / "codes"
    (codes / "patches").mkdir(parents=True, exist_ok=True)
    (codes / "notes").mkdir(parents=True, exist_ok=True)
    patch_name = f"0001-{operator}.patch"
    (codes / "patches" / patch_name).write_text(diff, encoding="utf-8")

    manifest = {
        "schema_version": patchkit.SCHEMA_VERSION,
        "operator_id": operator,
        "logical_operator": args.symbol,
        "image": args.image,
        "apply_mode": patchkit.APPLY_OVERLAY,
        "files": [
            {
                "container_path": placeholder_path,
                "base_sha256": patchkit.sha256_file(stock),
                "patch": patch_name,
                "change": "modify",
            }
        ],
        # Optional in the contract, supplied here, and the reason the mock is
        # worth having: without it `check_patch_live` can only prove the bytes
        # are in place, never that they ran.
        "runtime_marker": {
            "import": rf"{IMPORT_MARKER}\s+{operator}\s+{REV}",
            "first_call": rf"{FIRST_CALL_MARKER}\s+{operator}\s+{REV}",
        },
        # What a real forge run would claim. Zeroed here and labelled, so nothing
        # downstream mistakes the mock's numbers for a measurement.
        "expect": {
            "source": "mock",
            "speedup": 1.0,
            "snr_db": None,
            "baseline_wall_ms": None,
            "optimized_wall_ms": None,
        },
        "provenance": {
            "operator_workset": None,
            "workset_evidence": None,
            "produced_by": "integration-demo/assets/seed_patch.task",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    (codes / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (codes / "notes" / "forge_result.json").write_text(
        json.dumps(
            {
                "source": "mock",
                "note": (
                    "The kernel-optimization stage had not landed when this package was "
                    "written, and analyze-demo's design leaves the schema of this file "
                    "undefined. It is a placeholder with the shape the contract expects."
                ),
                "correctness": {"snr_db": None, "allclose": None},
                "performance": {"baseline_wall_ms": None, "optimized_wall_ms": None},
                "iterations": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (codes / "notes" / "program.md").write_text(
        "# What this patch changes\n\n"
        f"`{args.symbol}` in `{rel}` gains two log lines and nothing else. The first\n"
        "runs at import, the second on the first call and never again — a module-level\n"
        "boolean guards it, so the steady-state cost is one branch.\n\n"
        "No arithmetic changes. That is deliberate: this patch exists to exercise the\n"
        "integration pipeline, and the only way to check a regression detector is to\n"
        "feed it a change whose correct answer is known. Here the correct answer is\n"
        "\"no difference\".\n",
        encoding="utf-8",
    )

    (out / "items" / "watchout").write_text(
        "This is a MOCK of the kernel-optimization stage's deliverable, not an optimised\n"
        "kernel. It changes no arithmetic, so every number measured against it should\n"
        "match the unpatched arm. Do not read a green integration report produced from\n"
        "this patch as evidence that a real optimisation would pass — it is evidence\n"
        "that the pipeline works and that the comparison does not invent regressions.\n"
        "\n"
        "The patch is cut against one image, named in manifest.json, and pinned by the\n"
        "sha256 of the file inside it rather than by a git commit: both repositories in\n"
        "that image are git working trees that are dirty relative to their own HEAD,\n"
        "because the build replaces the sglang python tree wholesale with a PR overlay.\n"
        "\n"
        "apply_mode is overlay_files. A patch that has to be compiled — HIP, CK,\n"
        "assembly — cannot be delivered this way and needs the image rebuilt.\n",
        encoding="utf-8",
    )

    (out / "README.md").write_text(
        f"""# kernel_patch

## Purpose

The deliverable of the kernel-optimization stage, in the form the integration
stage consumes: a patch set against one named engine image, plus a manifest
saying which operator it optimises, which files it touches, what each of those
files hashed before the patch, and how it is meant to be applied.

**This copy is a mock.** It was produced by `seed_patch` because the
kernel-optimization stage has not landed. It instruments
`{args.symbol}` with two log markers and changes no arithmetic. The
contract it satisfies is real; the optimisation is not.

## Interface

`items/codes/manifest.json` is the entry point. Its `files[]` array is the
work list: for each entry, take `patches/<patch>`, apply it with `patch -p1`
to a tree in which the file sits at its path relative to its container root,
and check that the file it was applied to hashed `base_sha256` first. A
mismatch means the patch was cut against a different image and must not be
applied.

`apply_mode` says how the result reaches the engine. `overlay_files` means each
patched file is bind-mounted read-only over its `container_path`; it works
because sglang is an editable install, so the interpreter reads the tree in the
image directly. `rebuild` means the image has to be built again and is not
implemented by the integration stage.

`runtime_marker` is optional and carries two regexes — one the patched module
logs at import, one it logs the first time the patched code runs. A consumer
that can see the engine log can use them to prove the patch was not merely
mounted but entered. Without them, only the mount can be proven.

`notes/forge_result.json` is where a real run's correctness and performance
report goes. Here it is a labelled placeholder.

## Boundary

This handoff carries a patch, not a kernel and not a build. It does not say
whether the optimisation is correct or fast — `workset_evidence` upstream
measured the operator standalone, and the integration stage measures the
service. It does not carry the source tree it patches; the tree is in the image
named by `image`, and the hashes are what tie the two together.

It cannot express a deletion, and it cannot express a change that needs
compiling. Both are stated rather than silently dropped: a patch that needs
either fails at `check_patch_shape` or at `apply_patch`, naming the reason.
""",
        encoding="utf-8",
    )

    # Everything published here was generated from an image, not from this
    # machine, so there is little for the seal to object to -- but the package
    # path reaches the notes through `produced_by` and the zone path could reach
    # a traceback. Redacting unconditionally is cheaper than deciding.
    subprocess.run(
        [
            sys.executable,
            str(Path(args.package) / "assets" / "lib" / "redact.py"),
            str(out),
            f"TASK_PACKAGE={args.package}",
            f"WORK_ROOT={Path.cwd()}",
            "TMPDIR=/tmp",
            f"HOME={Path.home()}",
        ],
        check=True,
    )

    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    print(f"seed: {patch_name}, +{added} lines, base {manifest['files'][0]['base_sha256'][:12]}…")
    print(f"seed: markers import={IMPORT_MARKER} first_call={FIRST_CALL_MARKER} operator={operator}")
    shutil.rmtree(stock.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
