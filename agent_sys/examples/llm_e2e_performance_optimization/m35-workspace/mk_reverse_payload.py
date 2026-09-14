#!/usr/bin/env python3
"""Build a REVERSE-OPTIMISATION payload for a degraded module 4, from the image.

**The payload is the engine's own module plus log lines that change no
arithmetic.** Not the workset's `baseline`: a real module-3 baseline is
harness-shaped (it defines `run(...)` and not the module's own surface), and
`apply_patch.task/apply.py:800,828` refuses any replacement that drops a
definition the stock module defined. A stock-module-shaped payload drops
nothing, which is the only payload shape that has ever passed `apply` anywhere.

**`--base-file` by construction, never `--base-sha256`.** The base hash is
computed from the bytes this program pulled out of the image, and the payload is
built from those same bytes, so the two cannot disagree. Supplying a hash by
hand is how the first round dismantled the one guard that was working: every
step was defensible and the combination let a non-Python file through.

**It refuses rather than returning empty.** Every check below exits non-zero
with the reason. A tool that returns a well-formed answer on bad input is the
dangerous kind.

Checks, all of which must pass before anything is written as final:

1. the pulled file parses as Python (`ast.parse`);
2. the payload parses as Python;
3. the payload drops no public module-level definition the stock file had —
   the same `_module_surface` rule `apply.py` uses, reimplemented here so this
   program refuses before a bring-up rather than after one;
4. the payload's hash differs from the base hash (`apply.py:853` refuses two
   identical arms);
5. if `--function` is given, the first-call marker was actually inserted.

Nothing is deleted. Output goes under a path the caller supplies.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

MARKER_PREFIX = "E2E_REVOPT"


def die(message: str) -> NoReturn:
    print(f"mk_reverse_payload: {message}", file=sys.stderr)
    raise SystemExit(1)


def module_surface(source: str) -> tuple[set[str], set[str]]:
    """`(defined, reexported)` — `apply_patch.task/apply.py:_module_surface`.

    Reimplemented rather than imported: this runs on the login node against a
    file that is not in a staged handoff yet, and the point is to answer
    `apply.py`'s question before a bring-up is spent finding out.
    """
    defined: set[str] = set()
    reexported: set[str] = set()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            reexported.update((a.asname or a.name).split(".")[0] for a in node.names)
    public = lambda names: {n for n in names if not n.startswith("_")}  # noqa: E731
    return public(defined), public(reexported) - public(defined)


def pull(image: str, inside: str) -> bytes:
    """The file's exact bytes, out of the image. `docker create` starts nothing.

    Streamed through `tar -xO` rather than copied to a directory: the file is
    not wanted on disk, only its bytes, so there is nothing to clean up and no
    second locality question. The idiom is
    `optimize_kernel.task/mock_adapt.py:_hash_from_image`'s, copied.
    """
    script = "\n".join([
        # **`pipefail`, and it is not decoration.** `docker cp ... | tar -xO` is a
        # pipeline whose exit status is `tar`'s. A failing `docker cp` with a
        # succeeding `tar` on empty input returns 0 with empty stdout, so the
        # returncode guard below would pass and the *second* guard would catch it
        # with the wrong message ("is empty" rather than "could not read").
        # Same family as m2's `[ -n "$(ls -A)" ]`: the failure survives into a
        # well-formed value.
        "set -e -o pipefail",
        f"CID=$(docker create '{image}' true)",
        "trap 'docker rm -f $CID >/dev/null 2>&1' EXIT",
        f'docker cp "$CID:{inside}" - | tar -xO',
    ])
    proc = subprocess.run(["bash", "-c", script], capture_output=True)
    if proc.returncode != 0:
        die(f"could not read {inside} out of {image}:\n"
            + proc.stderr.decode("utf-8", "replace").strip())
    if not proc.stdout:
        die(f"{inside} in {image} is empty — refusing rather than writing a zero-byte payload")
    return proc.stdout


def insert_first_call(source: str, function: str, marker: str) -> str:
    """A print as the first statement of `function` (or `Class.method`).

    An append cannot do this and `check_patch_live`'s `first_call` regex is
    matched **in the engine log**, not in the file — so the marker has to be a
    statement that executes when the operator is called.
    """
    parts = function.split(".")
    tree = ast.parse(source)

    def find(body, names):
        for node in body:
            if isinstance(node, ast.ClassDef) and names and node.name == names[0]:
                return find(node.body, names[1:])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and len(names) == 1 and node.name == names[0]:
                return node
        return None

    target = find(tree.body, parts)
    if target is None:
        die(f"--function {function!r} is not defined in this module; "
            f"nothing was written. Name it exactly as it appears, e.g. 'Sampler.forward'.")

    lines = source.splitlines(keepends=True)
    # Insert after the def line and after a docstring if there is one, so the
    # docstring stays a docstring.
    anchor = target.body[0]
    if isinstance(anchor, ast.Expr) and isinstance(anchor.value, ast.Constant) \
            and isinstance(anchor.value.value, str):
        insert_at = anchor.end_lineno          # 1-based, insert after it
    else:
        insert_at = anchor.lineno - 1          # insert before the first statement
    indent = " " * anchor.col_offset
    stmt = (f'{indent}print("{marker}", flush=True)  '
            f'# {MARKER_PREFIX}: log only, no arithmetic changed\n')
    lines.insert(insert_at, stmt)
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--container-path", required=True,
                    help="the path INSIDE the image, fully expanded (no @PLACEHOLDER@)")
    ap.add_argument("--operator", required=True, help="operator_id, used only in the markers")
    ap.add_argument("--function", default=None,
                    help="the entry function to carry the first_call marker, e.g. 'Sampler.forward'. "
                         "Omit and check_patch_live loses its second layer — see --help of that validator.")
    ap.add_argument("--delegate-to", default=None,
                    help="the module-level symbol a top-level `run(**inputs)` should delegate to. "
                         "REQUIRED unless --no-run-delegation: check_speedup_substantiated execs "
                         "this file as the harness's `--impl` and the harness raises "
                         "\"the Definition's <label> defines no `run`\" without it "
                         "(build_workset.task/harness/_common.py:283-294).")
    ap.add_argument("--no-run-delegation", action="store_true",
                    help="build without a `run` delegation and accept that "
                         "check_speedup_substantiated cannot measure this payload. Explicit so "
                         "the omission cannot be silent.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not args.delegate_to and not args.no_run_delegation:
        die("pass --delegate-to <symbol> or --no-run-delegation.\n"
            "  Two validators pull this file in opposite directions and both must be satisfied:\n"
            "    apply_patch/apply.py:828        the payload must keep the stock module's whole\n"
            "                                    public surface, so it must BE the stock module;\n"
            "    check_speedup_substantiated     execs it as the harness `--impl`, which requires\n"
            "                                    a top-level `run(**inputs)`.\n"
            "  The only file satisfying both is the stock module plus an appended `run` that\n"
            "  delegates to the operator's public symbol. A call_site_fragment operator has no\n"
            "  such symbol, which is a second independent reason it cannot be used here.")

    # **Created after every check, not before.** The first version made it here
    # and a refusal then left an empty directory behind while the message said
    # "nothing was written" — true of content, false of the filesystem, and this
    # package charges for that distinction.
    out = Path(args.out)

    raw = pull(args.image, args.container_path)
    try:
        stock = raw.decode("utf-8")
    except UnicodeDecodeError:
        die(f"{args.container_path} is not UTF-8 text; this mechanism only replaces Python source")
    try:
        ast.parse(stock)
    except SyntaxError as exc:
        die(f"{args.container_path} does not parse as Python (line {exc.lineno}); "
            "refusing — a payload built on it could not be an overlay")

    # **The base hash comes from these bytes and nowhere else.**
    base_sha256 = hashlib.sha256(raw).hexdigest()

    import_marker = f"{MARKER_PREFIX}_IMPORT {args.operator} rev1"
    first_call_marker = f"{MARKER_PREFIX}_FIRST_CALL {args.operator} rev1"

    payload = stock
    if args.function:
        payload = insert_first_call(payload, args.function, first_call_marker)
    payload += (
        f'\n\n# ----- {MARKER_PREFIX}: reverse optimisation, added by mk_reverse_payload.py -----\n'
        f'# This file is the engine\'s own module with log lines added and NO arithmetic\n'
        f'# changed. Measured against itself it is 1.0 by construction. It exists so the\n'
        f'# chain has a real-shaped artefact to carry; it is not an optimisation.\n'
        f'print("{import_marker}", flush=True)\n'
    )

    if args.delegate_to:
        # **The harness execs this file in a bare namespace and looks up `run`.**
        # `*args, **kwargs` rather than the Definition's input names: the
        # delegation must not restate a signature it does not own, and a
        # mismatch between these names and the Definition's `inputs` would be a
        # silent wrong-arity failure at measurement time. The caller is
        # responsible for `--delegate-to` naming a symbol whose parameters
        # accept the Definition's inputs; the check below only establishes that
        # the symbol exists at module level.
        payload += (
            f'\n\ndef run(*args, **kwargs):\n'
            f'    """Harness entry point. Delegates; adds nothing."""\n'
            f'    return {args.delegate_to}(*args, **kwargs)\n'
        )

    # ---- the checks, before anything is called final --------------------------
    # **`compile`, not `ast.parse`, and matching the validator exactly.**
    # `check_overlay_applies.validator/check.py:116` runs
    # `compile(source, rel, "exec")` under `compile_python: true` and catches
    # `SyntaxError, ValueError`. `ast.parse` is the weaker instrument — it is a
    # parse and nothing more — so using it here would let this program pass
    # something the validator later refuses, which is the whole failure this
    # check exists to move earlier.
    try:
        compile(payload, args.container_path, "exec")
    except (SyntaxError, ValueError) as exc:
        die(f"the payload does not compile ({type(exc).__name__}: {exc}) — "
            "check_overlay_applies would refuse it")

    stock_defined, stock_reexported = module_surface(stock)
    new_defined, new_reexported = module_surface(payload)
    dropped = stock_defined - (new_defined | new_reexported)
    lost_reexports = stock_reexported - (new_defined | new_reexported)
    if dropped:
        die("the payload drops definition(s) the stock module provides: "
            + ", ".join(sorted(dropped))
            + "\n  apply.py:828 refuses exactly this. Nothing was written as final.")

    payload_bytes = payload.encode("utf-8")
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if payload_sha256 == base_sha256:
        die("the payload is byte-identical to the stock file; apply.py:853 refuses "
            "two identical arms")

    if args.function and first_call_marker not in payload:
        die("--function was given but the first_call marker is not in the payload")

    if args.delegate_to:
        # The symbol must be defined at module level in the STOCK file — a name
        # only reachable through an import would make `run` a NameError the
        # moment the harness execs it, and that failure would arrive as
        # "the candidate measured 0 shapes" rather than as this sentence.
        if args.delegate_to not in stock_defined:
            die(f"--delegate-to {args.delegate_to!r} is not a public module-level definition of "
                f"{args.container_path}.\n  It defines: " + ", ".join(sorted(stock_defined))
                + "\n  A `run` delegating to a name this module does not define is a NameError "
                  "at measurement time, reported as an empty result.")
        if "run" not in new_defined:
            die("the payload defines no top-level `run` after delegation was requested — "
                "the harness would raise \"defines no `run`\"")

    out.mkdir(parents=True, exist_ok=True)
    (out / "stock.py").write_bytes(raw)
    (out / "optimized_kernel.py").write_bytes(payload_bytes)
    record = {
        "_what_this_is": (
            "A REVERSE OPTIMISATION. The payload is the engine's own module with log "
            "lines added and no arithmetic changed. Any ratio measured against it is "
            "1.0 by construction. It is NOT an optimisation and must not be quoted as one."
        ),
        "degraded": True,
        "degradation": "no campaign was run; the payload is the stock module plus log lines",
        "image": args.image,
        "container_path": args.container_path,
        "operator_id": args.operator,
        "base_sha256": base_sha256,
        "payload_sha256": payload_sha256,
        "base_sha256_source": "sha256 of the bytes this program pulled out of the image; "
                              "the payload is built from the same bytes, so the two cannot disagree",
        "runtime_marker": {
            "import": import_marker.replace(" ", r"\s+"),
            **({"first_call": first_call_marker.replace(" ", r"\s+")} if args.function else {}),
        },
        "surface": {
            "stock_defined": sorted(stock_defined),
            "dropped": [],
            "lost_reexports": sorted(lost_reexports),
        },
        "run_delegates_to": args.delegate_to,
        "expected_speedup": 1.0,
    }
    (out / "payload_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"stock      {len(raw)} bytes  sha256 {base_sha256}")
    print(f"payload    {len(payload_bytes)} bytes  sha256 {payload_sha256}")
    print(f"surface    {len(stock_defined)} public definition(s), 0 dropped"
          + (f", {len(lost_reexports)} re-export(s) not carried" if lost_reexports else ""))
    print(f"markers    import{' + first_call' if args.function else ' only (no --function)'}")
    print(f"written    {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
