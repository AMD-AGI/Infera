#!/usr/bin/env python3
"""Everything the degraded m4 artefact needs from m3's workset, in one command.

    python3 /data/yihou/e2e_verify_20260906/m35/m3_extract.py <operator_workset content dir>

The content dir is the one holding `items/codes/workset.yaml` — a staged input
(`$AGENT_SYS_INPUT_OPERATOR_WORKSET`), or the output slot of a finished m3.

**Why a script rather than four greps.** Four facts have to agree with each other
and two of them are stop conditions, so reading them one at a time is how you get
three of four and launch. It prints, per operator:

  substitution      STOP if `call_site_fragment` — apply.py:808 hard-stops that
                    paired with `overlay_files`, and `apply_mode`'s enum in
                    workset.schema.json has exactly one value, so the pair is
                    unsatisfiable. Second, independent reason: there is then no
                    module-level symbol for `run` to delegate to.
  build_step        STOP if non-empty — apply.py:395 refuses; needs a rebuild.
  public_symbol     -> mk_reverse_payload.py --delegate-to
  target_files      -> mk_reverse_payload.py --container-path
  entry_function    -> mk_reverse_payload.py --function   (first_call marker)
  definition inputs -> the signature risk. The delegation is written
                    `run(*args, **kwargs)`, so it restates nothing, but the
                    public symbol's parameters must accept these keys. If they
                    do not, the failure arrives as an EMPTY MEASUREMENT, not as
                    a signature error — see PRE-REGISTER's pre-registered
                    reading of that outcome.

It **errors** on a missing or unreadable workset rather than printing an empty
table: a tool that returns a well-formed answer on bad input is the dangerous
kind, and this one exists to be run in a hurry.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import NoReturn

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("m3_extract: this needs PyYAML; try a python3 that has it")


def die(message: str) -> NoReturn:
    sys.exit(f"m3_extract: {message}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.exit(__doc__)
    root = Path(argv[0])
    ws = root / "items" / "codes" / "workset.yaml"
    if not ws.is_file():
        die(f"no workset.yaml at {ws}\n"
            "  Pass the handoff's CONTENT directory — the one containing items/codes/.")
    try:
        doc = yaml.safe_load(ws.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        die(f"{ws} does not parse as YAML: {exc}")

    operators = doc.get("operators") or []
    if not operators:
        die(f"{ws} lists no operators. A workset with an empty operator list is "
            "the pre-registered 'thin worklist' outcome — see PRE-REGISTER item D "
            "(repos == [], min_resolve_ratio 0.0) before charging it to a producer.")

    stops: list[str] = []
    print(f"workset: {ws}")
    print(f"operators: {len(operators)}\n")

    for op in operators:
        oid = op.get("operator_id", "<unnamed>")
        integ = op.get("integration") or {}
        edit = op.get("edit_target") or {}
        sub = integ.get("substitution")
        symbol = integ.get("public_symbol")
        targets = integ.get("target_files") or []
        root_var = integ.get("_repo_root_var") or edit.get("repo_root_var") or ""
        build_step = integ.get("build_step")
        entry = edit.get("entry_function") or ""

        print(f"=== {oid}")
        print(f"  substitution   {sub!r}")
        print(f"  public_symbol  {symbol!r}")
        print(f"  target_files   {targets}")
        print(f"  repo_root_var  {root_var!r}")
        print(f"  apply_mode     {integ.get('apply_mode')!r}")
        print(f"  build_step     {build_step!r}")
        print(f"  entry_function {entry!r}")

        # the Definition's inputs — the signature risk
        rel = op.get("definition")
        inputs: list[str] = []
        if rel:
            dpath = root / "items" / "codes" / str(rel)
            if dpath.is_file():
                try:
                    inputs = sorted((json.loads(dpath.read_text(encoding="utf-8"))
                                     or {}).get("inputs") or {})
                except (OSError, ValueError) as exc:
                    print(f"  definition     UNREADABLE: {exc}")
            else:
                print(f"  definition     MISSING at {rel}")
        print(f"  inputs         {inputs}")

        # **The baseline's SHAPE — this is the fact m35 and m2 disagree on.**
        # `--var forge_mock=1` seeds `optimized_kernel.py` from this very string
        # (`30_run_forge.sh:64-115`). Two different failures hang off it:
        #   no `def run(`            -> 30_run_forge.sh REFUSES outright
        #   `run` but no module surface -> apply.py:828 refuses at m5, AFTER a
        #                              bring-up, because the overlay drops every
        #                              definition the engine module provides
        # m2's chain-with-forge_mock works only in the second-and-not-first case
        # AND only if the baseline carries the stock module's surface. Nothing
        # else in the graph reads this before m5 does.
        baseline = ""
        if rel:
            dpath = root / "items" / "codes" / str(rel)
            if dpath.is_file():
                try:
                    baseline = (json.loads(dpath.read_text(encoding="utf-8"))
                                or {}).get("baseline") or ""
                except (OSError, ValueError):
                    baseline = ""
        if not isinstance(baseline, str) or not baseline.strip():
            print("  baseline       ABSENT or not a source string")
            mine_pre = [f"{oid}: the Definition carries no `baseline` source; "
                        "forge_mock=1 has nothing to seed from (30_run_forge.sh refuses)."]
        else:
            has_run = "def run(" in baseline
            try:
                defined = {n.name for n in ast.parse(baseline).body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                defined |= {tg.id for n in ast.parse(baseline).body
                            if isinstance(n, ast.Assign)
                            for tg in n.targets if isinstance(tg, ast.Name)}
            except SyntaxError as exc:
                defined = set()
                print(f"  baseline       DOES NOT PARSE: line {exc.lineno}")
            pub = sorted(n for n in defined if not n.startswith("_"))
            print(f"  baseline       {len(baseline)} chars, def run(: {has_run}, "
                  f"public defs: {pub[:6]}{' …' if len(pub) > 6 else ''}")
            mine_pre = []
            if not has_run:
                mine_pre.append(f"{oid}: the baseline defines no top-level `run` — "
                                "forge_mock=1 refuses (30_run_forge.sh:110).")
            elif symbol and symbol not in defined:
                mine_pre.append(
                    f"{oid}: HARNESS-SHAPED BASELINE. It defines `run` but not "
                    f"{symbol!r}, so it is not the engine module. "
                    "forge_mock=1 would seed an overlay that drops the module's "
                    "whole surface and apply.py:828 refuses AFTER a bring-up. "
                    "Use the reverse payload (mk_reverse_payload.py) instead.")

        mine: list[str] = list(mine_pre)
        if sub == "call_site_fragment":
            mine.append(f"{oid}: substitution is 'call_site_fragment' — apply.py:808 "
                         "hard-stops this with overlay_files, and there is no module-level "
                         "symbol for `run` to delegate to. UNUSABLE for the degraded m4 route.")
        if not symbol:
            mine.append(f"{oid}: no public_symbol, so --delegate-to has no target.")
        if build_step:
            mine.append(f"{oid}: declares build_step {build_step!r} — apply.py:395 refuses; "
                         "this operator needs the image rebuilt and cannot be an overlay.")
        if not targets:
            mine.append(f"{oid}: no target_files, so --container-path is unknown.")
        if not entry:
            mine.append(f"{oid}: no entry_function — the first_call marker cannot be placed, "
                         "and check_patch_live loses its second layer.")

        stops.extend(mine)
        if not mine and symbol and targets:
            tail = str(targets[0]).lstrip("/")
            print("\n  ready to run:")
            print(f"    python3 /data/yihou/e2e_verify_20260906/m35/mk_reverse_payload.py \\")
            print(f"      --image infera/engine-sglang:qwen3-local-20260906 \\")
            print(f"      --container-path <SGLANG_ROOT>/{tail} \\")
            print(f"      --operator {oid} \\")
            print(f"      --function {entry or '<Class.method>'} \\")
            print(f"      --delegate-to {symbol} \\")
            print(f"      --out /data/yihou/e2e_verify_20260906/m35/payload.{oid}")
            print("    # SGLANG_ROOT on this image = /sgl-workspace/sglang/python/sglang")
            print(f"    # then check that {symbol}'s parameters accept {inputs} —")
            print("    #   a mismatch surfaces as an EMPTY measurement, not an error")
        print()

    if stops:
        print("STOP — report to the leader before m5 spends a bring-up:")
        for s in stops:
            print(f"  - {s}")
        return 1
    print("no stop conditions found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
