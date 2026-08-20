# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""env-mgr CLI: parse stage/filters, run, report, exit code."""

from __future__ import annotations

import argparse
import sys

from .outcome import Outcome
from .recipe import RecipeError, load_recipe
from .report import render_human, render_json
from .runner import STAGES, Filters, run


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="env-mgr")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("recipe", help="path to the recipe yaml")
    p.add_argument("--tag", action="append", default=[], dest="tags")
    p.add_argument("--installer", default=None)
    p.add_argument("--importance", default=None)
    p.add_argument("--item", default=None)
    p.add_argument("--path", default=None, help="override target.path")
    p.add_argument("--workspace", default=None, help="override target.parent.workspace")
    p.add_argument(
        "--on-conflict",
        choices=("fail", "weak"),
        default="fail",
        help="cross-layer version-conflict policy. 'fail' (default) records the "
        "conflict and halts before install (exit 2); 'weak' is a v1 no-op that "
        "skips conflict detection and proceeds (exit 0).",
    )
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        target, items = load_recipe(args.recipe)
        if args.path:
            target.path = args.path
        if args.workspace:
            target.parent["workspace"] = args.workspace
        filters = Filters(
            tags=args.tags,
            installer=args.installer,
            importance=args.importance,
            item=args.item,
        )
        outcomes, status = run(target, items, args.stage, filters, on_conflict=args.on_conflict)
    except (RecipeError, KeyError, ValueError) as e:
        outcomes = [Outcome("fail", f"{type(e).__name__}: {e}")]
        status = "FAIL"
    text = render_json(outcomes, status) if args.json else render_human(outcomes, status)
    print(text)
    return 2 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
