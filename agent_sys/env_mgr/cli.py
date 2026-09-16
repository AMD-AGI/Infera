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
    subs = p.add_subparsers(dest="stage", required=True, metavar="stage")

    # The four shipped stages, unchanged in every observable way. Measured: all
    # six shipped call shapes parse identically under sub-parsers, set the same
    # `stage` / `recipe` / `--json` attributes, and preserve SystemExit(2) on an
    # invalid stage. The one difference is that a *global* flag placed before
    # the sub-command no longer parses — no shipped test or documented
    # invocation does that.
    for stage in STAGES:
        s = subs.add_parser(stage)
        s.add_argument("recipe", help="path to the recipe yaml")
        s.add_argument("--tag", action="append", default=[], dest="tags")
        s.add_argument("--installer", default=None)
        s.add_argument("--importance", default=None)
        s.add_argument("--item", default=None)
        s.add_argument("--path", default=None, help="override target.path")
        s.add_argument("--workspace", default=None, help="override target.parent.workspace")
        s.add_argument(
            "--on-conflict",
            choices=("fail", "weak"),
            default="fail",
            help="version-conflict policy. 'fail' (default) records the "
            "conflict and halts before install (exit 2); 'weak' is a v1 no-op that "
            "skips conflict detection and proceeds (exit 0).",
        )
        s.add_argument("--json", action="store_true")

    # New: domain and zone inspection (spec §9). Both are read-only, and both
    # live above the decoupling wall — this file is the only module allowed to
    # see both sides of it.
    d = subs.add_parser("domain", help="inspect registered domains")
    d.add_argument("name", nargs="?", default=None)
    d.add_argument("--meta", default=None, help="path to the env_mgr metadata file")
    d.add_argument("--json", action="store_true")

    z = subs.add_parser("zone", help="inspect a task's zones")
    z.add_argument("task_id", nargs="?", default=None, metavar="task-id")
    z.add_argument("--meta", default=None, help="path to the env_mgr metadata file")
    z.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def _inspect(args: argparse.Namespace) -> int:
    """`domain` and `zone`. Imported lazily so the shipped stages pay nothing."""
    from .inspection import render_domains, render_zones

    text = render_domains(args) if args.stage == "domain" else render_zones(args)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.stage in ("domain", "zone"):
        return _inspect(args)
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
