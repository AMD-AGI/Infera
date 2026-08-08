#!/usr/bin/env python3
"""Uniform CLI for the op optimize-loop scaffold (issue #40).

One driver, any registered op — measure / profile / tune by name:

  python loop.py measure --op moe_experts
  python loop.py profile --op moe_experts --kernels
  python loop.py tune    --op moe_experts --inject
  python loop.py measure --op moe_experts -d tokens=1 -d experts=64   # override dims
  python loop.py list

Adding an op is writing one ``OpSpec`` in ``ops/<name>.py`` (see
``ops/moe_experts.py`` as the template) — no new script. The plugin kernel is
just the candidate the op's spec points at.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import framework as fw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["measure", "profile", "tune", "list"])
    ap.add_argument("--op", help="registered op name (see `loop.py list`)")
    ap.add_argument("-d", "--dim", action="append", default=[], help="dims override k=v")
    ap.add_argument("--kernels", action="store_true", help="profile: per-kernel split")
    ap.add_argument("--inject", action="store_true", help="tune: bake winner into the plugin")
    args = ap.parse_args()

    if args.stage == "list":
        import glob
        import importlib

        for f in glob.glob(os.path.join(os.path.dirname(__file__), "ops", "*.py")):
            name = os.path.splitext(os.path.basename(f))[0]
            if not name.startswith("_"):
                importlib.import_module(f"ops.{name}")
        print("registered ops:", ", ".join(fw.list_ops()) or "(none)")
        return
    if not args.op:
        ap.error("--op is required")
    spec = fw.get_op(args.op)
    overrides = dict(kv.split("=", 1) for kv in args.dim)
    if args.stage == "measure":
        fw.measure(spec, overrides)
    elif args.stage == "profile":
        fw.profile(spec, overrides, kernels=args.kernels)
    elif args.stage == "tune":
        fw.tune(spec, overrides, inject=args.inject)


if __name__ == "__main__":
    main()
