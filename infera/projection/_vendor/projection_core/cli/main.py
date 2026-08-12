###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Compatibility shim so the vendored tuning agent can spawn the projector the
same way it did in Primus.

The tuning-agent evaluator shells out to::

    python <primus_root>/primus/cli/main.py projection inference ...

In Infera the projection CLI lives at ``infera.projection.cli``. This shim maps
the ``projection inference`` invocation onto it, so the evaluator's command
builders keep working unchanged. Only the ``inference`` suite is supported
(the Megatron ``performance`` / ``memory`` training suites were not ported).
"""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "projection":
        raise SystemExit(
            "infera projection shim: expected 'projection <suite> ...' "
            f"(got {argv!r})"
        )
    suite_argv = argv[1:]  # drop the leading 'projection'
    suite = suite_argv[0] if suite_argv else None
    if suite != "inference":
        raise SystemExit(
            f"infera projection shim: only the 'inference' suite is ported "
            f"(got {suite!r}). Training 'performance'/'memory' suites live in Primus."
        )
    from infera.projection.cli import main as _projection_main

    return _projection_main(suite_argv)


if __name__ == "__main__":
    raise SystemExit(main())
