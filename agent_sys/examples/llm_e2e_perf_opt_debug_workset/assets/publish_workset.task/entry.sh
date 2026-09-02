#!/bin/sh
# A SHIM. The body it runs is the real package's, not a copy of it.
#
# A debug harness that tests a *copy* of a body is testing the wrong body, and a
# second copy of `publish.py` is a second thing to keep in step. Symlinking the
# asset directory was tried first and does not work: `spec_loader/assets.py:105`
# walks `assets_root.rglob("*")`, and `rglob` does not descend into symlinked
# directories, so the bodies were simply never found.
#
# So this package duplicates only the two files the assets convention *requires*
# — a readme and an entry — and both are thin. All real logic stays in one place.
set -eu
: "${KFO_REAL_PACKAGE:?pass --var real_package=<abs path to .../llm_e2e_performance_optimization/kernel-opt-demo>}"
exec python3 "$KFO_REAL_PACKAGE/assets/publish_workset.task/publish.py"
