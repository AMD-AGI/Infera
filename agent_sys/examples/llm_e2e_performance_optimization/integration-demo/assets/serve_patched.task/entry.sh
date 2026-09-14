#!/bin/sh
# The patched arm: every entry in the mount plan becomes a read-only bind mount.
#
# Its first act is to tear the stock deployment down, which is why its edge in
# the graph points at `measure_stock` rather than at `serve_stock` — the edge has
# to mean "the stock arm has been measured", not "the stock arm exists".
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

IT_ARM=patched \
IT_OUTPUT_DIR="${AGENT_SYS_OUTPUT_DEPLOYMENT_PATCHED:?AGENT_SYS_OUTPUT_DEPLOYMENT_PATCHED is unset}" \
exec /usr/bin/env bash "$PKG/assets/serve/round.sh"
