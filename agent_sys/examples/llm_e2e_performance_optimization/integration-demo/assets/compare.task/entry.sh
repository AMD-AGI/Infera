#!/bin/sh
# Put the two arms side by side. Pure computation over four handoffs: it touches
# no cluster, so it is the one leaf in this package that can be re-run for free
# against a finished run's store.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

exec python3 "$PKG/assets/compare.task/compare.py" \
  --accept-stock "${AGENT_SYS_INPUT_ACCEPTANCE_STOCK:?}" \
  --accept-patched "${AGENT_SYS_INPUT_ACCEPTANCE_PATCHED:?}" \
  --bench-stock "${AGENT_SYS_INPUT_BENCH_STOCK:?}" \
  --bench-patched "${AGENT_SYS_INPUT_BENCH_PATCHED:?}" \
  --patch "${AGENT_SYS_INPUT_KERNEL_PATCH:?}" \
  --out "${AGENT_SYS_OUTPUT_INTEGRATION_REPORT:?}" \
  --package "$PKG"
