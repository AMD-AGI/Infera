#!/bin/sh
# Measure the patched arm. The same script as the stock arm with one variable
# changed — running a second implementation here would make the comparison
# meaningless, because any difference it introduced would be indistinguishable
# from a difference the patch caused.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"

IT_ARM=patched \
IT_OUTPUT_ACCEPT="${AGENT_SYS_OUTPUT_ACCEPTANCE_PATCHED:?AGENT_SYS_OUTPUT_ACCEPTANCE_PATCHED is unset}" \
IT_OUTPUT_BENCH="${AGENT_SYS_OUTPUT_BENCH_PATCHED:?AGENT_SYS_OUTPUT_BENCH_PATCHED is unset}" \
exec /usr/bin/env bash "$PKG/assets/accept/measure.sh"
