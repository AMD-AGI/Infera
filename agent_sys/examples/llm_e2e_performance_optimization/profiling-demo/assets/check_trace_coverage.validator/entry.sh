#!/bin/sh
# The body of `check_trace_coverage`. It reads the manifest the capture built by
# parsing every rank, rather than re-parsing 462 MB of traces itself — which is
# why its cost tag is `minutes` for the producer and seconds here.
set -eu
exec "${AGENT_SYS_DEMO_PYTHON:-python3}" \
  "${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}/assets/check_trace_coverage.validator/check.py"
