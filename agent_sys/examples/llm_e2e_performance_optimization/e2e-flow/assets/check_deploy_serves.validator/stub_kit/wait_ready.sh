#!/bin/sh
# Blocks until the service answers. EXITS NON-ZERO ON TIMEOUT -- a readiness wait
# that exits 0 when it gave up turns "never loaded" into "measured nothing".
set -eu
. "$(dirname "$0")/stub_env.sh"
: "${KIT_READY_TIMEOUT_S:=60}"
i=0
while [ "$i" -lt "$KIT_READY_TIMEOUT_S" ]; do
  if curl -sf -m 2 "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1; then
    echo "ready after ${i}s"; exit 0
  fi
  i=$((i + 1)); sleep 1
done
echo "NOT ready after ${KIT_READY_TIMEOUT_S}s" >&2
exit 1
