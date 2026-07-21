#!/bin/bash
# Launch the Infera kv-aware router on the prefill node.
# Run inside the Infera container:  bash launch/launch_router.sh
# ROUTER_PORT overridable (default 8100).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; source "$HERE/env.sh"
exec python3 -m infera.server --host 0.0.0.0 --port "${ROUTER_PORT:-8100}" \
  --discovery-backend etcd --etcd-endpoint $ETCD_EP \
  --request-transport http --router-policy kv-aware \
  --router-tokenizer-path "$MODEL"
