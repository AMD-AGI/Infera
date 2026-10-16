#!/usr/bin/env bash
# The infera kv-aware router. RUNS INSIDE THE ENGINE CONTAINER; deploy.sh stages
# it and starts it with `docker exec -d`. From the infera repository's
# examples/sglang_mix_glm5.3/engine/up.sh step 7, with the router bound to the
# node IP (not 0.0.0.0) and E2E_KIT_ROUTER_EXTRA_ARGS appended last.
set -u
: "${MY_IP:?}" "${ROUTER_PORT:?}" "${ETCD_PORT:?}" "${MODEL:?}" "${LOG:?}"
: "${ROUTER_EXTRA_ARGS:=}"
read -r -a EXTRA_A <<< "$ROUTER_EXTRA_ARGS"
ARGV=(python3 -m infera.server --host "$MY_IP" --port "$ROUTER_PORT"
  --discovery-backend etcd --etcd-endpoint "$MY_IP:$ETCD_PORT"
  --request-transport http --kv-event-transport zmq
  --router-policy kv-aware --router-tokenizer-path "$MODEL"
  "${EXTRA_A[@]}")
mkdir -p "$(dirname "$LOG")"
{ printf '[router] argv:'; printf ' %q' "${ARGV[@]}"; echo; } > "$LOG"
exec "${ARGV[@]}" >> "$LOG" 2>&1
