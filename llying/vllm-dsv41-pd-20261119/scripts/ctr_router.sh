#!/usr/bin/env bash
# Inside the engine container: the infera router (etcd discovery, HTTP to the workers).
#   ROUTER_POLICY  round-robin | kv-aware (default round-robin)
#   ROUTER_BACKEND python | rust (default python; 0.3.3's Rust router has no vLLM MoRIIO/NIXL P/D protocol)
#   ETCD_IP        etcd host (default this node)
set -eo pipefail
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/env.sh
: "${ROUTER_POLICY:=round-robin}" "${ROUTER_BACKEND:=python}" "${ETCD_IP:=$(node_ip)}"
exec python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
    --router-backend "$ROUTER_BACKEND" --router-policy "$ROUTER_POLICY" \
    --discovery-backend etcd --etcd-endpoint "$ETCD_IP:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq \
    --router-tokenizer-path "$MODEL" --timeout-keep-alive 600 ${ROUTER_EXTRA_ARGS:-}
