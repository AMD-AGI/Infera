#!/bin/bash
# Router for the cross-restart experiment: policy kv-aware, default weights.
# The SAME router configuration is used in both rounds — the only thing that
# changes between them is that the engine legs were killed and relaunched.
#
# Run as a STAGED FILE (`docker exec -d $CTR bash /run_router.sh`). The
# `docker exec -d $CTR bash -lc '...'` form does NOT persist — it exits and
# takes the router with it, leaving no process, no log and no error.
POLICY="${POLICY:-kv-aware}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
ETCD="${ETCD:-10.2.122.10:2379}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
pkill -9 -f "infera.server" 2>/dev/null; sleep 3
exec python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
  --discovery-backend etcd --etcd-endpoint "$ETCD" \
  --request-transport http --kv-event-transport zmq \
  --router-policy "$POLICY" --router-tokenizer-path "$MODEL" \
  > /tmp/router.log 2>&1
