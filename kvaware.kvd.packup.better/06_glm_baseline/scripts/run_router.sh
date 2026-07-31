#!/bin/bash
# Router for the CONTROL run: policy round-robin.
#
# round-robin, not kv-aware, on purpose. kvaware is OFF in this experiment, so
# the workers publish no KV events and a kv-aware scorer would have nothing to
# score. Using it anyway would add a variable for no benefit.
#
# Run as a STAGED FILE (`docker exec -d $CTR bash /run_router.sh`). The
# `docker exec -d $CTR bash -lc '...'` form does NOT persist — it exits and
# takes the router with it, leaving no process, no log and no error.
POLICY="${POLICY:-round-robin}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
ETCD="${ETCD:-10.2.122.10:2379}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
pkill -9 -f "infera.server" 2>/dev/null; sleep 2
exec python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
  --discovery-backend etcd --etcd-endpoint "$ETCD" \
  --request-transport http --kv-event-transport zmq \
  --router-policy "$POLICY" --router-tokenizer-path "$MODEL" \
  > /tmp/router.log 2>&1
