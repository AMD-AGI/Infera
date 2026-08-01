#!/usr/bin/env bash
# Router on the prefill node, kv-aware with the PD-aware overlap weights.
# Same command as the kvaware/kvd baseline; only the container name differs.
set -u
CTR="${CTR:-merge_g0}"
MY_IP="${MY_IP:-10.2.122.10}"
PORT="${PORT:-8100}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
KVD_SOCK=/tmp/kvd/kvd.sock

docker exec "$CTR" bash -c "pkill -9 -f 'infera.server' 2>/dev/null; sleep 2" || true
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $PORT --discovery-backend etcd --etcd-endpoint $MY_IP:2379 --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0 --kvd-socket-path $KVD_SOCK' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"
