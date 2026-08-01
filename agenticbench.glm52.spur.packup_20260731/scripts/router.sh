#!/bin/bash
# infera router on the prefill node, policy kv-aware.
# From pr.final/scripts/kvaware_start_router.sh, spur-shaped.
#
# ALWAYS pass a fresh port on a restart. A router whose circuit breaker is still
# open returns 503 in ~0.4 s, which looks exactly like a backend failure. Read
# the latency: ~0.4 s is the breaker, 12-23 s is a real backend fault.
#
# Usage: router.sh [port]
set -eu
PORT="${1:-8180}"
JOB=19254
MY_IP=10.245.153.38
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
KVD_SOCK=/tmp/kvd/kvd.sock
CTR=agbench

spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
set -e
# Kill a previous router by its OWN pid only. A bare pgrep -f 'infera.server'
# matches the `docker exec bash -c ...` command string that CONTAINS that text,
# i.e. this very shell -- it silently killed itself here and `set -e` aborted the
# whole script with no output. Match the python process, exclude this pid.
docker exec $CTR bash -c \"pgrep -f 'python3 -m infera.server' | grep -v \\\$\\\$ | xargs -r kill -9 2>/dev/null; true\"
docker exec $CTR bash -c \"printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $PORT --discovery-backend etcd --etcd-endpoint $MY_IP:2379 --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0 --kvd-socket-path $KVD_SOCK' > /run_router.sh && chmod +x /run_router.sh\"
docker exec -d $CTR bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec $CTR bash -c \"curl -sf -m5 http://$MY_IP:$PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }\"
echo '  --- registered workers ---'
docker exec $CTR bash -c \"grep -oE 'registered.{0,90}' /tmp/router.log | head\"
" 2>&1 | grep -v libtinfow
