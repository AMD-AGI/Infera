#!/bin/bash
# infera router on the prefill node, with a selectable policy.
#
# Generalises router.sh, which hardcoded --router-policy kv-aware and the dead
# 28485 allocation's job/IP.
#
# POLICY=kv-aware  ->  cost = w*(request_blocks - hits) + active_blocks, with the
#   product-default weights (prefill 20.0 / decode 2.0). At par8's 0.89 hit rate
#   and page_size 64 the prefill overlap term dominates the load term ~10x, which
#   is the regime those weights are designed for.
# POLICY=round-robin -> stateless rotation over EXPANDED targets, i.e. over
#   (worker, dp_rank) pairs, with an INDEPENDENT counter per candidate set. So a
#   PD request's prefill pool and decode pool rotate separately
#   (round_robin.py:26-32 spells out why a single shared counter would pin every
#   prefill pick when there are exactly two pools).
#
# There is no per-role policy switch: --router-policy is one global value
# (infera/server/args.py:88, and the Rust side rejects anything else at
# rust/router/src/config.rs:72). "prefill and decode both round-robin" is
# therefore exactly `--router-policy round-robin`.
#
# ALWAYS pass a fresh port on a restart. A router whose circuit breaker is still
# open returns 503 in ~0.4 s, which looks exactly like a backend failure. Read
# the latency: ~0.4 s is the breaker, 12-23 s is a real backend fault.
#
# Usage: JOB=<job> MY_IP=<prefill ip> [POLICY=..] [PORT=..] ab_router.sh
set -eu
PORT="${PORT:-8190}"
JOB="${JOB:?JOB=prefill node job id}"
MY_IP="${MY_IP:?MY_IP=prefill node ens3 ip}"
POLICY="${POLICY:-kv-aware}"
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
KVD_SOCK=/tmp/kvd/kvd.sock
CTR="${CTR:-agbench_mtp}"

# --router-tokenizer-path is REQUIRED unconditionally by the argument parser, not
# just by kv-aware -- omitting it on the round-robin arm exits 2 with a usage
# dump. round-robin then discards it (_build_round_robin(**_) drops every kwarg),
# so it is inert there, but it must be on the command line.
#
# The overlap weights ARE kv-aware-only, and are passed only there: on the
# round-robin arm they would read as an active knob in the recorded command line
# when nothing consumes them.
POLICY_ARGS="--router-policy $POLICY --router-tokenizer-path $MODEL"
if [ "$POLICY" = "kv-aware" ]; then
  POLICY_ARGS="$POLICY_ARGS --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0"
fi

spur exec "$JOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
set -e
# Kill a previous router by its OWN pid only. A bare pgrep -f 'infera.server'
# matches the \`docker exec bash -c ...\` command string that CONTAINS that text,
# i.e. this very shell -- it silently killed itself here once and \`set -e\`
# aborted the whole script with no output.
docker exec $CTR bash -c \"pgrep -f 'python3 -m infera.server' | grep -v \\\$\\\$ | xargs -r kill -9 2>/dev/null; true\"
docker exec $CTR bash -c \"printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $PORT --discovery-backend etcd --etcd-endpoint $MY_IP:2379 --request-transport http --kv-event-transport zmq $POLICY_ARGS --kvd-socket-path $KVD_SOCK' > /run_router.sh && chmod +x /run_router.sh\"
docker exec -d $CTR bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec $CTR bash -c \"curl -sf -m5 http://$MY_IP:$PORT/health && echo '  router healthy (policy=$POLICY)' || { echo '  router not ready'; tail -20 /tmp/router.log; }\"
echo '  --- policy actually in force, read back from the router log ---'
docker exec $CTR bash -c \"grep -oE 'router-policy=[a-z-]+.*' /tmp/router.log | head -2\"
echo '  --- registered workers ---'
docker exec $CTR bash -c \"grep -oE 'registered.{0,90}' /tmp/router.log | head\"
" 2>&1 | grep -v libtinfow
