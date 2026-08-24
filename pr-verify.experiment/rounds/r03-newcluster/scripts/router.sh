#!/usr/bin/env bash
# Start the infera router for the 2-node 1P1D rig, with round-robin routing.
#
# Why this exists instead of using common.sh:start_router --
#   common.sh:91 passes --router-tokenizer-path only on the kv-aware branch, but
#   infera/server/args.py:140 declares it required=True unconditionally. So a
#   round-robin router never starts:
#     __main__.py: error: the following arguments are required: --router-tokenizer-path
#   common.sh offers no hook to add a flag, and patching the shared kit is out of
#   scope for this experiment, so the router is started here instead. Everything
#   else matches common.sh:start_router.
#
# round-robin, not kv-aware, is deliberate: kv-aware routing is orthogonal to the
# KV-transfer race under test and would add a variable to the A/B.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_CMD="${SSH_CMD:-$HERE/spur_ssh.sh}"

NODE="${PREFILL_NODE:-crsuse2-m2m-237}"
IP="${PREFILL_IP:-10.245.154.191}"
CTR="${CTR:-glm52_pd}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
ETCD_PORT="${ETCD_PORT:-2379}"
BACKEND="${ROUTER_BACKEND:-rust}"
POLICY="${ROUTER_POLICY:-round-robin}"
TOKENIZER="${TOKENIZER:-/mnt/m2m_nobackup/models/zai-org__GLM-5.2-FP8}"

# Kill a previous router by pid only -- a bare `pkill -f infera.server` also matches
# the `docker exec bash -c ...` command string that contains that text, i.e. this shell.
$SSH_CMD "$NODE" "docker exec $CTR bash -c \"pgrep -f 'python3 -m infera.server' | xargs -r kill -9 2>/dev/null; true\""
sleep 2

$SSH_CMD "$NODE" "docker exec -d $CTR bash -c \"nohup python3 -m infera.server \
  --host 0.0.0.0 --port $ROUTER_PORT --router-backend $BACKEND \
  --discovery-backend etcd --etcd-endpoint $IP:$ETCD_PORT \
  --request-transport http --kv-event-transport zmq \
  --router-policy $POLICY --router-tokenizer-path $TOKENIZER \
  > /tmp/router.log 2>&1\""

for i in $(seq 1 24); do
  code=$($SSH_CMD "$NODE" "docker exec $CTR bash -c \"curl -s -o /dev/null -w '%{http_code}' -m 3 http://$IP:$ROUTER_PORT/health\"" 2>/dev/null | tr -d '\r\n ')
  if [ "$code" = "200" ]; then
    echo "router up on :$ROUTER_PORT (backend=$BACKEND policy=$POLICY)"
    exit 0
  fi
  sleep 5
done

echo "router did not come up; last 30 lines:" >&2
$SSH_CMD "$NODE" "docker exec $CTR bash -c 'tail -30 /tmp/router.log'" >&2
exit 1
