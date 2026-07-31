#!/usr/bin/env bash
# Start the sglang_router mini-LB in the PREFILL container, in front of the two PD legs.
# Written to a file inside the container (not `bash -lc`) because the detached login-shell
# form does not persist — exp07 gotcha.
set -euo pipefail
PREFILL_HOST="${PREFILL_HOST:-chi2867}"; PREFILL_IP="${PREFILL_IP:-10.2.122.44}"
DECODE_IP="${DECODE_IP:-10.2.122.10}"
ROUTER_PORT="${ROUTER_PORT:-8002}"
CTR=pd_uni
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 \"$2\""; }

J "$PREFILL_HOST" "docker exec $CTR bash -c 'cat > /run_router.sh <<EOF
#!/bin/bash
pkill -9 -f sglang_router 2>/dev/null; sleep 2
exec python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://$PREFILL_IP:30000 8998 --decode http://$DECODE_IP:30001 \
  --host 0.0.0.0 --port $ROUTER_PORT > /tmp/router.log 2>&1
EOF
chmod +x /run_router.sh'"
J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router.sh"
sleep 16
J "$PREFILL_HOST" "docker exec $CTR bash -c 'curl -s -m 10 http://$PREFILL_IP:$ROUTER_PORT/v1/models | head -c 300; echo; tail -5 /tmp/router.log'"
