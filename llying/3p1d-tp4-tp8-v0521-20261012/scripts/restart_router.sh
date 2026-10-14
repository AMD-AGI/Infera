#!/usr/bin/env bash
# Restart only the router. Usage:
#   restart_router.sh ROUND=<r> AFFINITY=<true|false> TAG=<suffix> [ROUTER_ENV="INFERA_X=v ..."]
# AFFINITY is the PD DP-rank affinity (same-rank vs cross-rank KV diagnostics). ROUTER_ENV
# passes router experiment switches (INFERA_* only, whitespace-separated KEY=VALUE).
# The running router is stopped and renamed <name>-<TAG>; the new one takes the name.
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${AFFINITY:?}" "${TAG:?}"
[[ "$AFFINITY" == true || "$AFFINITY" == false ]] || { echo "AFFINITY must be true|false" >&2; exit 2; }
router_env=()
for kv in ${ROUTER_ENV:-}; do
    [[ "$kv" =~ ^INFERA_[A-Z0-9_]+=[A-Za-z0-9_.-]+$ ]] || { echo "bad ROUTER_ENV item '$kv'" >&2; exit 2; }
    router_env+=(-e "$kv")
done
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$HERE/../config/config.3p1d.sh"; set +a
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
router="$CONTAINER_PREFIX-router"
[[ "$router" == *llying* ]] || exit 2
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
if on "$CONTROL_NODE" docker inspect "$router" >/dev/null 2>&1; then
    on "$CONTROL_NODE" docker stop --time 30 "$router" >/dev/null
    on "$CONTROL_NODE" docker rename "$router" "$router-$TAG"
fi
key="${CONTROL_NODE##*-}"
on "$CONTROL_NODE" docker run -d --init --name "$router" --network host \
    -e "INFERA_PD_DP_RANK_AFFINITY=$AFFINITY" ${router_env[@]+"${router_env[@]}"} \
    -v "$WS:$WS:ro" -v "$NODE_RW_ROOT:$NODE_RW_ROOT" -v "$MODEL:$MODEL:ro" \
    --tmpfs "$NODE_RW_ROOT/tmp/$key/router:rw,exec,size=2g" \
    -e "TMPDIR=$NODE_RW_ROOT/tmp/$key/router" \
    -e "HF_HOME=$NODE_RW_ROOT/cache/$key/router" \
    -e "XDG_CACHE_HOME=$NODE_RW_ROOT/cache/$key/router" -e PYTHONDONTWRITEBYTECODE=1 \
    "$IMAGE" python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
    --router-backend rust --discovery-backend etcd --etcd-endpoint "$control_ip:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq --router-tokenizer-path "$MODEL" \
    --router-policy kv-aware --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT" \
    --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT" >/dev/null
expected="$(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | wc -l)"
for i in $(seq 1 60); do
    n="$(curl -fsS --max-time 5 "http://$control_ip:$ROUTER_PORT/v1/workers" 2>/dev/null | python3 -c '
import json,sys
p=json.load(sys.stdin); w=p if isinstance(p,list) else (p.get("workers") or p.get("data") or [])
print(len(w))' 2>/dev/null || echo 0)"
    [[ "$n" == "$expected" ]] && { echo "router up (affinity=$AFFINITY, env='${ROUTER_ENV:-}'), $n workers"; exit 0; }
    sleep 3
done
echo "router did not see $expected workers" >&2; exit 1
