#!/usr/bin/env bash
# (Re)start the aggregated stack's router on CONTROL_NODE. A running router is stopped and
# renamed <router>-<TAG> so its log survives. Waits until every topology worker is listed and
# checks the session-affinity mode in the startup log.
# Usage: router_agg.sh ROUND=<r> [ROUTER_ENV="INFERA_X=v ..."] [TAG=<suffix>]
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$HERE/../config/config.agg.sh"; set +a
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router="$CONTAINER_PREFIX-router"
[[ "$router" == *llying* ]] || exit 2

if on "$CONTROL_NODE" docker inspect "$router" >/dev/null 2>&1; then
    on "$CONTROL_NODE" docker stop --time 30 "$router" >/dev/null
    on "$CONTROL_NODE" docker rename "$router" "$router-${TAG:-$(date -u +%H%M%S)}"
fi
router_env=()
for item in $ROUTER_ENV; do
    [[ "$item" =~ ^INFERA_[A-Z0-9_]+=[A-Za-z0-9_.:-]*$ ]] || { echo "bad ROUTER_ENV item: $item" >&2; exit 2; }
    router_env+=(-e "$item")
done
key="${CONTROL_NODE##*-}"
on "$CONTROL_NODE" docker run -d --init --name "$router" --network host ${router_env[@]+"${router_env[@]}"} \
    -v "$WS:$WS:ro" -v "$NODE_RW_ROOT:$NODE_RW_ROOT" -v "$MODEL:$MODEL:ro" \
    --tmpfs "$NODE_RW_ROOT/tmp/$key/router:rw,exec,size=2g" \
    -e "TMPDIR=$NODE_RW_ROOT/tmp/$key/router" \
    -e "HF_HOME=$NODE_RW_ROOT/cache/$key/router" \
    -e "XDG_CACHE_HOME=$NODE_RW_ROOT/cache/$key/router" -e PYTHONDONTWRITEBYTECODE=1 \
    "$IMAGE" python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" \
    --router-backend rust --discovery-backend etcd --etcd-endpoint "$control_ip:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq --router-tokenizer-path "$MODEL" \
    --router-policy kv-aware --kv-overlap-weight "$KV_OVERLAP_WEIGHT" >/dev/null
expected="$(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | wc -l)"
for i in $(seq 1 60); do
    n="$(curl -fsS --max-time 5 "http://$control_ip:$ROUTER_PORT/v1/workers" 2>/dev/null | python3 -c '
import json, sys
p = json.load(sys.stdin); w = p if isinstance(p, list) else (p.get("workers") or p.get("data") or [])
print(len(w))' 2>/dev/null || echo 0)"
    [[ "$n" == "$expected" ]] && break
    sleep 5
done
[[ "$n" == "$expected" ]] || { echo "router lists $n workers, expected $expected" >&2; exit 1; }
want=Off
[[ " $ROUTER_ENV " == *" INFERA_SESSION_AFFINITY=both "* ]] && want=Both
[[ " $ROUTER_ENV " == *" INFERA_SESSION_AFFINITY=prefill "* ]] && want=Prefill
on "$CONTROL_NODE" docker logs "$router" 2>&1 | sed -E 's/\x1b\[[0-9;]*m//g' |
    grep -q "session affinity configuration mode=$want " ||
    { echo "router session affinity is not $want" >&2; exit 1; }
echo "router $router: $n worker(s), affinity=$want, kv-overlap-weight=$KV_OVERLAP_WEIGHT"
