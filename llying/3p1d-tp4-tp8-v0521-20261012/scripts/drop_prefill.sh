#!/usr/bin/env bash
# Take one prefill instance out of the live stack (3P1D -> 2P1D) without touching the others.
# Usage: drop_prefill.sh ROUND=<r> INSTANCE=<prefill-N> TAG=<suffix>
#   1. append EXCLUDE_INSTANCES="<INSTANCE>" to config/round.<ROUND>.sh: every script that
#      sources the config (topology.py, agentx_env.py validation, tune_round.sh) then uses the
#      reduced topology, and AgentX's per-GPU denominator follows the worker count;
#   2. stop the router and the instance (renamed <name>-<TAG>, kept for logs), move its server
#      log to <instance>.before-<TAG>.log;
#   3. restart the router, wait for the reduced worker count, smoke one request.
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${ROUND:?}" "${INSTANCE:?}" "${TAG:?}"
[[ "$INSTANCE" =~ ^prefill-[0-9]+$ ]] || { echo "INSTANCE must be prefill-N" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
CONFIG="$R/config/config.3p1d.sh"
set -a; source "$CONFIG"; set +a
LOGS="$RUNS/launch/server-logs"
read -r -a ssh_args <<< "$SSH_OPTS"
q() { local out="" i; for i in "$@"; do printf -v i '%q' "$i"; out+="${out:+ }$i"; done; echo "$out"; }
on() { local n="$1"; shift; ssh "${ssh_args[@]}" "$n" "$(q "$@")" </dev/null; }
say() { echo "$(date -u +%FT%TZ) drop_prefill: $*"; }
node="$(EXCLUDE_INSTANCES= python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' -v i="$INSTANCE" '$2==i {print $4}')"
[[ -n "$node" ]] || { echo "$INSTANCE not in $TOPOLOGY" >&2; exit 2; }
container="$CONTAINER_PREFIX-$INSTANCE"; router="$CONTAINER_PREFIX-router"
[[ "$container" == *llying* && "$router" == *llying* ]] || exit 2

round_file="$R/config/round.$ROUND.sh"
[[ -e "$round_file" ]] && cp -p "$round_file" "$round_file.before-$TAG"
printf '\n# %s (%s): %s stopped for a reduced-topology test.\nEXCLUDE_INSTANCES="%s"\n' \
    "$(date -u +%FT%TZ)" "$TAG" "$INSTANCE" "$INSTANCE" >> "$round_file"
set -a; source "$CONFIG"; set +a
python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | awk -F'\t' -v i="$INSTANCE" '$2==i {exit 1}' ||
    { say "topology still lists $INSTANCE"; exit 1; }
say "EXCLUDE_INSTANCES=$EXCLUDE_INSTANCES recorded in $(basename "$round_file")"

for c in "$router:$CONTROL_NODE" "$container:$node"; do
    name="${c%%:*}"; host="${c#*:}"
    if on "$host" docker inspect "$name" >/dev/null 2>&1; then
        on "$host" docker stop --time 120 "$name" >/dev/null
        on "$host" docker rename "$name" "$name-$TAG"
        say "retired $name on $host as $name-$TAG"
    fi
done
for i in $(seq 1 60); do
    n_follow="$(on "$node" bash -c "ps -eo args | grep -c '^[d]ocker logs .* $container\$' || true")"
    [[ "$n_follow" == 0 ]] && break
    sleep 5
done
[[ -e "$LOGS/$INSTANCE.log" ]] && mv "$LOGS/$INSTANCE.log" "$LOGS/$INSTANCE.before-$TAG.log"

bash "$HERE/restart_router.sh" "ROUND=$ROUND" AFFINITY=false "TAG=unused-$TAG" > "$RUNS/launch/router.after-$TAG.log" 2>&1 ||
    { say "router restart failed (see router.after-$TAG.log)"; exit 1; }
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
curl -fsS --max-time 300 "http://$control_ip:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":16,\"temperature\":0}" \
    > "$RUNS/launch/smoke.after-$TAG.json" || { say "smoke request failed"; exit 1; }
say "done: $(tail -1 "$RUNS/launch/router.after-$TAG.log"), smoke ok"
