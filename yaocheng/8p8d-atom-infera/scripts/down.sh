#!/usr/bin/env bash
# Save logs and `docker inspect` of every kit container into the current run
# directory, then remove the containers. Usage: scripts/down.sh
source "$(dirname "$0")/common.sh" "$@"

out="$(current_run 2>/dev/null || echo "$TMP_DIR")/logs"
mkdir -p "$out"
for target in "$CONTROL_NODE:router" "$DECODE_NODE:decode" "$PREFILL_NODE:prefill" \
    "$CONTROL_NODE:etcd"; do
    node="${target%%:*}" role="${target#*:}"
    inspect="$(on "$node" docker inspect "$PREFIX-$role" 2>/dev/null)" || continue
    echo "$inspect" >"$out/$role.inspect.json"
    on "$node" docker logs --timestamps "$PREFIX-$role" >"$out/$role.log" 2>&1
    on "$node" docker rm -f "$PREFIX-$role" >/dev/null
    echo "removed $PREFIX-$role on $node"
done
# Removal returns before the GPU driver has released the engines' memory.
sleep 15
