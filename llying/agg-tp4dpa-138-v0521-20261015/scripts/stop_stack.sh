#!/usr/bin/env bash
# Stop one round's aggregated stack (router, workers, etcd; containers stay for their logs) and wait
# until the GPUs it used are free. Only containers named <CONTAINER_PREFIX>-* (llying) are touched.
# Usage: stop_stack.sh ROUND=<r>
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
HERE="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$HERE/../config/config.agg.sh"; set +a
[[ "$CONTAINER_PREFIX" == *llying* ]] || exit 2
read -r -a ssh_args <<< "$SSH_OPTS"
names="$(ssh "${ssh_args[@]}" -n "$CONTROL_NODE" docker ps --format '{{.Names}}' |
    grep -E "^$CONTAINER_PREFIX-(router|mixed-[0-9]+|etcd)\$" || true)"
if [[ -n "$names" ]]; then
    # shellcheck disable=SC2086
    ssh "${ssh_args[@]}" -n "$CONTROL_NODE" docker stop --time 120 $names >/dev/null
fi
for i in $(seq 1 60); do
    busy="$(ssh "${ssh_args[@]}" -n "$CONTROL_NODE" rocm-smi --showmemuse --json | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(" ".join(k for k, v in d.items() if k.startswith("card") and float(v.get("GPU Memory Allocated (VRAM%)", 0)) > 1))')"
    [[ -z "$busy" ]] && { echo "stopped: ${names//$'\n'/ }; GPUs free"; exit 0; }
    sleep 10
done
echo "GPUs still in use after 10 min: $busy" >&2
exit 1
