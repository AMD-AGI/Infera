#!/usr/bin/env bash
# Sourced by every script: KEY=VALUE overrides, config, node helpers.
set -euo pipefail
for assignment in "$@"; do export "$assignment"; done
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

# on NODE CMD...: run CMD on NODE; words are re-quoted so JSON survives SSH.
on() {
    local node="$1"; shift
    if [[ "$node" == "$(hostname -s)" ]]; then "$@"
    else ssh $SSH_OPTS "$node" "$(printf '%q ' "$@")"; fi
}

# wait_ready NODE CONTAINER URL: poll URL until it answers; stop if the container exits.
wait_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT))
    until curl -sf -m 5 -o /dev/null "$3"; do
        [[ "$(on "$1" docker inspect -f '{{.State.Running}}' "$2")" == true ]] ||
            { echo "$2 exited on $1, see: docker logs $2" >&2; return 1; }
        (( SECONDS < deadline )) || { echo "timeout waiting for $3" >&2; return 1; }
        sleep 15
    done
}

current_run() { cat "$TMP_DIR/current_run"; }
