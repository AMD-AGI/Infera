#!/usr/bin/env bash
# Sourced by every script: KEY=VALUE overrides, config, node helpers.
set -euo pipefail
for assignment in "$@"; do export "$assignment"; done
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

# on NODE CMD...: run CMD on NODE; words are re-quoted so JSON survives SSH.
on() {
    local node="$1"; shift
    if [[ "$1" == docker && " $SUDO_DOCKER_NODES " == *" $node "* ]]; then set -- sudo "$@"; fi
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

# wait_vram_free NODE: poll until every GPU on NODE uses less than 4 GiB of VRAM.
# The driver can hold a removed engine's VRAM for minutes while reclaiming it.
wait_vram_free() {
    local deadline=$((SECONDS + 900)) max
    while max="$(on "$1" sh -c 'cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null' |
                 sort -n | tail -1)"; (( max >= 4 << 30 )); do
        (( SECONDS < deadline )) || { echo "VRAM on $1 not released: $((max >> 30)) GiB" >&2; return 1; }
        sleep 10
    done
}

current_run() { cat "$TMP_DIR/current_run"; }
