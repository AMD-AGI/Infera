#!/usr/bin/env bash
# Collect manually-run AgentX points and draw a multi-P/D Pareto curve.
set -euo pipefail
COMPONENT=analyze
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"
source "$DIR/lib/common.sh"

load_config "$@"
RESULT_DIR="${RESULT_DIR:?set RESULT_DIR=<directory containing AgentX points>}"
[[ "$RESULT_DIR" == /* ]] || RESULT_DIR="$DIR/$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"
[[ "$RESULT_DIR" == "$WORKSPACE_ROOT"/* ]] ||
    die "RESULT_DIR must be under the shared workspace: $WORKSPACE_ROOT"

if [[ "$(bool01 "${UPDATE_REFERENCE:-0}")" == 1 ]]; then
    python3 "$DIR/tools/update_inferencex_ref.py"
fi

read -r -a SSH_ARGS <<< "$SSH_OPTS"
plot_image="${PLOT_IMAGE:-$IMAGE}"
log "collecting AgentX results under $RESULT_DIR"
ssh_exec "$CONTROL_NODE" docker run --rm \
    -v "$WORKSPACE_ROOT:$WORKSPACE_ROOT" \
    -w "$DIR" \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    "$plot_image" bash -c '
        set -euo pipefail
        root="$1"
        trap "chown -R $HOST_UID:$HOST_GID \"$root\"" EXIT
        python3 tools/plot_agentx.py "$root"
    ' _ "$RESULT_DIR"

log "CSV: $RESULT_DIR/results.csv"
log "plot: $RESULT_DIR/pareto.png"
