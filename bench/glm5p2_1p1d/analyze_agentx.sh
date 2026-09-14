#!/usr/bin/env bash
# Build a clean Pareto input tree from PASS points and plot on InferenceX axes.
# Usage: bash analyze_agentx.sh [RESULTS_ROOT]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"

ROOT="${1:-$RUN_ROOT}"
ROOT="$(cd "$ROOT" && pwd)"
[[ -d "$ROOT/agentx" ]] || {
    echo "missing AgentX results directory: $ROOT/agentx" >&2
    exit 66
}
RUN_NAME="$(basename "$ROOT")"
CURVE="$ROOT/analysis/curve"
rm -rf "$CURVE"
mkdir -p "$CURVE"

count=0
for point in "$ROOT"/agentx/c*; do
    [[ -d "$point" && -f "$point/PASS" ]] || continue
    json=("$point"/agentx_conc*.json)
    [[ -s "${json[0]}" ]] || continue
    name="$(basename "$point")"
    mkdir -p "$CURVE/$name"
    cp "${json[0]}" "$CURVE/$name/"
    ((count += 1))
done
(( count > 0 )) || { echo "no PASS AgentX points under $ROOT" >&2; exit 1; }

WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"
REMOTE_CURVE="$REMOTE_WORK_ROOT/$RUN_NAME/analysis/curve"
tar -C "$CURVE" -cf - . \
    | ssh $SSH_OPTS "$PREFILL_NODE" \
        "rm -rf '$REMOTE_CURVE' && mkdir -p '$REMOTE_CURVE' && tar -C '$REMOTE_CURVE' -xf -"
ssh $SSH_OPTS "$PREFILL_NODE" docker run -i --rm \
    --network host \
    -v "$WORKSPACE_ROOT:$WORKSPACE_ROOT" \
    -v "$REMOTE_CURVE:$REMOTE_CURVE" \
    -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
    "$IMAGE" bash -s -- "$REMOTE_CURVE" "$ROCM_LLM_BENCH_DIR/utils/plot.py" <<'PLOT_CLIENT'
set -e
root=$1
plot=$2
trap "chown -R $HOST_UID:$HOST_GID $root" EXIT
python3 "$plot" "$root"
PLOT_CLIENT
ssh $SSH_OPTS "$PREFILL_NODE" "tar -C '$REMOTE_CURVE' -cf - ." \
    | tar -C "$CURVE" -xf -

echo "[analysis] $count points"
echo "[analysis] CSV: $CURVE/results.csv"
echo "[analysis] plot: $CURVE/pareto.png"
