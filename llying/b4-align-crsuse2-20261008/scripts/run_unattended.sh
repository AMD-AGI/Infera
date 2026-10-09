#!/usr/bin/env bash
# Unattended B4 run, then stop_after.sh on it. Run under setsid on the Prefill
# node; extra B4_* variables pass through.
#   No STAGE: wait (up to WAIT_MIN minutes, default 120) until every GPU on both
#     nodes holds < 2% VRAM, then run_b4.sh all under a new B4_RUN_ID.
#   STAGE...: run those stages in order on the existing run B4_RUN_ID.
# Usage: B4_PREFILL_NODE=<p> B4_DECODE_NODE=<d> [B4_RUN_ID=<id>] run_unattended.sh [STAGE...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
nodes=("${B4_PREFILL_NODE:-crsuse2-m2m-136}" "${B4_DECODE_NODE:-crsuse2-m2m-138}")
busy() {
    for node in "${nodes[@]}"; do
        ssh -o BatchMode=yes "$node" rocm-smi --showmeminfo vram --csv
    done | awk -F, '/^card/ && $3 >= 0.02 * $2 {n++} END {print n + 0}'
}
if (( $# == 0 )); then
    set -- all
    for _ in $(seq 1 "${WAIT_MIN:-120}"); do
        [[ "$(busy)" == 0 ]] && break
        sleep 60
    done
    [[ "$(busy)" == 0 ]] || { echo "$(date -u +%FT%TZ) GPUs still hold VRAM; not starting"; exit 1; }
    export B4_RUN_ID="b4-crsuse2-${nodes[0]##*-}-${nodes[1]##*-}-$(date -u +%Y%m%dT%H%MZ)"
fi
: "${B4_RUN_ID:?set B4_RUN_ID to resume an existing run}"
echo "$(date -u +%FT%TZ) $B4_RUN_ID: stages $*"
(for stage in "$@"; do bash "$HERE/run_b4.sh" "$stage" || exit; done) &
pid=$!
wait "$pid"
rc=$?
echo "$(date -u +%FT%TZ) stages exited rc=$rc"
bash "$HERE/stop_after.sh" "$pid"
