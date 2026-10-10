#!/usr/bin/env bash
# Unattended B4 run, then stop_after.sh on it. Run under setsid on the Prefill
# node; extra B4_* variables pass through.
#   No STAGE: wait (up to WAIT_MIN minutes, default 120) until every GPU the run
#     uses (P: 0-7, D: B4_DECODE_GPUS or 0..B4_DECODE_TP-1) holds < 2% VRAM,
#     then run_b4.sh all under a new B4_RUN_ID.
#   STAGE...: run those stages in order on the existing run B4_RUN_ID.
# Usage: B4_PREFILL_NODE=<p> B4_DECODE_NODE=<d> [B4_RUN_ID=<id>] run_unattended.sh [STAGE...]
set -uo pipefail
HERE="${B4_SCRIPTS:-$(cd "$(dirname "$0")" && pwd)}"
# bash reads a script as it runs; a git checkout replacing this file on NFS
# mid-run ("Stale file handle") would kill the cleanup step, so run a copy.
if [[ -z "${B4_SCRIPTS:-}" ]]; then
    copy="$(mktemp /tmp/run_unattended.XXXXXX.sh)"
    cp "$0" "$copy"
    B4_SCRIPTS="$HERE" exec bash "$copy" "$@"
fi
nodes=("${B4_PREFILL_NODE:-crsuse2-m2m-136}" "${B4_DECODE_NODE:-crsuse2-m2m-138}")
gpus=(0,1,2,3,4,5,6,7 "${B4_DECODE_GPUS:-$(seq -s, 0 $((${B4_DECODE_TP:-8} - 1)))}")  # per node
busy() {
    for i in 0 1; do
        ssh -o BatchMode=yes "${nodes[i]}" rocm-smi --showmeminfo vram --csv |
            awk -F, -v use=",${gpus[i]}," '/^card/ { sub("card", "", $1)
                if (index(use, "," $1 ",") && $3 >= 0.02 * $2) busy++ } END { print busy + 0 }'
    done | awk '{ total += $1 } END { print total + 0 }'
}
if (( $# == 0 )); then
    set -- all
    for _ in $(seq 1 "${WAIT_MIN:-120}"); do
        [[ "$(busy)" == 0 ]] && break
        sleep 60
    done
    [[ "$(busy)" == 0 ]] || { echo "$(date -u +%FT%TZ) GPUs still hold VRAM; not starting"; exit 1; }
    shape=$([[ "${B4_DECODE_TP:-8}" == 8 ]] || echo "-d${B4_DECODE_TP}")${B4_TAG:+-$B4_TAG}
    export B4_RUN_ID="b4-crsuse2-${nodes[0]##*-}-${nodes[1]##*-}${shape}-$(date -u +%Y%m%dT%H%MZ)"
fi
: "${B4_RUN_ID:?set B4_RUN_ID to resume an existing run}"
echo "$(date -u +%FT%TZ) $B4_RUN_ID: stages $*"
(for stage in "$@"; do bash "$HERE/run_b4.sh" "$stage" || exit; done) &
pid=$!
wait "$pid"
rc=$?
echo "$(date -u +%FT%TZ) stages exited rc=$rc"
bash "$HERE/stop_after.sh" "$pid"
