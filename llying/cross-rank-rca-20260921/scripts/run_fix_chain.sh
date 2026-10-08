#!/usr/bin/env bash
# Run C80 reproductions back to back on one node pair, unattended (setsid on the
# Prefill node). Each run: check both nodes are idle and hold the config's image
# (seeding its AITER cache from the base image's), run_reproduction.sh, stop.sh,
# gzip the aiperf server metrics, wait for VRAM release.
# Usage: RCA_PREFILL_NODE=<p> RCA_DECODE_NODE=<d> [RCA_COUNTER_INTERVAL=<s>]
#        run_fix_chain.sh CONFIG:RUN_PREFIX [CONFIG:RUN_PREFIX ...]
#   Run IDs are <RUN_PREFIX>-<P>-<D>-<UTC minute>; progress goes to stdout.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_ID=fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35
NODES=("$RCA_PREFILL_NODE" "$RCA_DECODE_NODE")
pair="${RCA_PREFILL_NODE##*-}-${RCA_DECODE_NODE##*-}"

log() { echo "$(date -u +%FT%TZ) $*"; }
fail() { log "ABORT: $*"; exit 1; }
gpus_holding() {  # GPUs above 2 GiB VRAM across both nodes
    for node in "${NODES[@]}"; do
        ssh -o BatchMode=yes "$node" rocm-smi --showmeminfo vram --csv
    done | awk -F, '/^card/ && $3 > 2*2^30 {n++} END {print n+0}'
}

prepare() {  # prepare <image> <image id>
    local cache="/tmp/aiter-jit-$(id -u)" id="${2#sha256:}" node got others
    for node in "${NODES[@]}"; do
        got="$(ssh -o BatchMode=yes "$node" docker image inspect "$1" --format '{{.Id}}')"
        [[ "$got" == "$2" ]] || fail "$node has $1 as '$got', expected $2"
        ssh -o BatchMode=yes "$node" \
            "[[ -d $cache/$id ]] || cp -a $cache/$BASE_ID $cache/$id" || fail "$node AITER cache"
        others="$(ssh -o BatchMode=yes "$node" docker ps --format '{{.Names}}' |
            grep -vE '^(crusoe-|yihou-expert8-tools-138$)' || true)"
        [[ -z "$others" ]] || fail "$node runs other containers: $others"
    done
    [[ "$(gpus_holding)" == 0 ]] || fail "GPUs are not free"
}

for spec in "$@"; do
    config="$ROOT/config/${spec%%:*}"
    id="${spec#*:}-$pair-$(date -u +%Y%m%dT%H%MZ)"
    eval "$(source "$config" >/dev/null && echo "image=$IMAGE image_id=$EXPECTED_IMAGE_ID topology=$RCA_TOPOLOGY bench=$BENCH_DIR")" ||
        fail "cannot source $config"
    prepare "$image" "$image_id"
    log "start $id (${spec%%:*})"
    CONFIG="$config" RUN_ID="$id" bash "$ROOT/scripts/run_reproduction.sh" \
        >"$ROOT/operations/$id.console.log" 2>&1
    log "$id finished rc=$?"
    bash "$bench/stop.sh" "RCA_PREFILL_NODE=$RCA_PREFILL_NODE" "RCA_DECODE_NODE=$RCA_DECODE_NODE" \
        "CONFIG=$config" "TOPOLOGY=$topology" >>"$ROOT/operations/$id.console.log" 2>&1
    log "$id stopped rc=$?"
    gzip -1 "$ROOT/runs/$id/bench/aiperf_artifacts/server_metrics_export.json" 2>/dev/null
    for _ in $(seq 1 40); do
        [[ "$(gpus_holding)" == 0 ]] && break
        sleep 15
    done
    log "$id GPUs holding VRAM: $(gpus_holding)"
done
log "done"
