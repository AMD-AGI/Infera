#!/usr/bin/env bash
# Residual ACK timeout runs on 138->136, run on 138 under setsid:
#   A: v3 as shipped (MC_IB_TIMEOUT=18); B: v3 with MC_IB_TIMEOUT=24 (69 s).
# Each is the C80/1200 s contract with 1 s HCA counter samples.
# Log: operations/ack-timeout-138-136-chain.log
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG=infera-sglang:v0519-yihou-0917-nextnfix-hicache-mcdestpin-ibto18
ID=sha256:88286215b79d43306dce88dcdf299a13d939c6b708691dc192ca7c7c33c518a5
BASE_ID=fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35
NODES=(crsuse2-m2m-138 crsuse2-m2m-136)
export RCA_PREFILL_NODE=crsuse2-m2m-138 RCA_COUNTER_INTERVAL=1

log() { echo "$(date -u +%FT%TZ) $*"; }
fail() { log "ABORT: $*"; exit 1; }

until grep -q '^rc=' "$ROOT/operations/image-transfer-136-to-138-v3.log"; do sleep 30; done
grep -q '^rc=0' "$ROOT/operations/image-transfer-136-to-138-v3.log" || fail "image transfer failed"
for node in "${NODES[@]}"; do
    got="$(ssh -o BatchMode=yes "$node" docker image inspect "$TAG" --format '{{.Id}}')"
    [[ "$got" == "$ID" ]] || fail "$node has $got, expected $ID"
done
log "v3 image $ID on both nodes"

cache="/tmp/aiter-jit-$(id -u)"
[[ -d "$cache/${ID#sha256:}" ]] || cp -a "$cache/$BASE_ID" "$cache/${ID#sha256:}" ||
    fail "AITER cache copy"
log "AITER cache ready for ${ID:7:12} on crsuse2-m2m-138"

idle() {
    for node in "${NODES[@]}"; do
        others="$(ssh -o BatchMode=yes "$node" docker ps --format '{{.Names}}' |
            grep -vE '^(crusoe-|yihou-expert8-tools-138$)' || true)"
        [[ -z "$others" ]] || fail "$node runs other containers: $others"
        vram="$(ssh -o BatchMode=yes "$node" rocm-smi --showmeminfo vram --csv |
            awk -F, '/^card/ && $3 > 2*2^30 {n++} END {print n+0}')"
        [[ "$vram" == 0 ]] || fail "$node has $vram GPUs holding VRAM"
    done
}

run() {  # run <config> <run-id prefix>
    local config="$1" id="$2-138-136-$(date -u +%Y%m%dT%H%MZ)"
    idle
    log "start $id ($(basename "$config"))"
    CONFIG="$config" RUN_ID="$id" bash "$ROOT/scripts/run_reproduction.sh" \
        >"$ROOT/operations/$id.console.log" 2>&1
    log "$id finished rc=$?"
    (source "$config" && bash "$BENCH_DIR/stop.sh" "RCA_PREFILL_NODE=$RCA_PREFILL_NODE" \
        "CONFIG=$config" "TOPOLOGY=$RCA_TOPOLOGY") >>"$ROOT/operations/$id.console.log" 2>&1
    log "$id stopped rc=$?"
    gzip -1 "$ROOT/runs/$id/bench/aiperf_artifacts/server_metrics_export.json" 2>/dev/null
    for _ in $(seq 1 40); do
        vram="$(for node in "${NODES[@]}"; do ssh -o BatchMode=yes "$node" rocm-smi --showmeminfo vram --csv; done |
            awk -F, '/^card/ && $3 > 2*2^30 {n++} END {print n+0}')"
        [[ "$vram" == 0 ]] && break
        sleep 15
    done
    log "$id GPUs released (holding: $vram)"
}

run "$ROOT/config/config.rca-fix.p8d8.sh" fix-04
run "$ROOT/config/config.rca-fix-ibto24.p8d8.sh" fix-05
log "done"
