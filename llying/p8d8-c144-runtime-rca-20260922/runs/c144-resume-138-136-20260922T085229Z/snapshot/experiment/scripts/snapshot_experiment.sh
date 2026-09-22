#!/usr/bin/env bash
# Snapshot every script/config and live image identity used by one run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/config.c144.p8d8.sh"
TOPOLOGY="$ROOT/config/topology.136-138.tsv"
source "$CONFIG"

OUT="${1:?usage: $0 RUN_DIR}"
SNAPSHOT="$OUT/snapshot"
mkdir -p "$SNAPSHOT/experiment" "$SNAPSHOT/harness" "$SNAPSHOT/source"

cp -a "$ROOT/config" "$ROOT/scripts" "$SNAPSHOT/experiment/"
cp -a \
    "$BENCH_DIR/launch.sh" \
    "$BENCH_DIR/engine.sh" \
    "$BENCH_DIR/agentx_bench.sh" \
    "$BENCH_DIR/stop.sh" \
    "$BENCH_DIR/tools/agentx_env.py" \
    "$SNAPSHOT/harness/"
cp -a \
    "$PACKUP_ROOT/scripts/config.yihou.p8d8.sh" \
    "$PACKUP_ROOT/scripts/bench-harness/config.full.sh" \
    "$PACKUP_ROOT/scripts/topology.yihou.tsv" \
    "$PACKUP_ROOT/scripts/run_full_sweep.yihou.sh" \
    "$SNAPSHOT/source/"

(
    cd "$SNAPSHOT"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$SNAPSHOT/sha256.txt"

git -C /home/liyingli/bench_agentx/baseline/Infera rev-parse HEAD \
    >"$SNAPSHOT/git-head.txt"
git -C /home/liyingli/bench_agentx/baseline/Infera status --short --branch \
    >"$SNAPSHOT/git-status.txt"
date -u --iso-8601=ns >"$SNAPSHOT/captured-at.txt"

set -a
source "$CONFIG"
set +a
for name in \
    IMAGE EXPECTED_IMAGE_ID PREFILL_NODE PREFILL_IP DECODE_NODE DECODE_IP \
    CONTROL_NODE CONTAINER_PREFIX CONC AGENTX_DURATION \
    AGENTX_WARMUP_REQUESTS_PER_LANE PREFILL_TP PREFILL_DP DECODE_TP DECODE_DP \
    PREFILL_HICACHE DECODE_HICACHE PREFILL_HICACHE_RATIO \
    PREFILL_MAX_RUNNING DECODE_MAX_RUNNING PREFILL_GRAPH_MAX_BS \
    DECODE_GRAPH_MAX_BS PD_DP_RANK_AFFINITY RDMA_DEVICE MC_TE_FILTERS \
    MC_GID_INDEX MC_ENABLE_DEST_DEVICE_AFFINITY DECODE_SIMULATE_ACC_LEN \
    JSON_MODEL_OVERRIDE_ARGS; do
    printf '%s=%q\n' "$name" "${!name-}"
done >"$SNAPSHOT/effective-config.env"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    actual="$(
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
            docker image inspect "$IMAGE" --format '{{.Id}}'
    )"
    printf '%s\n' "$actual" >"$SNAPSHOT/image-id-$node.txt"
    if [[ "$actual" != "$EXPECTED_IMAGE_ID" ]]; then
        echo "$node image mismatch: $actual expected=$EXPECTED_IMAGE_ID" >&2
        exit 1
    fi
done

echo "snapshot captured at $SNAPSHOT"
