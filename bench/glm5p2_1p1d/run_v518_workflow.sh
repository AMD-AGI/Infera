#!/usr/bin/env bash
# Run the complete reproducible v0.5.18 ionic 8-rail validation.
# Resume with WORKFLOW_START_AT=build|preflight|correctness|agentx|analysis.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESULTS_DIR="${RESULTS_DIR:-$DIR/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$RESULTS_DIR/$RUN_ID}"
export RESULTS_DIR RUN_ID RUN_ROOT
OUT="$RUN_ROOT/workflow"
mkdir -p "$OUT"

# Inventory must precede config.sh because config.sh represents an already
# selected pair. Prefer the caller's pair (or the previous pair on resume), but
# replace either node automatically when it no longer satisfies the gates.
previous_prefill=""
previous_decode=""
if [[ -f "$OUT/selected-nodes.env" ]]; then
    previous_prefill="$(awk -F= '$1 == "PREFILL_NODE" {print $2}' "$OUT/selected-nodes.env")"
    previous_decode="$(awk -F= '$1 == "DECODE_NODE" {print $2}' "$OUT/selected-nodes.env")"
elif [[ -f "$RUN_ROOT/manifest.txt" ]]; then
    previous_prefill="$(
        awk -F= '$1 == "prefill_node" || $1 == "active_prefill_node" {value=$2} END {print value}' \
            "$RUN_ROOT/manifest.txt"
    )"
    previous_decode="$(
        awk -F= '$1 == "decode_node" || $1 == "active_decode_node" {value=$2} END {print value}' \
            "$RUN_ROOT/manifest.txt"
    )"
fi
preferred_prefill="${PREFILL_NODE:-$previous_prefill}"
preferred_decode="${DECODE_NODE:-$previous_decode}"

inventory_args=()
if [[ -n "${INVENTORY_NODES:-}" ]]; then
    read -r -a inventory_args <<<"$INVENTORY_NODES"
fi
selection_id="$(date -u +%Y%m%dT%H%M%S%NZ)_pid$$"
selection_out="$OUT/node-selection-$selection_id"
set +e
PREFERRED_PREFILL_NODE="$preferred_prefill" \
PREFERRED_DECODE_NODE="$preferred_decode" \
INVENTORY_OUT_DIR="$selection_out" \
RUN_ID="$RUN_ID" RUN_ROOT="$RUN_ROOT" \
    bash "$DIR/inventory_nodes.sh" "${inventory_args[@]}" \
    2>&1 | tee "$OUT/node-selection-$selection_id.log"
selection_rc=${PIPESTATUS[0]}
set -e
(( selection_rc == 0 )) || {
    echo "[workflow] node selection failed with rc=$selection_rc" >&2
    exit "$selection_rc"
}

# selected.env is generated from probed hostnames with shell escaping.
source "$selection_out/selected.env"
cp "$selection_out/selected.env" "$OUT/selected-nodes.env"
export PREFILL_NODE DECODE_NODE
nodes_reselected=0
if [[ -z "$preferred_prefill" || -z "$preferred_decode" ||
    "$PREFILL_NODE" != "$preferred_prefill" || "$DECODE_NODE" != "$preferred_decode" ]]; then
    nodes_reselected=1
fi

if (( nodes_reselected == 1 )); then
    # These values may have been exported by a config.sh sourced for the old
    # pair. Force config.sh to derive them from the replacement pair.
    unset PREFILL_IP DECODE_IP BENCH_NODE_PAIR BENCH_LOCK_FILE
    unset GLM52_BENCH_LOCK_FD GLM52_BENCH_LOCK_HELD
fi
source "$DIR/config.sh"
acquire_bench_lock

[[ "$AGENTX_CONCURRENCIES" == "8" && "$AGENTX_DURATION" == "3600" ]] || {
    echo "[workflow] canonical validation requires AGENTX_CONCURRENCIES=8 and AGENTX_DURATION=3600" >&2
    exit 64
}
[[ "$PREFILL_TP_SIZE/$PREFILL_EP_SIZE/$PREFILL_DP_SIZE" == "8/1/1" &&
    "$DECODE_TP_SIZE/$DECODE_EP_SIZE/$DECODE_DP_SIZE" == "8/1/1" ]] || {
    echo "[workflow] canonical validation requires TP8/EP1/DP1 on both legs" >&2
    exit 64
}
[[ "$ENABLE_KV_AWARE" == "1" && "$ENABLE_MTP" == "1" &&
    "$KV_P2P_TRANSFER" == "mooncake" ]] || {
    echo "[workflow] canonical validation requires KV-aware routing, MTP, and Mooncake" >&2
    exit 64
}
[[ "$PREFILL_DPA/$DECODE_DPA" == "0/0" &&
    "$SPEC_STEPS/$SPEC_DRAFT_TOKENS/$SPEC_TOPK" == "5/6/1" &&
    "$SIMULATE_ACC_LEN" == "3.61" ]] || {
    echo "[workflow] canonical validation requires DPA off and the calibrated 5/6/1 MTP profile" >&2
    exit 64
}
[[ "$PREFILL_ENABLE_HICACHE/$DECODE_ENABLE_HICACHE" == "1/0" &&
    "$HICACHE_RATIO" == "1.5" &&
    "$HICACHE_WRITE_POLICY/$HICACHE_IO_BACKEND/$HICACHE_MEM_LAYOUT" == "write_through/kernel/page_first" ]] || {
    echo "[workflow] canonical validation requires the production prefill HiCache profile" >&2
    exit 64
}
canonical_rdma="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
[[ "$RDMA_DEVICE" == "$canonical_rdma" && "$MC_TE_FILTERS" == "$canonical_rdma" &&
    "$MC_GID_INDEX" == "1" && "$MOONCAKE_DISABLE_HIP_DMABUF" == "0" &&
    "$MC_ENABLE_DEST_DEVICE_AFFINITY" == "1" ]] || {
    echo "[workflow] canonical validation requires the exact ionic 8-rail dma-buf configuration" >&2
    exit 64
}

compute_core_source_id() {
    local repo="$DIR/../.."
    {
        tar --sort=name --mtime='UTC 1970-01-01' \
            --owner=0 --group=0 --numeric-owner \
            --exclude='bench/glm5p2_1p1d/results' \
            --exclude='bench/glm5p2_1p1d/VALIDATION*.md' \
            --exclude='bench/glm5p2_1p1d/agentx_point.sh' \
            --exclude='bench/glm5p2_1p1d/run_agentx_sweep.sh' \
            --exclude='bench/glm5p2_1p1d/analyze_agentx.sh' \
            --exclude='*/__pycache__' --exclude='*.pyc' \
            -C "$repo" -cf - \
            infera deploy examples/sglang_1p1d_glm5.2 bench/glm5p2_1p1d
    } | sha256sum | awk '{print $1}'
}

compute_agentx_source_id() {
    {
        echo "core_source_id=$CORE_SOURCE_ID"
        sha256sum \
            "$DIR/agentx_point.sh" \
            "$DIR/run_agentx_sweep.sh" \
            "$DIR/analyze_agentx.sh"
        git -C "$ROCM_LLM_BENCH_DIR" rev-parse HEAD 2>/dev/null || true
        git -C "$ROCM_LLM_BENCH_DIR" diff --binary --no-ext-diff 2>/dev/null || true
        git -C "$INFERENCEX_DIR" rev-parse HEAD 2>/dev/null || true
        git -C "$INFERENCEX_DIR" diff --binary --no-ext-diff 2>/dev/null || true
    } | sha256sum | awk '{print $1}'
}

compute_core_config_id() {
    {
        printf '%s=%s\n' \
            PREFILL_TP_SIZE "$PREFILL_TP_SIZE" DECODE_TP_SIZE "$DECODE_TP_SIZE" \
            PREFILL_EP_SIZE "$PREFILL_EP_SIZE" DECODE_EP_SIZE "$DECODE_EP_SIZE" \
            PREFILL_DP_SIZE "$PREFILL_DP_SIZE" DECODE_DP_SIZE "$DECODE_DP_SIZE" \
            PREFILL_DPA "$PREFILL_DPA" DECODE_DPA "$DECODE_DPA" \
            CONTEXT_LENGTH "$CONTEXT_LENGTH" CHUNKED_PREFILL_SIZE "$CHUNKED_PREFILL_SIZE" \
            PREFILL_MEM_FRACTION "$PREFILL_MEM_FRACTION" \
            DECODE_MEM_FRACTION "$DECODE_MEM_FRACTION" \
            ENABLE_MTP "$ENABLE_MTP" SPEC_STEPS "$SPEC_STEPS" \
            SPEC_DRAFT_TOKENS "$SPEC_DRAFT_TOKENS" SPEC_TOPK "$SPEC_TOPK" \
            ENABLE_KV_AWARE "$ENABLE_KV_AWARE" \
            KV_PREFILL_OVERLAP_WEIGHT "$KV_PREFILL_OVERLAP_WEIGHT" \
            KV_DECODE_OVERLAP_WEIGHT "$KV_DECODE_OVERLAP_WEIGHT" \
            ENABLE_HICACHE "$ENABLE_HICACHE" \
            PREFILL_ENABLE_HICACHE "$PREFILL_ENABLE_HICACHE" \
            DECODE_ENABLE_HICACHE "$DECODE_ENABLE_HICACHE" \
            HICACHE_RATIO "$HICACHE_RATIO" HICACHE_WRITE_POLICY "$HICACHE_WRITE_POLICY" \
            HICACHE_IO_BACKEND "$HICACHE_IO_BACKEND" \
            HICACHE_MEM_LAYOUT "$HICACHE_MEM_LAYOUT" \
            KV_P2P_TRANSFER "$KV_P2P_TRANSFER" RDMA_DEVICE "$RDMA_DEVICE" \
            MC_TE_FILTERS "$MC_TE_FILTERS" MC_GID_INDEX "$MC_GID_INDEX" \
            MOONCAKE_DISABLE_HIP_DMABUF "$MOONCAKE_DISABLE_HIP_DMABUF" \
            MC_ENABLE_DEST_DEVICE_AFFINITY "$MC_ENABLE_DEST_DEVICE_AFFINITY" \
            READY_TIMEOUT "$READY_TIMEOUT" LEG_HEALTH_TRIES "$LEG_HEALTH_TRIES" \
            GPU_IDLE_TIMEOUT "$GPU_IDLE_TIMEOUT" \
            MODEL "$MODEL" \
            SERVED_MODEL "$SERVED_MODEL"
    } | sha256sum | awk '{print $1}'
}

compute_agentx_config_id() {
    {
        echo "core_config_id=$CORE_CONFIG_ID"
        printf '%s=%s\n' \
            SIMULATE_ACC_LEN "$SIMULATE_ACC_LEN" \
            AGENTX_CONCURRENCIES "$AGENTX_CONCURRENCIES" \
            AGENTX_DURATION "$AGENTX_DURATION" \
            AGENTX_FAILED_REQUEST_THRESHOLD "$AGENTX_FAILED_REQUEST_THRESHOLD"
    } | sha256sum | awk '{print $1}'
}

CORE_SOURCE_ID="$(compute_core_source_id)"
AGENTX_SOURCE_ID="$(compute_agentx_source_id)"
CORE_CONFIG_ID="$(compute_core_config_id)"
AGENTX_CONFIG_ID="$(compute_agentx_config_id)"
# The sweep consumes these exported names in its point-local signature.
SOURCE_ID="$AGENTX_SOURCE_ID"
CONFIG_ID="$AGENTX_CONFIG_ID"
export CORE_SOURCE_ID AGENTX_SOURCE_ID CORE_CONFIG_ID AGENTX_CONFIG_ID
export SOURCE_ID CONFIG_ID

stage_source_id() {
    [[ "$1" == "agentx" ]] && echo "$AGENTX_SOURCE_ID" || echo "$CORE_SOURCE_ID"
}

stage_config_id() {
    [[ "$1" == "agentx" ]] && echo "$AGENTX_CONFIG_ID" || echo "$CORE_CONFIG_ID"
}

if [[ ! -f "$RUN_ROOT/manifest.txt" ]]; then
    {
        echo "run_id=$RUN_ID"
        echo "run_root=$RUN_ROOT"
        echo "prefill_node=$PREFILL_NODE"
        echo "decode_node=$DECODE_NODE"
        echo "prefill_ip=$PREFILL_IP"
        echo "decode_ip=$DECODE_IP"
        echo "image=$IMAGE"
        echo "rdma_device=$RDMA_DEVICE"
        echo "mc_gid_index=$MC_GID_INDEX"
        echo "mc_te_filters=$MC_TE_FILTERS"
        echo "mooncake_disable_hip_dmabuf=$MOONCAKE_DISABLE_HIP_DMABUF"
        echo "mc_enable_dest_device_affinity=$MC_ENABLE_DEST_DEVICE_AFFINITY"
        echo "host_rdma_lib=$HOST_RDMA_LIB"
        echo "host_rdma_mount=$HOST_RDMA_MOUNT"
        echo "ready_timeout=$READY_TIMEOUT"
        echo "leg_health_tries=$LEG_HEALTH_TRIES"
        echo "gpu_idle_timeout=$GPU_IDLE_TIMEOUT"
        echo "container_stop_timeout=$CONTAINER_STOP_TIMEOUT"
        echo "preflight_mooncake_opcode=$INFERA_PREFLIGHT_MOONCAKE_OPCODE"
        echo "agentx_concurrencies=$AGENTX_CONCURRENCIES"
        echo "agentx_duration=$AGENTX_DURATION"
        echo "core_source_id=$CORE_SOURCE_ID"
        echo "agentx_source_id=$AGENTX_SOURCE_ID"
        echo "core_config_id=$CORE_CONFIG_ID"
        echo "agentx_config_id=$AGENTX_CONFIG_ID"
        echo "infera_head=$(git -C "$DIR/../.." rev-parse HEAD 2>/dev/null || true)"
        echo "rocm_llm_bench_head=$(git -C "$ROCM_LLM_BENCH_DIR" rev-parse HEAD 2>/dev/null || true)"
        echo "inferencex_head=$(git -C "$INFERENCEX_DIR" rev-parse HEAD 2>/dev/null || true)"
        echo "started_at=$(date -Is)"
    } >"$RUN_ROOT/manifest.txt"
else
    echo "resumed_at=$(date -Is),stage=${WORKFLOW_START_AT:-cleanup}" \
        >>"$RUN_ROOT/manifest.txt"
fi
{
    echo "node_selection_at=$(date -Is)"
    echo "preferred_prefill_node=$preferred_prefill"
    echo "preferred_decode_node=$preferred_decode"
    echo "active_prefill_node=$PREFILL_NODE"
    echo "active_decode_node=$DECODE_NODE"
    echo "nodes_reselected=$nodes_reselected"
    echo "node_selection_artifacts=$selection_out"
} >>"$RUN_ROOT/manifest.txt"
snapshot="$(date -u +%Y%m%dT%H%M%SZ)"
{
    echo "prefill_node=$PREFILL_NODE"
    echo "decode_node=$DECODE_NODE"
    echo "rdma_device=$RDMA_DEVICE"
    echo "mc_gid_index=$MC_GID_INDEX"
    echo "mc_te_filters=$MC_TE_FILTERS"
    echo "mooncake_disable_hip_dmabuf=$MOONCAKE_DISABLE_HIP_DMABUF"
    echo "mc_enable_dest_device_affinity=$MC_ENABLE_DEST_DEVICE_AFFINITY"
    echo "host_rdma_lib=$HOST_RDMA_LIB"
    echo "host_rdma_mount=$HOST_RDMA_MOUNT"
    echo "ready_timeout=$READY_TIMEOUT"
    echo "leg_health_tries=$LEG_HEALTH_TRIES"
    echo "gpu_idle_timeout=$GPU_IDLE_TIMEOUT"
    echo "container_stop_timeout=$CONTAINER_STOP_TIMEOUT"
    echo "preflight_mooncake_opcode=$INFERA_PREFLIGHT_MOONCAKE_OPCODE"
    echo "agentx_concurrencies=$AGENTX_CONCURRENCIES"
    echo "agentx_duration=$AGENTX_DURATION"
    echo "core_source_id=$CORE_SOURCE_ID"
    echo "agentx_source_id=$AGENTX_SOURCE_ID"
    echo "core_config_id=$CORE_CONFIG_ID"
    echo "agentx_config_id=$AGENTX_CONFIG_ID"
} >"$OUT/config_$snapshot.txt"

capture_source_snapshot() {
    local stamp="$1" repo="$DIR/../.."
    git -C "$repo" status --short >"$OUT/git-status_$stamp.txt" 2>&1 || true
    git -C "$repo" diff >"$OUT/source_$stamp.diff" 2>&1 || true
    git -C "$ROCM_LLM_BENCH_DIR" status --short \
        >"$OUT/rocm-llm-bench-status_$stamp.txt" 2>&1 || true
    git -C "$ROCM_LLM_BENCH_DIR" diff \
        >"$OUT/rocm-llm-bench-source_$stamp.diff" 2>&1 || true
    git -C "$INFERENCEX_DIR" status --short \
        >"$OUT/inferencex-status_$stamp.txt" 2>&1 || true
    git -C "$INFERENCEX_DIR" diff \
        >"$OUT/inferencex-source_$stamp.diff" 2>&1 || true
    # git diff omits untracked files. Archive the complete small Infera source
    # tree (including this untracked bench kit and new patch files), excluding
    # only VCS metadata, bytecode, and multi-GiB run artifacts.
    tar \
        --exclude='./.git' \
        --exclude='./bench/glm5p2_1p1d/results' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        -C "$repo" -czf "$OUT/infera-source_$stamp.tgz" .
    sha256sum "$OUT/infera-source_$stamp.tgz" \
        >"$OUT/infera-source_$stamp.tgz.sha256"
}

capture_source_snapshot "$snapshot"

run_stage() {
    local name="$1"
    shift
    echo "[workflow] starting $name"
    local rc
    set +e
    "$@" 2>&1 | tee "$OUT/$name.log"
    rc=${PIPESTATUS[0]}
    set -e
    printf '%s\t%s\trc=%d\n' "$(date -Is)" "$name" "$rc" >>"$OUT/stages.tsv"
    (( rc == 0 )) || {
        echo "[workflow] $name failed with rc=$rc" >&2
        return "$rc"
    }
}

GATES="$OUT/gates"
mkdir -p "$GATES"

image_id_on() {
    ssh $SSH_OPTS "$1" docker image inspect -f '{{.Id}}' "$IMAGE"
}

write_gate() {
    local stage="$1" prefill_id decode_id source_id config_id tmp
    prefill_id="$(image_id_on "$PREFILL_NODE")"
    decode_id="$(image_id_on "$DECODE_NODE")"
    [[ "$prefill_id" == "$decode_id" ]] || {
        echo "[workflow] cannot certify $stage: image IDs differ" >&2
        return 1
    }
    source_id="$(stage_source_id "$stage")"
    config_id="$(stage_config_id "$stage")"
    tmp="$GATES/$stage.env.tmp"
    {
        printf 'stage=%q\n' "$stage"
        printf 'prefill_node=%q\n' "$PREFILL_NODE"
        printf 'decode_node=%q\n' "$DECODE_NODE"
        printf 'image=%q\n' "$IMAGE"
        printf 'image_id=%q\n' "$prefill_id"
        printf 'source_id=%q\n' "$source_id"
        printf 'config_id=%q\n' "$config_id"
        printf 'passed_at=%q\n' "$(date -Is)"
    } >"$tmp"
    mv "$tmp" "$GATES/$stage.env"
}

gate_matches() {
    local stage="$1" file
    local gate_prefill gate_decode gate_image gate_id gate_source gate_config
    local expected_source expected_config prefill_id decode_id
    file="$GATES/$stage.env"
    [[ -f "$file" ]] || return 1
    gate_prefill="$(awk -F= '$1 == "prefill_node" {print $2}' "$file")"
    gate_decode="$(awk -F= '$1 == "decode_node" {print $2}' "$file")"
    gate_image="$(awk -F= '$1 == "image" {print $2}' "$file")"
    gate_id="$(awk -F= '$1 == "image_id" {print $2}' "$file")"
    gate_source="$(awk -F= '$1 == "source_id" {print $2}' "$file")"
    gate_config="$(awk -F= '$1 == "config_id" {print $2}' "$file")"
    expected_source="$(stage_source_id "$stage")"
    expected_config="$(stage_config_id "$stage")"
    [[ "$gate_prefill" == "$PREFILL_NODE" &&
        "$gate_decode" == "$DECODE_NODE" &&
        "$gate_image" == "$IMAGE" &&
        "$gate_source" == "$expected_source" &&
        "$gate_config" == "$expected_config" ]] || return 1
    prefill_id="$(image_id_on "$PREFILL_NODE" 2>/dev/null)" || return 1
    decode_id="$(image_id_on "$DECODE_NODE" 2>/dev/null)" || return 1
    [[ "$prefill_id" == "$gate_id" && "$decode_id" == "$gate_id" ]]
}

stages=(cleanup inventory build preflight correctness agentx analysis)
requested_start="${WORKFLOW_START_AT:-cleanup}"
start="$requested_start"
if (( nodes_reselected == 1 )); then
    case "$requested_start" in
        preflight|correctness|agentx)
            start=build
            echo "[workflow] node pair changed; restarting at build to validate and distribute the exact image"
            ;;
    esac
fi
# A selected-nodes file alone is not proof that this exact pair completed the
# prerequisite stages with the image currently tagged on both nodes. Resumes
# may start late only when pair+digest gate files prove the whole chain.
if [[ "$start" == "$requested_start" ]]; then
    case "$requested_start" in
        preflight)
            gate_matches build || start=build
            ;;
        correctness)
            gate_matches build || start=build
            ;;
        agentx)
            gate_matches build &&
                gate_matches correctness || start=build
            ;;
        analysis)
            gate_matches build &&
                gate_matches correctness &&
                gate_matches agentx || start=build
            ;;
    esac
    if [[ "$start" != "$requested_start" ]]; then
        echo "[workflow] prerequisite pair/image gates missing or stale; restarting at build"
    fi
fi
start_seen=0
for stage in "${stages[@]}"; do
    if [[ "$stage" == "$start" ]]; then
        start_seen=1
    fi
    (( start_seen == 1 )) || continue
    case "$stage" in
        cleanup)
            run_stage cleanup bash "$DIR/stop.sh"
            ;;
        inventory)
            run_stage inventory cat "$selection_out/summary.tsv"
            ;;
        build)
            run_stage build bash "$DIR/build_v518_image.sh" all
            rm -f "$GATES/correctness.env" "$GATES/agentx.env"
            write_gate build
            ;;
        preflight)
            if run_stage preflight bash "$DIR/preflight.sh" all; then
                preflight_rc=0
            else
                preflight_rc=$?
                echo "[workflow] preflight diagnostic failed with rc=$preflight_rc; continuing to correctness" >&2
            fi
            echo "preflight_rc=$preflight_rc,at=$(date -Is)" >>"$RUN_ROOT/manifest.txt"
            ;;
        correctness)
            run_stage correctness bash "$DIR/run_correctness.sh"
            rm -f "$GATES/agentx.env"
            write_gate correctness
            ;;
        agentx)
            run_stage agentx bash "$DIR/run_agentx_sweep.sh"
            write_gate agentx
            ;;
        analysis)
            run_stage analysis bash "$DIR/analyze_agentx.sh" "$RUN_ROOT"
            ;;
    esac
done

(( start_seen == 1 )) || {
    echo "invalid WORKFLOW_START_AT=$start" >&2
    exit 64
}
snapshot="$(date -u +%Y%m%dT%H%M%SZ)"
capture_source_snapshot "$snapshot"
echo "finished_at=$(date -Is)" >>"$RUN_ROOT/manifest.txt"
echo "[workflow] complete: $RUN_ROOT"
