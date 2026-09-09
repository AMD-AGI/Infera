#!/usr/bin/env bash
# Run the first v0.5.18 TP8/EP1 curve on one fixed prefill/decode pair.
# Usage: bash run_agentx_sweep.sh [CONC ...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

export RUN_ID
ROOT="$RUN_ROOT"
mkdir -p "$ROOT/agentx"
[[ -n "$SIMULATE_ACC_LEN" ]] || {
    echo "[sweep] SIMULATE_ACC_LEN must be non-empty for the performance sweep" >&2
    exit 64
}

if (( $# > 0 )); then
    concurrencies=("$@")
else
    read -r -a concurrencies <<<"$AGENTX_CONCURRENCIES"
fi
for concurrency in "${concurrencies[@]}"; do
    [[ "$concurrency" =~ ^[0-9]+$ ]] || {
        echo "invalid concurrency: $concurrency" >&2
        exit 64
    }
done

{
    echo "run_id=$RUN_ID"
    echo "prefill_node=$PREFILL_NODE"
    echo "decode_node=$DECODE_NODE"
    echo "image=$IMAGE"
    echo "topology=prefill_tp${PREFILL_TP_SIZE}_ep${PREFILL_EP_SIZE}_dp${PREFILL_DP_SIZE}+decode_tp${DECODE_TP_SIZE}_ep${DECODE_EP_SIZE}_dp${DECODE_DP_SIZE}"
    echo "concurrencies=${concurrencies[*]}"
    echo "duration_seconds=$AGENTX_DURATION"
    echo "simulate_acc_len=$SIMULATE_ACC_LEN"
    echo "started_at=$(date -Is)"
} >"$ROOT/agentx/run_manifest.txt"

[[ "$(ssh $SSH_OPTS "$PREFILL_NODE" hostname)" == "$PREFILL_NODE" ]]
[[ "$(ssh $SSH_OPTS "$DECODE_NODE" hostname)" == "$DECODE_NODE" ]]
prefill_image_id="$(ssh $SSH_OPTS "$PREFILL_NODE" docker image inspect -f '{{.Id}}' "$IMAGE")"
decode_image_id="$(ssh $SSH_OPTS "$DECODE_NODE" docker image inspect -f '{{.Id}}' "$IMAGE")"
[[ "$prefill_image_id" == "$decode_image_id" ]] || {
    echo "[sweep] image IDs differ: prefill=$prefill_image_id decode=$decode_image_id" >&2
    exit 1
}
{
    echo "prefill_image_id=$prefill_image_id"
    echo "decode_image_id=$decode_image_id"
    echo "source_id=${SOURCE_ID:-standalone}"
    echo "config_id=${CONFIG_ID:-standalone}"
} >>"$ROOT/agentx/run_manifest.txt"

write_point_signature() {
    local concurrency="$1" output="$2"
    {
        printf '%s=%s\n' \
            signature_format 1 run_id "$RUN_ID" \
            prefill_node "$PREFILL_NODE" decode_node "$DECODE_NODE" \
            prefill_ip "$PREFILL_IP" decode_ip "$DECODE_IP" \
            image "$IMAGE" image_id "$prefill_image_id" \
            model "$MODEL" served_model "$SERVED_MODEL" \
            concurrency "$concurrency" duration_seconds "$AGENTX_DURATION" \
            agentx_cache_root "${AGENTX_CACHE_ROOT:-$WORKSPACE_ROOT/.cache/agentx}" \
            topology "prefill_tp${PREFILL_TP_SIZE}_ep${PREFILL_EP_SIZE}_dp${PREFILL_DP_SIZE}+decode_tp${DECODE_TP_SIZE}_ep${DECODE_EP_SIZE}_dp${DECODE_DP_SIZE}" \
            context_length "$CONTEXT_LENGTH" chunked_prefill_size "$CHUNKED_PREFILL_SIZE" \
            prefill_mem_fraction "$PREFILL_MEM_FRACTION" \
            decode_mem_fraction "$DECODE_MEM_FRACTION" \
            prefill_dpa "$PREFILL_DPA" decode_dpa "$DECODE_DPA" \
            enable_mtp "$ENABLE_MTP" spec_steps "$SPEC_STEPS" \
            spec_draft_tokens "$SPEC_DRAFT_TOKENS" spec_topk "$SPEC_TOPK" \
            simulate_acc_len "$SIMULATE_ACC_LEN" enable_kv_aware "$ENABLE_KV_AWARE" \
            kv_prefill_overlap_weight "$KV_PREFILL_OVERLAP_WEIGHT" \
            kv_decode_overlap_weight "$KV_DECODE_OVERLAP_WEIGHT" \
            enable_hicache "$ENABLE_HICACHE" \
            prefill_enable_hicache "$PREFILL_ENABLE_HICACHE" \
            decode_enable_hicache "$DECODE_ENABLE_HICACHE" \
            hicache_ratio "$HICACHE_RATIO" hicache_write_policy "$HICACHE_WRITE_POLICY" \
            hicache_io_backend "$HICACHE_IO_BACKEND" hicache_mem_layout "$HICACHE_MEM_LAYOUT" \
            kv_p2p_transfer "$KV_P2P_TRANSFER" rdma_device "$RDMA_DEVICE" \
            mc_te_filters "$MC_TE_FILTERS" mc_gid_index "$MC_GID_INDEX" \
            failed_request_threshold "$AGENTX_FAILED_REQUEST_THRESHOLD" \
            gpu_idle_timeout "$GPU_IDLE_TIMEOUT" \
            source_id "${SOURCE_ID:-standalone}" config_id "${CONFIG_ID:-standalone}"
        printf 'config_sha256=%s\n' "$(sha256sum "$DIR/config.sh" | awk '{print $1}')"
        printf 'launch_sha256=%s\n' "$(sha256sum "$DIR/launch.sh" | awk '{print $1}')"
        printf 'agentx_point_sha256=%s\n' "$(sha256sum "$DIR/agentx_point.sh" | awk '{print $1}')"
        printf 'benchmark_lib_sha256=%s\n' "$(
            sha256sum "$INFERENCEX_DIR/benchmarks/benchmark_lib.sh" | awk '{print $1}'
        )"
    } >"$output"
}

overall_rc=0
for concurrency in "${concurrencies[@]}"; do
    point="$ROOT/agentx/c$concurrency"
    signature_tmp="$ROOT/agentx/.c${concurrency}.signature.$$"
    write_point_signature "$concurrency" "$signature_tmp"
    if [[ -s "$point/agentx_conc$concurrency.json" &&
        -f "$point/PASS" &&
        -f "$point/validation_signature.txt" ]] &&
        cmp -s "$signature_tmp" "$point/validation_signature.txt"; then
        echo "[sweep] C$concurrency already complete; skipping"
        rm -f "$signature_tmp"
        continue
    fi
    if [[ -e "$point/PASS" ]]; then
        echo "[sweep] C$concurrency PASS exists but its validation signature is stale; rerunning"
    fi

    passed=0
    for attempt in 1 2; do
        if [[ -d "$point" ]]; then
            failed_point="${point}.failed_$(date -u +%Y%m%dT%H%M%S%NZ)_pid$$"
            [[ ! -e "$failed_point" ]] || {
                echo "[sweep] archive already exists: $failed_point" >&2
                exit 73
            }
            mv "$point" "$failed_point"
        fi
        mkdir -p "$point"
        echo "[sweep] C$concurrency attempt $attempt/2"

        set +e
        RUN_ID="$RUN_ID" \
        MAX_RUNNING_REQUESTS="$concurrency" \
        CUDA_GRAPH_MAX_BS="$concurrency" \
        SIMULATE_ACC_LEN="$SIMULATE_ACC_LEN" \
            bash "$DIR/launch.sh" 2>&1 | tee "$point/launch.log"
        launch_rc=${PIPESTATUS[0]}
        set -e
        if (( launch_rc != 0 )); then
            echo "[sweep] C$concurrency launch failed (rc=$launch_rc)" | tee "$point/FAIL"
            RUN_ID="$RUN_ID" bash "$DIR/stop.sh" >"$point/stop.log" 2>&1 || true
            continue
        fi

        set +e
        RUN_ID="$RUN_ID" \
        MAX_RUNNING_REQUESTS="$concurrency" \
        CUDA_GRAPH_MAX_BS="$concurrency" \
        SIMULATE_ACC_LEN="$SIMULATE_ACC_LEN" \
            bash "$DIR/agentx_point.sh" "$concurrency" "$point" \
            2>&1 | tee "$point/agentx.log"
        point_rc=${PIPESTATUS[0]}
        set -e
        if (( point_rc == 0 )); then
            cp "$signature_tmp" "$point/validation_signature.txt"
            date -Is >"$point/PASS"
            passed=1
            break
        fi

        echo "[sweep] C$concurrency AgentX failed (rc=$point_rc)" | tee "$point/FAIL"
        RUN_ID="$RUN_ID" bash "$DIR/stop.sh" >"$point/stop.log" 2>&1 || true
    done

    if (( passed == 0 )); then
        overall_rc=1
        echo "[sweep] C$concurrency failed both attempts; continuing on the same nodes"
    fi
    rm -f "$signature_tmp"
done

{
    echo "finished_at=$(date -Is)"
    echo "overall_rc=$overall_rc"
} >>"$ROOT/agentx/run_manifest.txt"
RUN_ID="$RUN_ID" bash "$DIR/stop.sh" >"$ROOT/agentx/final_stop.log" 2>&1 || true
exit "$overall_rc"
