#!/usr/bin/env bash
# Run the TP4/EP4 AgentX campaign, rotating idle prefill nodes when possible.
# The production-WRITE preflight must be reviewed before invoking this script.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ID="${CAMPAIGN_ID:-v518_tp4ep4_campaign_$(date -u +%Y%m%d_%H%MZ)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$DIR/results/$CAMPAIGN_ID}"
DECODE_NODE="${DECODE_NODE:-crsuse2-m2m-142}"
read -r -a PREFILL_CANDIDATES <<<"${PREFILL_CANDIDATES:-crsuse2-m2m-140 crsuse2-m2m-138}"
read -r -a CONCURRENCIES <<<"${CONCURRENCIES:-4 8 16 32 64}"
IDLE_POLL_SECONDS="${IDLE_POLL_SECONDS:-300}"
IDLE_VRAM_LIMIT="${IDLE_VRAM_LIMIT:-10}"
AGENTX_DURATION="${AGENTX_DURATION:-1200}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1500000}"
DPA_JSON_MODEL_OVERRIDE_ARGS="${DPA_JSON_MODEL_OVERRIDE_ARGS:-{\"index_share_for_mtp_iteration\":false}}"
DPA_AIPERF_WARMUP_REQUESTS_PER_LANE="${DPA_AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no}"
IMAGE="${IMAGE:-infera/engine-sglang:glm52-v518-402df1e-2c71811}"
# Pair-specific values may have leaked from a previously sourced config.sh.
# Every point must resolve fresh data-plane IPs and its own lifecycle lock.
unset PREFILL_IP DECODE_IP BENCH_NODE_PAIR BENCH_LOCK_FILE
unset GLM52_BENCH_LOCK_FD GLM52_BENCH_LOCK_HELD

mkdir -p "$CAMPAIGN_ROOT/campaign"
CAMPAIGN_LOG="$CAMPAIGN_ROOT/campaign/run.log"
SCHEDULE="$CAMPAIGN_ROOT/campaign/schedule.tsv"
exec > >(tee -a "$CAMPAIGN_LOG") 2>&1
exec 9>"$CAMPAIGN_ROOT/campaign/driver.lock"
flock -n 9 || {
    echo "[campaign] another driver owns $CAMPAIGN_ROOT"
    exit 75
}

if [[ ! -f "$SCHEDULE" ]]; then
    printf 'timestamp\tstrategy\tconcurrency\tattempt\tprefill\tdecode\tstatus\trc\n' >"$SCHEDULE"
fi

max_vram_pct() {
    ssh $SSH_OPTS "$1" rocm-smi --showmemuse 2>/dev/null |
        awk -F': ' '
            /VRAM%/ {n++; if ($NF + 0 > max) max=$NF + 0}
            END {if (n == 8) print max + 0}
        '
}

image_id() {
    ssh $SSH_OPTS "$1" docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null
}

idle_node() {
    local node="$1" vram
    vram="$(max_vram_pct "$node")" || return 1
    [[ -n "$vram" && "$vram" -le "$IDLE_VRAM_LIMIT" ]]
}

wait_for_decode() {
    local vram
    while ! idle_node "$DECODE_NODE"; do
        vram="$(max_vram_pct "$DECODE_NODE" 2>/dev/null || echo unknown)"
        echo "[campaign] decode $DECODE_NODE not idle (max VRAM=$vram%); retrying in ${IDLE_POLL_SECONDS}s"
        sleep "$IDLE_POLL_SECONDS"
    done
}

SELECTED_PREFILL=""
select_prefill() {
    local preferred_index="$1" excluded="${2:-}" count="${#PREFILL_CANDIDATES[@]}"
    local offset index node vram
    while true; do
        for ((offset = 0; offset < count; offset++)); do
            index=$(((preferred_index + offset) % count))
            node="${PREFILL_CANDIDATES[$index]}"
            [[ "$node" != "$excluded" ]] || continue
            if idle_node "$node"; then
                SELECTED_PREFILL="$node"
                return 0
            fi
            vram="$(max_vram_pct "$node" 2>/dev/null || echo unknown)"
            echo "[campaign] prefill $node not idle (max VRAM=$vram%)"
        done
        echo "[campaign] no eligible prefill is idle; retrying in ${IDLE_POLL_SECONDS}s"
        sleep "$IDLE_POLL_SECONDS"
        excluded=""
    done
}

record_gpu_state() {
    local output="$1" node
    {
        echo "captured_at=$(date -Is)"
        for node in "${PREFILL_CANDIDATES[@]}" "$DECODE_NODE"; do
            echo "===== $node ====="
            ssh $SSH_OPTS "$node" 'rocm-smi --showpids; rocm-smi --showmeminfo vram --json' || true
        done
    } >"$output"
}

expected_image_id="$(image_id "$DECODE_NODE")" || {
    echo "[campaign] image missing on decode $DECODE_NODE: $IMAGE"
    exit 66
}
for node in "${PREFILL_CANDIDATES[@]}"; do
    candidate_id="$(image_id "$node")" || {
        echo "[campaign] image missing on prefill candidate $node: $IMAGE"
        exit 66
    }
    [[ "$candidate_id" == "$expected_image_id" ]] || {
        echo "[campaign] image mismatch: $node=$candidate_id $DECODE_NODE=$expected_image_id"
        exit 66
    }
done

echo "[campaign] id=$CAMPAIGN_ID root=$CAMPAIGN_ROOT"
echo "[campaign] image=$IMAGE id=$expected_image_id"
echo "[campaign] prefills=${PREFILL_CANDIDATES[*]} decode=$DECODE_NODE"
echo "[campaign] concurrencies=${CONCURRENCIES[*]} duration=${AGENTX_DURATION}s"
echo "[campaign] ionic max_total_tokens=$MAX_TOTAL_TOKENS"

overall_rc=0
point_index=0
for strategy in tp4ep4 tp4ep4_dpa; do
    strategy_root="$CAMPAIGN_ROOT/$strategy"
    strategy_run_id="${CAMPAIGN_ID}_${strategy}"
    mkdir -p "$strategy_root/agentx"
    if [[ "$strategy" == "tp4ep4" ]]; then
        dp_size=1
        dpa=0
        json_model_override_args=""
        aiperf_warmup_requests_per_lane=10
    else
        dp_size=4
        dpa=1
        json_model_override_args="$DPA_JSON_MODEL_OVERRIDE_ARGS"
        aiperf_warmup_requests_per_lane="$DPA_AIPERF_WARMUP_REQUESTS_PER_LANE"
    fi

    for concurrency in "${CONCURRENCIES[@]}"; do
        point="$strategy_root/agentx/c$concurrency"
        if [[ -s "$point/agentx_conc$concurrency.json" && -f "$point/PASS" ]]; then
            echo "[campaign] $strategy C$concurrency already has PASS; skipping"
            ((point_index += 1))
            continue
        fi

        point_passed=0
        previous_prefill=""
        for pair_attempt in 1 2; do
            wait_for_decode
            select_prefill "$((point_index % ${#PREFILL_CANDIDATES[@]}))" "$previous_prefill"
            prefill="$SELECTED_PREFILL"
            previous_prefill="$prefill"
            echo "[campaign] starting $strategy C$concurrency pair_attempt=$pair_attempt on $prefill/$DECODE_NODE"
            printf '%s\t%s\t%s\t%s\t%s\t%s\tSTART\t\n' \
                "$(date -Is)" "$strategy" "$concurrency" "$pair_attempt" \
                "$prefill" "$DECODE_NODE" >>"$SCHEDULE"

            set +e
            RUN_ID="$strategy_run_id" \
            RUN_ROOT="$strategy_root" \
            PREFILL_NODE="$prefill" \
            DECODE_NODE="$DECODE_NODE" \
            IMAGE="$IMAGE" \
            PREFILL_TP_SIZE=4 DECODE_TP_SIZE=4 \
            PREFILL_EP_SIZE=4 DECODE_EP_SIZE=4 \
            PREFILL_DP_SIZE="$dp_size" DECODE_DP_SIZE="$dp_size" \
            PREFILL_DPA="$dpa" DECODE_DPA="$dpa" \
            JSON_MODEL_OVERRIDE_ARGS="$json_model_override_args" \
            AGENTX_DURATION="$AGENTX_DURATION" \
            AIPERF_WARMUP_REQUESTS_PER_LANE="$aiperf_warmup_requests_per_lane" \
            AGENTX_CONCURRENCIES="$concurrency" \
            MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
            SWEEP_ATTEMPTS=1 \
            GPU_IDLE_TIMEOUT=3600 \
            SOURCE_ID="${SOURCE_ID:-campaign}" \
            CONFIG_ID="${strategy}_dp${dp_size}_dpa${dpa}_duration${AGENTX_DURATION}_warmup${aiperf_warmup_requests_per_lane}_tokens${MAX_TOTAL_TOKENS}_indexshare$([[ -n "$json_model_override_args" ]] && echo off || echo default)" \
                bash "$DIR/run_agentx_sweep.sh" "$concurrency"
            point_rc=$?
            set -e

            cp "$strategy_root/agentx/run_manifest.txt" \
                "$strategy_root/agentx/run_manifest_c${concurrency}_attempt${pair_attempt}.txt" \
                2>/dev/null || true
            record_gpu_state \
                "$strategy_root/agentx/gpu_after_c${concurrency}_attempt${pair_attempt}.log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$(date -Is)" "$strategy" "$concurrency" "$pair_attempt" \
                "$prefill" "$DECODE_NODE" \
                "$([[ "$point_rc" == 0 ]] && echo PASS || echo FAIL)" "$point_rc" \
                >>"$SCHEDULE"

            if (( point_rc == 0 )); then
                point_passed=1
                break
            fi
            echo "[campaign] $strategy C$concurrency failed on $prefill; trying the alternate prefill"
        done

        if (( point_passed == 0 )); then
            overall_rc=1
            echo "[campaign] $strategy C$concurrency failed on both pair attempts; continuing"
        fi
        ((point_index += 1))
    done

    set +e
    RUN_ID="$strategy_run_id" \
    RUN_ROOT="$strategy_root" \
    PREFILL_NODE="${PREFILL_CANDIDATES[0]}" \
    DECODE_NODE="$DECODE_NODE" \
    IMAGE="$IMAGE" \
    PREFILL_TP_SIZE=4 DECODE_TP_SIZE=4 \
    PREFILL_EP_SIZE=4 DECODE_EP_SIZE=4 \
    PREFILL_DP_SIZE="$dp_size" DECODE_DP_SIZE="$dp_size" \
    PREFILL_DPA="$dpa" DECODE_DPA="$dpa" \
    JSON_MODEL_OVERRIDE_ARGS="$json_model_override_args" \
    MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
        bash "$DIR/analyze_agentx.sh" "$strategy_root" \
        >"$strategy_root/analysis.log" 2>&1
    analysis_rc=$?
    set -e
    (( analysis_rc == 0 )) || {
        overall_rc=1
        echo "[campaign] analysis failed for $strategy (rc=$analysis_rc)"
    }
done

record_gpu_state "$CAMPAIGN_ROOT/campaign/final_gpu_state.log"
echo "[campaign] finished overall_rc=$overall_rc at $(date -Is)"
exit "$overall_rc"
