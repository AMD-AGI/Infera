#!/usr/bin/env bash
# Run one canonical InferenceX AgentX point against an already-running 1P1D stack.
# Usage: bash agentx_point.sh CONC [OUT_DIR]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

CONC="${1:?usage: $0 CONC [OUT_DIR]}"
[[ "$CONC" =~ ^[0-9]+$ ]] || { echo "CONC must be an integer" >&2; exit 64; }
OUT="${2:-$RUN_ROOT/agentx/c$CONC}"
mkdir -p "$OUT"

capture_container() {
    local node="$1" container="$2" name="$3"
    ssh $SSH_OPTS "$node" docker logs "$container" \
        >"$OUT/$name.log" 2>&1 || true
    ssh $SSH_OPTS "$node" docker inspect "$container" \
        >"$OUT/${name}_container.final.json" 2>&1 || true
}

capture_stack_logs() {
    capture_container "$PREFILL_NODE" glm52-pd-prefill prefill
    capture_container "$DECODE_NODE" glm52-pd-decode decode
    capture_container "$PREFILL_NODE" glm52-pd-router router
}

cleanup() {
    local pid
    for pid in "${monitor_prefill:-}" "${monitor_decode:-}"; do
        [[ -n "$pid" ]] || continue
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
    done
    ssh $SSH_OPTS "$PREFILL_NODE" docker rm -f glm52-agentx-client >/dev/null 2>&1 || true
}

on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    cleanup
    capture_stack_logs
    exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

REMOTE_RUN_ROOT="$REMOTE_WORK_ROOT/$RUN_ID"
REMOTE_OUT="$REMOTE_RUN_ROOT/agentx/c$CONC"
REMOTE_CACHE_ROOT="${AGENTX_CACHE_ROOT:-$WORKSPACE_ROOT/.cache/agentx}"
ssh $SSH_OPTS "$PREFILL_NODE" \
    "rm -rf '$REMOTE_OUT' && mkdir -p '$REMOTE_OUT' '$REMOTE_CACHE_ROOT/aiperf' '$REMOTE_CACHE_ROOT/hf'"
trace_cache_repo="$REMOTE_CACHE_ROOT/hf/hub/datasets--semianalysisai--cc-traces-weka-062126"
hf_offline=0
if [[ -s "$trace_cache_repo/refs/main" ]]; then
    trace_revision="$(<"$trace_cache_repo/refs/main")"
    [[ -s "$trace_cache_repo/snapshots/$trace_revision/traces.jsonl" ]] && hf_offline=1
fi

# AIPerf loads the tokenizer from $MODEL inside the client container.
ssh $SSH_OPTS "$PREFILL_NODE" "test -r '$MODEL/tokenizer_config.json'" || {
    echo "[agentx] tokenizer not readable at $MODEL on $PREFILL_NODE" >&2
    exit 66
}

URL="http://$PREFILL_IP:$ROUTER_PORT"
curl -fsS -m 30 "$URL/health" >"$OUT/router_health.json"
curl -fsS -m 30 "$URL/v1/workers" >"$OUT/workers.json"
curl -fsS -m 30 "http://$PREFILL_IP:$PREFILL_PORT/server_info" >"$OUT/prefill_server_info.json"
curl -fsS -m 30 "http://$DECODE_IP:$DECODE_PORT/server_info" >"$OUT/decode_server_info.json"
ssh $SSH_OPTS "$DECODE_NODE" docker inspect glm52-pd-decode \
    >"$OUT/decode_container.json"

detect_runner_type() {
    ssh $SSH_OPTS "$1" rocm-smi --showproductname 2>/dev/null \
        | awk '{
            if (!found) {
                line = toupper($0)
                if (match(line, /MI[0-9]+[A-Z]*/)) {
                    print tolower(substr(line, RSTART, RLENGTH))
                    found = 1
                }
            }
        }'
}
prefill_runner_type="$(detect_runner_type "$PREFILL_NODE")"
decode_runner_type="$(detect_runner_type "$DECODE_NODE")"
[[ -n "$prefill_runner_type" && "$prefill_runner_type" == "$decode_runner_type" ]] || {
    echo "[agentx] cannot establish one matching GPU type: prefill=$prefill_runner_type decode=$decode_runner_type" >&2
    exit 66
}
[[ "$prefill_runner_type" == "$AGENTX_RUNNER_TYPE" ]] || {
    echo "[agentx] detected $prefill_runner_type, expected AGENTX_RUNNER_TYPE=$AGENTX_RUNNER_TYPE" >&2
    exit 66
}

python3 - "$OUT" "$CONC" <<'PY'
import json
import os
import sys
from pathlib import Path

out, conc = Path(sys.argv[1]), int(sys.argv[2])
expected = {
    "prefill": {
        "tp_size": int(os.environ["PREFILL_TP_SIZE"]),
        "ep_size": int(os.environ["PREFILL_EP_SIZE"]),
        "dp_size": int(os.environ["PREFILL_DP_SIZE"]),
        "enable_dp_attention": os.environ["PREFILL_DPA"] == "1",
        "enable_hierarchical_cache": os.environ["PREFILL_ENABLE_HICACHE"] == "1",
    },
    "decode": {
        "tp_size": int(os.environ["DECODE_TP_SIZE"]),
        "ep_size": int(os.environ["DECODE_EP_SIZE"]),
        "dp_size": int(os.environ["DECODE_DP_SIZE"]),
        "enable_dp_attention": os.environ["DECODE_DPA"] == "1",
        "enable_hierarchical_cache": os.environ["DECODE_ENABLE_HICACHE"] == "1",
    },
}
workers = json.loads((out / "workers.json").read_text())["workers"]
assert len(workers) == 2, workers
assert {w["disagg_mode"] for w in workers} == {"prefill", "decode"}, workers
info_by_role = {}
for role in ("prefill", "decode"):
    info = json.loads((out / f"{role}_server_info.json").read_text())
    info_by_role[role] = info
    for key in ("tp_size", "ep_size", "dp_size"):
        actual = info.get(key) if key == "tp_size" else info.get(key) or 1
        assert actual == expected[role][key], (
            role, key, actual, expected[role][key]
        )
    for key in ("enable_dp_attention", "enable_hierarchical_cache"):
        actual = bool(info.get(key, expected[role][key]))
        assert actual == expected[role][key], (
            role, key, actual, expected[role][key]
        )
    assert info.get("max_running_requests") == conc, (
        role, info.get("max_running_requests"), conc
    )

served = info_by_role["prefill"].get("served_model_name") or os.environ["SERVED_MODEL"]
served = served[0] if isinstance(served, list) else served
decode_served = info_by_role["decode"].get("served_model_name") or served
decode_served = decode_served[0] if isinstance(decode_served, list) else decode_served
assert served == decode_served == os.environ["SERVED_MODEL"], (
    served, decode_served, os.environ["SERVED_MODEL"]
)

container = json.loads((out / "decode_container.json").read_text())[0]
container_cmd = container["Config"].get("Cmd", [])
expected_mtp = os.environ["ENABLE_MTP"] == "1"
assert ("--speculative-algorithm" in container_cmd) == expected_mtp, (
    "MTP command mismatch",
    container_cmd,
    expected_mtp,
)
transfer_index = container_cmd.index("--disaggregation-transfer-backend")
assert container_cmd[transfer_index + 1] == os.environ["KV_P2P_TRANSFER"], (
    "KV transfer mismatch",
    container_cmd[transfer_index + 1],
    os.environ["KV_P2P_TRANSFER"],
)
container_env = dict(
    item.split("=", 1)
    for item in container["Config"].get("Env", [])
    if "=" in item
)
actual_simulate_acc_len = container_env.get("SGLANG_SIMULATE_ACC_LEN", "")
assert actual_simulate_acc_len == os.environ["SIMULATE_ACC_LEN"], (
    "SIMULATE_ACC_LEN mismatch",
    actual_simulate_acc_len or "off",
    os.environ["SIMULATE_ACC_LEN"] or "off",
)

hicache = any(
    expected[role]["enable_hierarchical_cache"] for role in ("prefill", "decode")
)
server_env = {
    "SERVED_MODEL_NAME": served,
    "SPEC_DECODING": "mtp" if expected_mtp else "none",
    "IS_MULTINODE": "true",
    "DISAGG": "true",
    "KV_P2P_TRANSFER": os.environ["KV_P2P_TRANSFER"],
    "PREFILL_NUM_WORKERS": "1",
    "PREFILL_TP": str(expected["prefill"]["tp_size"]),
    "PREFILL_EP": str(expected["prefill"]["ep_size"]),
    "PREFILL_DP_ATTN": str(expected["prefill"]["enable_dp_attention"]).lower(),
    "DECODE_NUM_WORKERS": "1",
    "DECODE_TP": str(expected["decode"]["tp_size"]),
    "DECODE_EP": str(expected["decode"]["ep_size"]),
    "DECODE_DP_ATTN": str(expected["decode"]["enable_dp_attention"]).lower(),
    "KV_OFFLOADING": "dram" if hicache else "none",
    "KV_OFFLOAD_BACKEND": "hicache" if hicache else "",
    "KV_OFFLOAD_BACKEND_METADATA": '{"name":"hicache"}' if hicache else "",
}
(out / "server.env").write_text(
    "".join(f"{key}={value}\n" for key, value in server_env.items())
)
print(
    "server contract: PASS "
    f"(simulate_acc_len={actual_simulate_acc_len or 'off'})"
)
PY

{
    echo "run_id=$RUN_ID"
    echo "prefill_node=$PREFILL_NODE"
    echo "decode_node=$DECODE_NODE"
    echo "prefill_ip=$PREFILL_IP"
    echo "decode_ip=$DECODE_IP"
    echo "image=$IMAGE"
    echo "concurrency=$CONC"
    echo "duration=$AGENTX_DURATION"
    echo "simulate_acc_len=${SIMULATE_ACC_LEN:-off}"
    echo "agentx_cache_root=$REMOTE_CACHE_ROOT"
    echo "hf_dataset_cache_ready=$hf_offline"
    echo "runner_type=$prefill_runner_type"
    echo "infera_head=$(git -C "$DIR/../.." rev-parse HEAD 2>/dev/null || true)"
    echo "rocm_llm_bench_head=$(git -C "$ROCM_LLM_BENCH_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "inferencex_head=$(git -C "$INFERENCEX_DIR" rev-parse HEAD 2>/dev/null || true)"
    echo "prefill_image_id=$(ssh $SSH_OPTS "$PREFILL_NODE" docker image inspect -f '{{.Id}}' "$IMAGE")"
    echo "decode_image_id=$(ssh $SSH_OPTS "$DECODE_NODE" docker image inspect -f '{{.Id}}' "$IMAGE")"
} >"$OUT/stack.txt"

monitor_gpu() {
    local node="$1" output="$2"
    ssh $SSH_OPTS "$node" '
        while true; do
            date -Is
            rocm-smi --showuse --showmemuse --showpower 2>/dev/null
            sleep 5
        done
    ' >"$output" 2>&1
}

monitor_gpu "$PREFILL_NODE" "$OUT/gpu_prefill.log" &
monitor_prefill=$!
monitor_gpu "$DECODE_NODE" "$OUT/gpu_decode.log" &
monitor_decode=$!

cp "$OUT/server.env" "$OUT/client.env"
cat >>"$OUT/client.env" <<EOF
# Replay inputs.
MODEL=$MODEL
MODEL_PREFIX=$MODEL_PREFIX
CONC=$CONC
DURATION=$AGENTX_DURATION
PORT=$ROUTER_PORT
# GLM-5.2 is a native 1M-context family, so replay the complete corpus and let
# benchmark_lib drop any MAX_MODEL_LEN cap. This matches the upstream recipe.
IS_AGENTIC=1

# Result metadata. Observable topology/speculation values were checked against
# /server_info and the running decode container above.
FRAMEWORK=$AGENTX_FRAMEWORK
PRECISION=$AGENTX_PRECISION
RUNNER_TYPE=$prefill_runner_type
IMAGE=$IMAGE
TOTAL_CPU_DRAM_GB=$AGENTX_TOTAL_CPU_DRAM_GB

# AIPerf validation and server metrics.
ENABLE_AGENTX_POWER=0
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:
AIPERF_FAILED_REQUEST_THRESHOLD=$AGENTX_FAILED_REQUEST_THRESHOLD
AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=$AGENTX_FAILED_REQUEST_THRESHOLD
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
AIPERF_SERVER_URL=http://localhost:$ROUTER_PORT
AIPERF_SERVER_METRICS_URLS=http://localhost:$ROUTER_PORT/metrics,http://$PREFILL_IP:$PREFILL_PORT/metrics,http://$DECODE_IP:$DECODE_PORT/metrics

# Client-container state and output paths.
AIPERF_RUNTIME_DIR=$REMOTE_CACHE_ROOT/aiperf
HF_HOME=$REMOTE_CACHE_ROOT/hf
AGENTX_HF_CACHE_READY=$hf_offline
HF_DATASETS_OFFLINE=$hf_offline
INFMAX_CONTAINER_WORKSPACE=$INFERENCEX_DIR
AGENTIC_OUTPUT_DIR=$REMOTE_OUT
RESULT_FILENAME=agentx_conc$CONC
HOST_UID=$(id -u)
HOST_GID=$(id -g)
PYTHONNOUSERSITE=1
EOF

echo "[agentx] C$CONC for ${AGENTX_DURATION}s on $PREFILL_NODE/$DECODE_NODE"
set +e
ssh $SSH_OPTS "$PREFILL_NODE" docker run -i --rm \
    --name glm52-agentx-client \
    --network host --ipc host --shm-size 32g \
    -v "$WORKSPACE_ROOT:$WORKSPACE_ROOT" \
    -v "$REMOTE_RUN_ROOT:$REMOTE_RUN_ROOT" \
    -v "$REMOTE_CACHE_ROOT:$REMOTE_CACHE_ROOT" \
    --mount "type=bind,src=$MODEL,dst=$MODEL,readonly" \
    --env-file "$OUT/client.env" \
    "$IMAGE" bash -s -- "$REMOTE_OUT" <<'AGENTX_CLIENT' 2>&1 | tee "$OUT/runner.log"
set -e
out=$1
trap "chown -R $HOST_UID:$HOST_GID $out $AIPERF_RUNTIME_DIR $HF_HOME" EXIT
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
install_agentic_deps
trace_ready=0
for attempt in 1 2 3 4; do
    if [[ "$AGENTX_HF_CACHE_READY" == "1" ]]; then
        export HF_HUB_OFFLINE=1
    fi
    if resolve_trace_source; then
        trace_ready=1
        unset HF_HUB_OFFLINE
        break
    fi
    unset HF_HUB_OFFLINE
    if (( attempt < 4 )); then
        delay=$((attempt * 120))
        echo "[agentx] trace source unavailable; retrying in ${delay}s ($attempt/4)" >&2
        sleep "$delay"
    fi
done
(( trace_ready == 1 )) || {
    echo "[agentx] trace source unavailable after 4 attempts" >&2
    exit 1
}
build_replay_cmd "$out"
run_agentic_replay_and_write_outputs "$out"
AGENTX_CLIENT
client_rc=${PIPESTATUS[0]}
set -e

ssh $SSH_OPTS "$PREFILL_NODE" "tar -C '$REMOTE_OUT' -cf - ." \
    | tar -C "$OUT" -xf -
(( client_rc == 0 )) || {
    echo "[agentx] remote client failed with rc=$client_rc; copied partial artifacts" >&2
    exit "$client_rc"
}

test -s "$OUT/agentx_conc$CONC.json"
echo "[agentx] PASS C$CONC: $OUT/agentx_conc$CONC.json"
