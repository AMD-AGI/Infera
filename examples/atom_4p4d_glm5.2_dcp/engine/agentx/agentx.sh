#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# One InferenceX AgentX (agentic-coding trace replay) point against the running
# service, from a client container in the same image. The concurrency is the
# CONC the engines were started with. Output: <run>/agentx/agentx_conc<CONC>.json
# plus the AIPerf exports, runtime.env and the replay command.
#
# The reference numbers used DURATION=3600, WARMUP_PER_LANE=10 and forced MTP
# acceptance (MTP_AL=2.99 for K3, 3.33 for K4, set on up.sh).
#
# Usage: bash engine/agentx/agentx.sh
#   DURATION=3600 WARMUP_PER_LANE=10   measured seconds, warm-up requests per lane
#   PREPARE_ONLY=1                     install client deps and the dataset only;
#                                      sends no requests, needs no running service
#   AGENTX_CONSTRAINTS=<file>          uv constraints; default engine/agentx/constraints.txt,
#                                      empty to resolve freshly
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../common.sh
source "$KIT/engine/common.sh"

INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_REF="${INFERENCEX_REF:-918524ff94045b3f091115f1051c22a8588edf2b}"
AGENTX_CONSTRAINTS="${AGENTX_CONSTRAINTS-$KIT/engine/agentx/constraints.txt}"
DURATION="${DURATION:-3600}"
WARMUP_PER_LANE="${WARMUP_PER_LANE:-10}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

if [[ "$PREPARE_ONLY" == 1 ]]; then
    resolve_model
    out="$WORK_DIR/prepared/c$CONC/agentx"
else
    load_run
    out="$RUN_DIR/agentx"
fi
ix="$WORK_DIR/cache/InferenceX"
cache="$WORK_DIR/cache/agentx"
mkdir -p "$out/tmp" "$cache"
if [[ ! -d "$ix/.git" ]]; then
    git clone --filter=blob:none "$INFERENCEX_REPO" "$ix"
    git -C "$ix" checkout --detach "$INFERENCEX_REF"
    git -C "$ix" submodule update --init --recursive
fi
[[ "$(git -C "$ix" rev-parse HEAD)" == "$INFERENCEX_REF" ]] ||
    die "InferenceX checkout $ix is not at $INFERENCEX_REF"

# AgentX metadata: the prefill's LMCache CPU tier is DRAM offload.
offload="KV_OFFLOADING=none"
if ((PREFILL_OFFLOAD_GB > 0)); then
    lmcache_version="$(docker run --rm "$IMAGE" pip show lmcache | awk '/^Version:/ {print $2}')"
    offload=$'KV_OFFLOADING=dram\nKV_OFFLOAD_BACKEND=lmcache\n'
    offload+="KV_OFFLOAD_BACKEND_METADATA={\"name\":\"lmcache\",\"version\":\"${lmcache_version%%+*}\"}"
fi
# The scraped /metrics come from the router's worker list, so workers on
# other nodes are included. PREPARE_ONLY has no router to ask: this node's ports.
if [[ "$PREPARE_ONLY" == 1 ]]; then
    metrics="$ROUTER_URL/metrics,http://$NODE_IP:$PREFILL_PORT/metrics,http://$NODE_IP:$DECODE_PORT/metrics"
else
    workers="$(curl -sf -m 10 "$ROUTER_URL/v1/workers")" || die "cannot list workers: $ROUTER_URL/v1/workers"
    metrics="$ROUTER_URL/metrics"
    for url in $(grep -o '"url": *"[^"]*"' <<<"$workers" | cut -d'"' -f4); do metrics+=",$url/metrics"; done
fi

# TMPDIR stays short: AIPerf's Unix socket paths must fit in 107 bytes.
cat >"$out/runtime.env" <<EOF
INFMAX_CONTAINER_WORKSPACE=$ix
INFERENCEX_COMMIT=$INFERENCEX_REF
MODEL=$MODEL
SERVED_MODEL_NAME=$MODEL
MODEL_PREFIX=glm5.2
IMAGE=$IMAGE
FRAMEWORK=atom
PRECISION=fp4
RUNNER_TYPE=mi355x
PREFILL_HARDWARE=mi355x
DECODE_HARDWARE=mi355x
PORT=$ROUTER_PORT
ROUTER_POLICY=round-robin/session-affinity-$SESSION_AFFINITY
AIPERF_SERVER_URL=$ROUTER_URL
AIPERF_SERVER_METRICS_URLS=$metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom:
AIPERF_FAILED_REQUEST_THRESHOLD=0.10
AIPERF_WARMUP_REQUESTS_PER_LANE=$WARMUP_PER_LANE
AIPERF_HTTP_TCP_USER_TIMEOUT=900000
CONC=$CONC
DURATION=$DURATION
IS_AGENTIC=1
SCENARIO_TYPE=agentic-coding
IS_MULTINODE=${IS_MULTINODE:-false}
DISAGG=true
ENABLE_AGENTX_POWER=0
TP=4
EP_SIZE=1
DP_ATTENTION=false
PREFILL_NUM_WORKERS=$PREFILL_NUM_WORKERS
PREFILL_TP=4
PREFILL_DCP_SIZE=4
PREFILL_EP=1
PREFILL_DP_ATTN=false
DECODE_NUM_WORKERS=1
DECODE_TP=4
DECODE_EP=1
DECODE_DCP_SIZE=4
DECODE_DP_ATTN=false
SPEC_DECODING=mtp
SIMULATE_ACC_LEN=$MTP_AL
$offload
TOTAL_CPU_DRAM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
RESULT_FILENAME=agentx_conc$CONC
AGENTIC_OUTPUT_DIR=$out
AIPERF_RUNTIME_DIR=$cache/aiperf
AIPERF_DATASET_MMAP_CACHE_DIR=$cache/dataset_mmap
MPLCONFIGDIR=$cache/matplotlib
HF_HOME=$cache/hf
UV_PYTHON_INSTALL_DIR=$cache/python
XDG_CACHE_HOME=$cache/xdg
XDG_CONFIG_HOME=$cache/xdg-config
UV_NO_MODIFY_PATH=1
TMPDIR=/ax-tmp
PYTHONNOUSERSITE=1
EOF
constraint_mount=()
if [[ -n "$AGENTX_CONSTRAINTS" ]]; then
    [[ -r "$AGENTX_CONSTRAINTS" ]] || die "cannot read AGENTX_CONSTRAINTS=$AGENTX_CONSTRAINTS"
    # Absolute and mounted, so uv in the container sees the same file.
    AGENTX_CONSTRAINTS="$(realpath "$AGENTX_CONSTRAINTS")"
    constraint_mount=(-v "$AGENTX_CONSTRAINTS:$AGENTX_CONSTRAINTS:ro")
    echo "UV_CONSTRAINT=$AGENTX_CONSTRAINTS" >>"$out/runtime.env"
fi

echo "AgentX C$CONC, ${DURATION}s via $ROUTER_URL -> $out"
# InferenceX's write_agentic_result_json is replaced so aggregation goes
# through aggregate.py (all PD GPUs, also on a single node, see that file).
# shellcheck disable=SC2016  # expanded inside the container
docker run --rm --name "$CTR-agentx" --network host --ipc host --shm-size 32g \
    --user "$(id -u):$(id -g)" -v "$KIT:$KIT:ro" -v "$WORK_DIR:$WORK_DIR" \
    -v "$MODEL:$MODEL:ro" -v "$out/tmp:/ax-tmp" "${constraint_mount[@]}" --env-file "$out/runtime.env" "$IMAGE" \
    bash -c '
        set -euo pipefail
        out="$1" prepare_only="$2" aggregate="$3"
        source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
        write_agentic_result_json() {
            (cd "$INFMAX_CONTAINER_WORKSPACE" && RESULT_DIR="$1" "$AIPERF_PYTHON" "$aggregate")
            "$AIPERF_PYTHON" "$INFMAX_CONTAINER_WORKSPACE/utils/generate_aiperf_plots.py" "$1" 2>&1 || true
        }
        install_agentic_deps
        resolve_trace_source
        build_replay_cmd "$out"
        REPLAY_CMD+=" --apply-chat-template"
        printf "%s\n" "$REPLAY_CMD" >"$out/replay-command.txt"
        if [[ "$prepare_only" == 1 ]]; then
            echo "AgentX preparation complete; no requests sent."
            exit 0
        fi
        run_agentic_replay_and_write_outputs "$out"
    ' _ "$out" "$PREPARE_ONLY" "$KIT/engine/agentx/aggregate.py" 2>&1 | tee "$out/runner.log"
[[ "$PREPARE_ONLY" == 1 ]] || [[ -s "$out/agentx_conc$CONC.json" ]] ||
    die "no aggregate written: $out/agentx_conc$CONC.json"
