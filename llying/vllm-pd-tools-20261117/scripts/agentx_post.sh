#!/usr/bin/env bash
# Re-run InferenceX's result aggregation (utils.agentic.aggregation.process_agentic_result) on an AgentX output
# directory whose replay finished but whose aggregation failed (s4: KV_OFFLOAD_BACKEND_METADATA was missing), with
# the run's own runtime.env plus EXTRA_ENV (space-separated K=V).
#   agentx_post.sh OUT_DIR
set -euo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
out=$1
WORK=/mnt/m2m_nobackup/liyingli_vpd/work
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
extra=()
for kv in ${EXTRA_ENV:-}; do extra+=(-e "$kv"); done
docker run --rm --network host --entrypoint bash --user "$(id -u):$(id -g)" -v "$WORK:$WORK" -v "$R:$R" \
    --env-file "$out/runtime.env" "${extra[@]}" "$IMAGE" -c '
        set -euo pipefail
        cd "$INFMAX_CONTAINER_WORKSPACE"
        RESULT_DIR="$1" AGENTIC_OUTPUT_DIR="$1" "$AIPERF_RUNTIME_DIR/venv/bin/python" -m utils.agentic.aggregation.process_agentic_result
    ' _ "$out"
ls -la "$out"/agentx_conc*.json
