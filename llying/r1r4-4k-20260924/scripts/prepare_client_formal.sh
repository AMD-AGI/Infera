#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
if [[ "$SMOKE_ONLY" == 1 ]]; then
 client_runtime="$AGENTX_CACHE_DIR/aiperf-r1r4-31719-client-smoke"
else
 client_runtime="$AGENTX_CACHE_DIR/aiperf-$RUN_ID-c80"
fi
docker run --rm --name "$CONTAINER_PREFIX-client-prepare" --network host -v /perf_apps:/perf_apps \
 -e "AIPERF_UV_CACHE_DIR=$AGENTX_CACHE_DIR/aiperf-r1r4-31719-client-smoke/uv-cache" -e "AIPERF_RUNTIME_DIR=$client_runtime" -e "UV_PYTHON_INSTALL_DIR=$TRACE_RUNTIME/cache/python" \
 -e "INFMAX_CONTAINER_WORKSPACE=$INFERENCEX_DIR" -e "UV_CONSTRAINT=$AGENTX_CLIENT_CONSTRAINTS" \
 "$IMAGE" bash -c 'set -euo pipefail; source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"; install_agentic_deps; "$AIPERF_PYTHON" -m aiperf --version'
