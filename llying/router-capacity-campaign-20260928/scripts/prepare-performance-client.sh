#!/usr/bin/env bash
set -euo pipefail
set -a; source "${CONFIG:?set CONFIG}"; set +a
docker run --rm --name llying-campaign-client-prepare --network host -v /perf_apps:/perf_apps \
 -e "AIPERF_RUNTIME_DIR=$AGENTX_CACHE_DIR/aiperf-$RUN_ID-c80" \
 -e "UV_PYTHON_INSTALL_DIR=/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924/cache/python" \
 -e "AIPERF_UV_CACHE_DIR=$AGENTX_CACHE_DIR/aiperf-r1r4-31719-client-smoke/uv-cache" \
 -e "INFMAX_CONTAINER_WORKSPACE=$INFERENCEX_DIR" -e "UV_CONSTRAINT=$AGENTX_CLIENT_CONSTRAINTS" \
 "$IMAGE" bash -c 'set -euo pipefail; source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"; install_agentic_deps; "$AIPERF_PYTHON" -m aiperf --version'
