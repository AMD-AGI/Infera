#!/usr/bin/env bash
# Torch-profile the decode engine for SECONDS under the running load (the deployment needs D_ARGS with
# --profiler-config '{"profiler":"torch","torch_profiler_dir":"<dir>","torch_profiler_with_stack":false}').
#   trace_decode.sh DECODE_URL SECONDS
set -uo pipefail
url=$1 secs=${2:-3}
curl -sf -m 60 -X POST "$url/start_profile" && echo "start_profile $(date -u +%T)"
sleep "$secs"
curl -sf -m 600 -X POST "$url/stop_profile" && echo "stop_profile $(date -u +%T)"
