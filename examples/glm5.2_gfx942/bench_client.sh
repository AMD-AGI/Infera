#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Run the agentic benchmark from the HOST, in a throwaway client container.
#
#   bash bench_client.sh <tag> <router-url> [engine url to flush ...]
#
#   bash bench_client.sh k8s http://10.42.0.31:8000 \
#        http://10.115.43.101:30001 http://10.115.61.101:31501
#
# The docker arm does not need this -- it already has an engine container to
# `docker exec` into, so run run_agentic_trace.sh there directly. This exists for
# the k8s arm, and for benchmarking from a third machine, where there is no such
# container but the same client has to be used so that the tokenizer, dataset,
# concurrency limiter and scorer are not silently part of the comparison.
#
# The client wants the GPU device nodes even though it computes nothing: importing
# sglang pulls in aiter, which shells out to rocminfo and fails without them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

TAG="${1:?usage: bash bench_client.sh <tag> <router-url> [flush url ...]}"
ROUTER="${2:?router url, e.g. http://10.42.0.31:8000}"
shift 2

docker image inspect "$IMAGE" >/dev/null \
  || { echo "[bench-client] image missing: $IMAGE" >&2; exit 1; }

MODEL_ARGS=(-v "$MODEL:$MODEL:ro")
MODEL_REAL="$(readlink -f "$MODEL")"
if [[ "$MODEL_REAL" != "$MODEL" ]]; then
  HF_REPO="$MODEL_REAL"
  [[ "$HF_REPO" == */snapshots/* ]] && HF_REPO="${HF_REPO%/snapshots/*}"
  MODEL_ARGS=(-v "$(dirname "$MODEL"):$(dirname "$MODEL"):ro" -v "$HF_REPO:$HF_REPO:ro")
fi

DATA_ARGS=()
if [[ -d "$DATA_DIR" && "$(readlink -f "$DATA_DIR")" != "$(readlink -f "$REPO")"/* ]]; then
  DATA_ARGS=(-v "$DATA_DIR:$DATA_DIR")
fi

echo "[bench-client] $TAG -> $ROUTER (flush: ${*:-none})"

docker run --rm --network host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v "$REPO:$REPO" "${MODEL_ARGS[@]}" "${DATA_ARGS[@]}" \
  -e "ROUTER_URL=$ROUTER" -e "FLUSH_URLS=$*" \
  -e "MODEL=$MODEL" -e "DATA_DIR=$DATA_DIR" -e "TRACE=$TRACE" \
  -e "OUTPUT_LEN=$OUTPUT_LEN" -e "NUM_PROMPTS=${NUM_PROMPTS}" -e "CONC=${CONC}" \
  -e "PAGE_SIZE=${PAGE_SIZE:-}" -e "SERVED_MODEL=${SERVED_MODEL:-}" \
  -w "$HERE" --entrypoint bash \
  "$IMAGE" run_agentic_trace.sh "$TAG"
