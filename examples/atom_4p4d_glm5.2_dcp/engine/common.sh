#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Sourced by every script in engine/, never run directly: loads ../env.sh,
# rejects unfilled placeholders and defines the shared helpers. The caller
# sets KIT, the kit root.
# shellcheck disable=SC2034  # the scripts that source this file use the variables
set -euo pipefail
# shellcheck source=../env.sh
source "$KIT/env.sh"

die() { echo "$*" >&2; exit 1; }

for var in NODE_IP MODEL IMAGE WORK_DIR; do
    case "${!var}" in "<"*) die "edit env.sh: $var is still a placeholder" ;; esac
done
ROUTER_URL="http://$NODE_IP:$ROUTER_PORT"

# Every container this kit may own, named explicitly. Never a pattern: on a
# shared node a pattern removes somebody else's containers.
CONTAINERS=("$CTR-agentx" "$CTR-eval" "$CTR-router" "$CTR-decode" "$CTR-prefill" "$CTR-etcd")

# Settings up.sh records per run, so later scripts report what actually runs.
RUN_VARS=(CONC MTP_K MTP_AL PREFILL_GPUS DECODE_GPUS PREFILL_MEM_UTIL DECODE_MEM_UTIL
    PREFILL_OFFLOAD_GB MOONCAKE_DISABLE_HIP_DMABUF SESSION_AFFINITY PREFILL_EXTRA_ARGS DECODE_EXTRA_ARGS
    ROUTER_EXTRA_ARGS IMAGE MODEL)

# resolve_model: check MODEL exists here and bind its resolved path; a symlink
# across an NFS mount boundary binds an empty directory into the container.
# Only scripts that mount MODEL call it, so down.sh works without the weights.
resolve_model() {
    [[ -d "$MODEL" ]] || die "MODEL is not a directory on this host: $MODEL"
    MODEL="$(realpath "$MODEL")"
}

# load_run: set RUN_DIR to the run up.sh started last and load its settings.
load_run() {
    [[ -f "$WORK_DIR/current_run" ]] || die "no run found: bash engine/up.sh first"
    RUN_DIR="$(cat "$WORK_DIR/current_run")"
    # shellcheck source=/dev/null
    source "$RUN_DIR/engine-config.env"
}

# require_real_acceptance: refuse content checks when decode forces MTP acceptance.
require_real_acceptance() {
    [[ -z "$MTP_AL" ]] ||
        die "the engines run with forced MTP acceptance ($MTP_AL); restart with: MTP_AL= bash engine/up.sh"
}

# wait_health URL CONTAINER...: poll URL until it answers; fail if a CONTAINER stops.
wait_health() {
    local url="$1" deadline=$((SECONDS + READY_TIMEOUT)) c
    shift
    until curl -sf -m 5 -o /dev/null "$url"; do
        for c in "$@"; do
            [[ "$(docker inspect --type container -f '{{.State.Running}}' "$c" 2>/dev/null)" == true ]] ||
                die "$c stopped while waiting for $url; see: docker logs $c"
        done
        ((SECONDS < deadline)) || die "timeout after ${READY_TIMEOUT}s waiting for $url"
        sleep 10
    done
}

# wait_vram_free: wait until every GPU of this node holds < 4 GiB of VRAM. The
# kit owns all eight GPUs, and `docker rm` returns before the driver reclaims.
wait_vram_free() {
    local deadline=$((SECONDS + VRAM_RELEASE_TIMEOUT)) used
    while used="$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1)"
        ((${used:-0} >= 4 << 30)); do
        ((SECONDS < deadline)) || die "VRAM not released after ${VRAM_RELEASE_TIMEOUT}s: $((used >> 30)) GiB in use"
        sleep 10
    done
}
