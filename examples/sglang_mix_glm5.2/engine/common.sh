#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Sourced by every script in engine/, never run directly. The caller sets KIT,
# the kit root.
# shellcheck disable=SC2034  # the scripts that source this file use the variables
set -euo pipefail
# shellcheck source=../env.sh
source "$KIT/env.sh"

die() { echo "$*" >&2; exit 1; }

for var in NODE_IP MODEL IMAGE WORK_DIR; do
    case "${!var}" in "<"*) die "edit env.sh: $var is still a placeholder" ;; esac
done
ROUTER_URL="http://$NODE_IP:$ROUTER_PORT"
ENGINE_URL="http://$NODE_IP:$ENGINE_PORT"
CONTAINERS=("$CTR-agentx" "$CTR-eval" "$CTR-router" "$CTR-engine" "$CTR-etcd")
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_REF="${INFERENCEX_REF:-918524ff94045b3f091115f1051c22a8588edf2b}"
IX="$WORK_DIR/cache/InferenceX"

# run_dir: the directory up.sh created for the current deployment.
run_dir() {
    [[ -f "$WORK_DIR/current_run" ]] || die "no run found: bash engine/up.sh first"
    cat "$WORK_DIR/current_run"
}

# wait_health URL CONTAINER: poll URL until it answers; fail if CONTAINER stops.
wait_health() {
    local url="$1" container="$2"
    until curl -sf -m 5 -o /dev/null "$url"; do
        [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" == true ]] ||
            die "$container stopped while waiting for $url; see: docker logs $container"
        sleep 10
    done
}

# inferencex_checkout: InferenceX at INFERENCEX_REF in $IX (AgentX and GSM8K).
inferencex_checkout() {
    if [[ ! -d "$IX/.git" ]]; then
        git clone --filter=blob:none "$INFERENCEX_REPO" "$IX"
        git -C "$IX" checkout --detach "$INFERENCEX_REF"
        git -C "$IX" submodule update --init --recursive
    fi
    [[ "$(git -C "$IX" rev-parse HEAD)" == "$INFERENCEX_REF" ]] ||
        die "InferenceX checkout $IX is not at $INFERENCEX_REF"
}
