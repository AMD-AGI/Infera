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

for var in PREFILL_NODE_IP DECODE_NODE_IP MODEL IMAGE WORK_DIR; do
    case "${!var}" in "<"*) die "edit env.sh: $var is still a placeholder" ;; esac
done
[[ "$PREFILL_WORKERS" == 1 || "$PREFILL_WORKERS" == 2 ]] || die "PREFILL_WORKERS must be 1 or 2"
ROUTER_URL="http://$PREFILL_NODE_IP:$ROUTER_PORT"
PREFILL_CONTAINERS=("$CTR-agentx" "$CTR-eval" "$CTR-router" "$CTR-prefill0" "$CTR-prefill1" "$CTR-etcd")
DECODE_CONTAINER="$CTR-decode"
# The env.sh settings of a deployment (not the benchmark ones): recorded per
# run, and forwarded to the decode node.
KIT_VARS=(PREFILL_NODE_IP DECODE_NODE_IP DECODE_SSH MODEL SERVED_MODEL IMAGE WORK_DIR
    RDMA_DEVICES MC_GID_INDEX MOONCAKE_DISABLE_HIP_DMABUF PREFILL_WORKERS PREFILL0_GPUS PREFILL1_GPUS DECODE_GPUS
    SIM_ACCEPT_LEN PREFILL_EXTRA_ARGS DECODE_EXTRA_ARGS ROUTER_EXTRA_ARGS
    ETCD_PORT ETCD_PEER_PORT ROUTER_PORT PREFILL_PORT DECODE_PORT BOOTSTRAP_PORT
    KV_EVENTS_PORT KV_SNAPSHOT_PORT ETCD_IMAGE CTR
    DECODE_ENV_EXTRA DECODE_MOUNT_EXTRA)   # yihou llrepro: decode-only hooks (KF kernels)

INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_REF="${INFERENCEX_REF:-918524ff94045b3f091115f1051c22a8588edf2b}"
IX="$WORK_DIR/cache/InferenceX"

# on_decode COMMAND: run one shell command line on the decode node.
on_decode() { ssh -o BatchMode=yes "$DECODE_SSH" "$1"; }

# run_dir: the directory up.sh created for the current deployment.
run_dir() {
    [[ -f "$WORK_DIR/current_run" ]] || die "no run found: bash engine/up.sh first"
    cat "$WORK_DIR/current_run"
}

# wait_health URL CONTAINER [decode]: poll URL until it answers; fail if
# CONTAINER (on this node, or on the decode node) stops.
wait_health() {
    local url="$1" container="$2" where="${3:-}" running
    until curl -sf -m 5 -o /dev/null "$url"; do
        if [[ "$where" == decode ]]; then
            running="$(on_decode "docker inspect -f '{{.State.Running}}' $container" 2>/dev/null || true)"
        else
            running="$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)"
        fi
        [[ "$running" == true ]] ||
            die "$container stopped while waiting for $url; see: ${where:+ssh $DECODE_SSH }docker logs $container"
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
