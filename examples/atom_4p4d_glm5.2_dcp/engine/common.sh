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

# Multi-node hooks, set by atom_8p4d_glm5.2_dcp; the defaults are this kit's
# one-node layout. ETCD_IP: where etcd runs. KIT_GPUS: the GPUs the kit owns on
# the docker host. KIT_ROLES: the roles of its containers there.
# REMOTE_PREFILLS: a script that starts (up) or removes (down) prefills on
# other nodes. PREFILL_NUM_WORKERS: prefills the router must list.
ETCD_IP="${ETCD_IP:-$NODE_IP}"
KIT_GPUS="${KIT_GPUS:-$PREFILL_GPUS,$DECODE_GPUS}"
KIT_ROLES="${KIT_ROLES:-agentx eval router decode prefill etcd}"
REMOTE_PREFILLS="${REMOTE_PREFILLS:-}"
PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-1}"

# Every container this kit may own, named explicitly. Never a pattern: on a
# shared node a pattern removes somebody else's containers.
CONTAINERS=()
for kit_role in $KIT_ROLES; do CONTAINERS+=("$CTR-$kit_role"); done

# Settings up.sh records per run, so later scripts report what actually runs.
RUN_VARS=(CONC MTP_K MTP_AL PREFILL_GPUS DECODE_GPUS PREFILL_NUM_WORKERS PREFILL_MEM_UTIL DECODE_MEM_UTIL
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

# on_docker_host CMD: run the shell command CMD on the node docker drives: this
# one, or the node of DOCKER_HOST=ssh://user@host.
on_docker_host() {
    case "${DOCKER_HOST:-}" in
        ssh://*) ssh -n "${DOCKER_HOST#ssh://}" "$1" ;;
        *) bash -c "$1" </dev/null ;;
    esac
}

# wait_vram_free: wait until each of KIT_GPUS holds < 4 GiB of VRAM on the
# docker host; `docker rm` returns before the driver reclaims. A failed read
# (an ssh hiccup) counts as not yet free.
wait_vram_free() {
    local deadline=$((SECONDS + VRAM_RELEASE_TIMEOUT)) used gpu vram out state
    # VRAM use per GPU in HIP device order: ROCr numbers the GPU nodes of the
    # KFD topology in node order (CPU nodes have render minor 0).
    # shellcheck disable=SC2016  # expanded on the docker host
    local read_vram='for node in $(ls -v /sys/class/kfd/kfd/topology/nodes); do
        minor=$(sed -n "s/^drm_render_minor //p" /sys/class/kfd/kfd/topology/nodes/$node/properties)
        if [ "${minor:-0}" -gt 0 ]; then cat /sys/class/drm/renderD$minor/device/mem_info_vram_used; fi
    done'
    while :; do
        if out="$(on_docker_host "$read_vram")"; then
            mapfile -t vram <<<"$out"
            used=0
            for gpu in ${KIT_GPUS//,/ }; do
                [[ -n "${vram[gpu]:-}" ]] || die "no GPU $gpu on the docker host"
                ((vram[gpu] <= used)) || used="${vram[gpu]}"
            done
            ((used >= 4 << 30)) || return 0
            state="$((used >> 30)) GiB in use"
        else
            state="cannot read the VRAM use"
        fi
        ((SECONDS < deadline)) || die "VRAM not released after ${VRAM_RELEASE_TIMEOUT}s: $state"
        sleep 10
    done
}
