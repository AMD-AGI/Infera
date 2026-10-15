#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# prefill1 on the second node, driven from the head node: the 4P4D kit's
# worker.sh and down.sh with docker pointed at PREFILL1_HOST over ssh. The
# 4P4D up.sh and down.sh call it through REMOTE_PREFILLS (env.sh).
#
# Usage: bash engine/prefill1.sh check|up|down
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../../atom_4p4d_glm5.2_dcp/engine/common.sh
source "$KIT/../atom_4p4d_glm5.2_dcp/engine/common.sh"

for var in PREFILL1_HOST PREFILL1_IP; do
    case "${!var}" in "<"*) die "edit env.sh: $var is still a placeholder" ;; esac
done

# Same recipe as prefill0; only the node, its GPUs and the container names
# differ. etcd stays on the head node, and the prefill is the only container
# here.
export DOCKER_HOST="ssh://$PREFILL1_HOST" ETCD_IP="$NODE_IP" NODE_IP="$PREFILL1_IP" \
    PREFILL_GPUS="$PREFILL1_GPUS" KIT_GPUS="$PREFILL1_GPUS" KIT_ROLES=prefill CTR="$CTR-p1" \
    REMOTE_PREFILLS=

case "${1:-}" in
    check)
        [[ "$PREFILL1_GPUS" =~ ^[0-9]+(,[0-9]+){3}$ ]] || die "prefill1 needs exactly four GPUs, got $PREFILL1_GPUS"
        head_image="$(env -u DOCKER_HOST docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
        image="$(docker image inspect -f '{{.Id}}' "$IMAGE")" || die "no $IMAGE on $PREFILL1_HOST"
        [[ "$image" == "$head_image" ]] ||
            die "$IMAGE differs between the nodes (${head_image:-missing} here, $image on $PREFILL1_HOST)"
        if docker inspect --type container "$CTR-prefill" >/dev/null 2>&1; then
            die "$CTR-prefill already exists on $PREFILL1_HOST: bash engine/down.sh first"
        fi
        ;;
    up)
        echo "prefill1: waiting for free VRAM on $PREFILL1_HOST GPUs $KIT_GPUS"
        wait_vram_free
        bash "$BASE_KIT/engine/worker.sh" prefill
        wait_health "http://$NODE_IP:$PREFILL_PORT/health" "$CTR-prefill"
        ;;
    down) bash "$BASE_KIT/engine/down.sh" ;;
    *) die "usage: bash engine/prefill1.sh check|up|down" ;;
esac
