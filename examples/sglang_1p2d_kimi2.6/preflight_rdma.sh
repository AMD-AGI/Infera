#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# RDMA preflight, run before the PD bring-up. Two checks in one script:
#   (1) always: count the active RDMA ports the container can see on THIS node — must equal the
#       node's active ports (e.g. 8), not 0. A 0 means the container's ionic provider doesn't match
#       the host driver and RDMA would silently fall back to TCP (the libionic.so bind-mount added by
#       the image entrypoint keeps them aligned — do NOT run with `--entrypoint ""`).
#   (2) only if DUMP_PATH is set: cross-node RoCE bandwidth + Mooncake KV transfer over the fabric.
#       Needs one task per node writing to the shared DUMP_PATH; srun sets SLURM_PROCID/NNODES/NODENAME.
#
# Per-node device check only:
#   bash preflight_rdma.sh
# Full check via SLURM (one task per node):
#   export IMAGE=inferaimage/infera:<current-tag> DUMP_PATH=<shared-dir>
#   srun --nodelist=<node-0>,<node-1>,<node-2> -N3 --ntasks-per-node=1 bash preflight_rdma.sh
# Full check without SLURM (run on each node, rank by hand):
#   SLURM_PROCID=0 SLURM_NNODES=3 SLURMD_NODENAME=$(hostname) DUMP_PATH=<shared-dir> bash preflight_rdma.sh
set -euo pipefail

: "${IMAGE:?set IMAGE=inferaimage/infera:<current-tag> (infera-sglang image, tag handed per test round)}"
HOST_LIBIONIC="${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}"

echo "[preflight] active RDMA ports visible in the container (expect the node's port count, not 0):"
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
    -v "$HOST_LIBIONIC:/host-libionic/libionic.so:ro" \
    "$IMAGE" bash -lc "ibv_devinfo | grep -c PORT_ACTIVE"

# The cross-node fabric check only runs when a shared DUMP_PATH is given.
if [[ -z "${DUMP_PATH:-}" ]]; then
    echo "[preflight] DUMP_PATH not set — skipping the cross-node bandwidth + Mooncake check."
    echo "[preflight] to run it, set DUMP_PATH=<shared-dir> and launch one task per node (see header)."
    exit 0
fi

: "${SLURM_PROCID:?set SLURM_PROCID=<0..N-1> (srun sets it automatically; otherwise set it per node by hand)}"
: "${SLURM_NNODES:?set SLURM_NNODES=<node count, e.g. 3> (srun sets it automatically)}"
export SLURMD_NODENAME="${SLURMD_NODENAME:-$(hostname)}"

echo "[preflight] cross-node fabric check: rank $SLURM_PROCID/$SLURM_NNODES on $SLURMD_NODENAME -> $DUMP_PATH"
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
    -e MC_GID_INDEX=1 -e SLURM_PROCID -e SLURM_NNODES -e SLURMD_NODENAME \
    -v "$HOST_LIBIONIC:/host-libionic/libionic.so:ro" \
    -v "$DUMP_PATH:$DUMP_PATH" "$IMAGE" \
    python -m infera.tools.preflight --dump-path "$DUMP_PATH" --netperf --mooncake
