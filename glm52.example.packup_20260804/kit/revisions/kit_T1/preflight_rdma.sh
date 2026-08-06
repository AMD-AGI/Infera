#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: run infera's RDMA preflight inside the engine container. Two subcommands:
#         mode    per-node registration-mode probe -> tells you WHICH cluster wrapper to use
#         fabric  cross-node RoCE bandwidth + mooncake KV transfer measured over rdma AND tcp
# why : PD moves the KV cache over the fabric on every request, and every way it can go wrong
#       is SILENT. A container that cannot see the RDMA devices does not raise — mooncake
#       falls back to a transport that works and is merely 5-20x slower. This turns that into
#       a number you can read before you deploy.
# how : both probes must run INSIDE the engine container: the mooncake engine library, the
#       GPU-side checks and ib_write_bw only exist there.
#
#   IMAGE=<engine image> bash preflight_rdma.sh mode
#   IMAGE=<engine image> DUMP_PATH=<shared dir> srun -N2 --ntasks-per-node=1 \
#       bash preflight_rdma.sh fabric
set -euo pipefail

: "${IMAGE:?set IMAGE=<the infera-sglang engine image>}"
CMD="${1:-mode}"

# Do NOT pass --entrypoint '' here if your image's entrypoint bind-maps a host RDMA provider
# library: skipping it leaves the container's provider mismatched with the host driver, the
# device count reads 0, and the report is measuring the wrong thing.
DOCKER_ARGS=(--rm --network host
             --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband
             --group-add video --group-add render --cap-add=IPC_LOCK
             --ulimit memlock=-1:-1 --security-opt seccomp=unconfined --ipc=host)
[ -n "${HOST_RDMA_LIB:-}" ] && DOCKER_ARGS+=(-v "$HOST_RDMA_LIB:/host-rdma/$(basename "$HOST_RDMA_LIB"):ro")

case "$CMD" in
mode)
  # Per-node. Reports, per RDMA NIC: vendor, link speed, ODP, PCI BDF, NUMA node and GID
  # index; and node-wide: whether a peer-memory module is loaded and whether the image's
  # mooncake engine has dma-buf compiled in. Then it enumerates registration modes A/B/C,
  # each marked viable or blocked WITH the reason, and prints the exact env + launch flags
  # for the one it picks.
  #
  # Read the peer-mem line first — it decides which cluster wrapper you use:
  #     peermem present -> mode A -> cluster/cluster.peermem.sh
  #     peermem absent, an ODP NIC exists -> mode B -> cluster/cluster.dmabuf.sh
  #
  # Exit code 0 = a full-KV mode is viable. Exit code 2 = the only viable path needs the KV
  # pool CAPPED (mode C) or nothing is viable — a human decision, not a default.
  echo "[preflight] registration-mode probe on $(hostname -s)"
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
    python3 -m infera.tools.preflight.mooncake_mode
  ;;
fabric)
  # Cross-node. Needs one task per node writing into a SHARED dump path; srun fills in
  # SLURM_PROCID / SLURM_NNODES / SLURMD_NODENAME. Rank 0 renders the combined HTML report
  # once every node has written its JSON.
  #
  # The rows to read for PD are the mooncake ones: they report KV-move bandwidth over rdma
  # and over tcp SEPARATELY. A fabric that will silently serve at TCP speed therefore shows
  # up as a number here, instead of as a deployment that is inexplicably slow later.
  : "${DUMP_PATH:?set DUMP_PATH=<a directory shared by all nodes>}"
  : "${SLURM_PROCID:?rank — srun sets this; otherwise set it per node by hand (0..N-1)}"
  : "${SLURM_NNODES:?node count — srun sets this}"
  export SLURMD_NODENAME="${SLURMD_NODENAME:-$(hostname -s)}"
  # Use a FRESH dump path each run: rank 0 decides everyone has reported by counting the
  # *.json files in the directory, so a stale file from a previous run makes it render early.
  echo "[preflight] fabric check: rank $SLURM_PROCID/$SLURM_NNODES on $SLURMD_NODENAME -> $DUMP_PATH"
  exec docker run "${DOCKER_ARGS[@]}" \
    -e SLURM_PROCID -e SLURM_NNODES -e SLURMD_NODENAME \
    -e PREFLIGHT_HOST="$SLURMD_NODENAME" -e PREFLIGHT_IMAGE="$IMAGE" \
    -v "$DUMP_PATH:$DUMP_PATH" "$IMAGE" \
    python3 -m infera.tools.preflight --dump-path "$DUMP_PATH" --netperf --mooncake
  ;;
*)
  echo "usage: IMAGE=<image> bash preflight_rdma.sh {mode|fabric}" >&2
  exit 1
  ;;
esac
