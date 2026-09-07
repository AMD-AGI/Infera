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
# HOST_RDMA_MOUNT must be the in-container path your image's entrypoint reads, or the
# injection silently no-ops and this probe measures a container whose provider does not
# match the host driver — it then reports zero devices and the report means nothing.
[ -n "${HOST_RDMA_LIB:-}" ] && DOCKER_ARGS+=(-v "$HOST_RDMA_LIB:${HOST_RDMA_MOUNT:-/host-rdma/$(basename "$HOST_RDMA_LIB")}:ro")
for v in INFERA_PREFLIGHT_RDMA_DEVICE INFERA_PREFLIGHT_GID_INDEX INFERA_PREFLIGHT_KV_GPUS; do
  [ -n "${!v:-}" ] && DOCKER_ARGS+=(-e "$v")
done

case "$CMD" in
mode)
  # Per-node registration-mode probe; prints the env and launch flags for the mode it picks.
  # Read the peer-mem line first — present -> cluster.peermem.sh, absent+ODP -> dmabuf.sh.
  # Exit 2 means only the KV-pool-CAPPED mode C is left: a human decision, not a default.
  echo "[preflight] registration-mode probe on $(hostname -s)"
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
    python3 -m infera.tools.preflight.mooncake_mode
  ;;
fabric)
  # Cross-node. One task per node writing into a SHARED dump path; rank 0 renders the report.
  # For PD, read the mooncake rows: they report KV-move bandwidth over rdma and over tcp
  # SEPARATELY, which turns a silent TCP fallback into a number you can read before deploying.
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
