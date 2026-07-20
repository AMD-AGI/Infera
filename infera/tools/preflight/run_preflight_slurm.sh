#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# One-command SLURM preflight: one task per node runs preflight inside the engine
# image; rank 0 renders the combined report. Nodes must already be in a SLURM
# allocation.
#
# Private image: export DOCKER_TOKEN at runtime to auto `docker login` on every
# node (never commit it); leave unset if the image is already present locally.
#
# Example:
#   DOCKER_TOKEN="$DOCKERHUB_TOKEN" ./run_preflight_slurm.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- config ---
NODES=${NODES:-}          # comma-separated SLURM node list, e.g. node1,node2 (required)
PARTITION=${PARTITION:-}  # SLURM partition name (required)
IMAGE=${IMAGE:-}          # engine image, e.g. inferaimage/infera:<tag> (required)
DOCKER_REGISTRY=${DOCKER_REGISTRY:-docker.io}
DOCKER_USER=${DOCKER_USER:-inferaimage}
# Local NVMe mount for the storage throughput / NVMe↔HBM test — bind-mounted into
# the container and pointed at via INFERA_PREFLIGHT_STORAGE_PATH. Must be a real
# mount point (a bare local NVMe filesystem). Set empty to skip (the probe then
# auto-picks, but a container sees no local NVMe mount).
STORAGE_PATH=${STORAGE_PATH:-/mnt/nvmeraid}

# NODES, PARTITION and IMAGE are site-specific and have no sensible default.
if [ -z "$NODES" ] || [ -z "$PARTITION" ] || [ -z "$IMAGE" ]; then
  echo "[preflight] ERROR: NODES, PARTITION and IMAGE are required" \
       "(e.g. NODES=node1,node2 PARTITION=k8s IMAGE=inferaimage/infera:<tag> ./run_preflight_slurm.sh)" >&2
  exit 1
fi

# --- derived ---
# Host libionic (RoCE): mounted into the container so its libibverbs matches the
# host ionic kernel ABI (needed by ib_write_bw); docker resolves the symlink.
LIBIONIC=${LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}
DUMP_PATH=${DUMP_PATH:-$HERE/preflight_result}
REPO=${REPO:-$(cd "$HERE/../../.." && pwd)}
N=$(echo "$NODES" | tr ',' '\n' | grep -c .)

echo "##### Infera Preflight #####"
echo "[preflight] partition=$PARTITION nodes=$NODES ($N)"
echo "[preflight] image=$IMAGE"
echo "[preflight] report dir: $DUMP_PATH"
rm -rf "$DUMP_PATH"; mkdir -p "$DUMP_PATH"

# Auto docker login on every node when a token is provided at runtime.
if [ -n "${DOCKER_TOKEN:-}" ]; then
  export DOCKER_TOKEN
  echo "[preflight] docker login as '$DOCKER_USER' @ $DOCKER_REGISTRY on $N node(s)..."
  srun --partition="$PARTITION" --nodelist="$NODES" -N"$N" --ntasks-per-node=1 \
    bash -c "printf %s \"\$DOCKER_TOKEN\" | docker login -u $DOCKER_USER --password-stdin $DOCKER_REGISTRY"
else
  echo "[preflight] DOCKER_TOKEN unset -> skip login (assuming image already present)"
fi

# Pre-pull the image on every node first, so the timed run starts in sync. If one
# node pulled 75GB while another already ran, the multi-node barriers would time
# out and report spurious failures.
echo "[preflight] pulling image on $N node(s)..."
srun --partition="$PARTITION" --nodelist="$NODES" -N"$N" --ntasks-per-node=1 \
  docker pull "$IMAGE"

echo "[preflight] launching preflight on $N node(s)..."
# Per-node run command. STORAGE_PATH is mounted + pointed at ONLY when it is a real
# mount point ON THAT NODE (the `mountpoint` test runs remotely). A plain
# `docker -v` on a missing path auto-creates an empty dir; testing that dir would
# silently measure the overlay fs instead of local NVMe — and the leftover empty
# dir would then fool a bare `-d` test, so we require an actual mount. Static
# values are interpolated here; the check and ${storage[@]} stay literal for the node.
RUN_SCRIPT=$(cat <<EOF
storage=()
if [ -n "$STORAGE_PATH" ] && mountpoint -q "$STORAGE_PATH"; then
  storage=(-v "$STORAGE_PATH:$STORAGE_PATH" -e "INFERA_PREFLIGHT_STORAGE_PATH=$STORAGE_PATH")
fi
# Keep the image ENTRYPOINT (infera-inject-host-ionic) so it swaps in the host's
# libionic; without it in-container ib_write_bw can't see the ionic RDMA devices.
exec docker run --rm --privileged --ipc host --network host \\
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \\
  -v /boot:/boot:ro \\
  -v "$REPO:$REPO" -w "$REPO" \\
  -v "$LIBIONIC:/host-libionic/libionic.so" \\
  "\${storage[@]}" \\
  -e SLURM_PROCID -e SLURM_NNODES -e SLURMD_NODENAME \\
  -e "PREFLIGHT_IMAGE=$IMAGE" \\
  "$IMAGE" bash -lc "python3 -m infera.tools.preflight --dump-path $DUMP_PATH"
EOF
)
# Preflight exits non-zero when a check FAILs; keep that code but still print the
# report path (set -e would otherwise abort before the final echo).
rc=0
srun --partition="$PARTITION" --nodelist="$NODES" -N"$N" --ntasks-per-node=1 \
  bash -c "$RUN_SCRIPT" || rc=$?

echo "[preflight] done (exit=$rc). report: $DUMP_PATH/infera_preflight_report.html"
exit "$rc"
