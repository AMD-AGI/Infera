#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: per-node RDMA preflight — which registration mode (and so which cluster wrapper) this node
#       needs, and the transport values to paste into it.
# why : PD moves KV over the fabric on every request and every way that goes wrong is SILENT: a
#       container that sees no RDMA device falls back to a 5-20x slower transport, and a wrong
#       registration mode only shows as an OOM at KV-cache registration minutes into a cold start.
# how : on each node, with docker:   IMAGE=<engine image> bash preflight_rdma.sh
set -uo pipefail
: "${IMAGE:?set IMAGE=<the engine image from build_image.sh>}"
GID_INDEX="${MC_GID_INDEX:-1}"

echo "== $(hostname -s)"
v="$(cat /sys/module/amdgpu/version 2>/dev/null)"; maj=${v%%.*}; min=${v#*.}; min=${min%%.*}
peer="$(grep -oE '^(ib_peer_mem|amdp2p) ' /proc/modules 2>/dev/null | tr -d ' ' | paste -sd, -)"
echo "amdgpu ${v:-unknown}   peer-memory module: ${peer:-none}"
# amdgpu 6.16 charges a peer-memory pin once and a dma-buf pin twice; 6.19 charges both twice.
# A GEM BO is charged once, so it is the answer wherever peer memory is not (fully) usable.
if [[ "$maj" =~ ^[0-9]+$ && "$min" =~ ^[0-9]+$ ]] && ((maj > 6 || (maj == 6 && min >= 19))); then
  mode=gem; why="amdgpu >= 6.19 charges peer-memory and dma-buf pins twice"
elif [ -n "$peer" ]; then
  mode=peermem; why="amdgpu 6.16 with a peer-memory module: ibv_reg_mr is charged once"
else
  mode=gem; why="no peer-memory module: dma-buf is the only path, and only a GEM BO is charged once there"
fi

devs=() nics=()
for d in $(ls /sys/class/infiniband 2>/dev/null | sort -V); do
  grep -q ACTIVE "/sys/class/infiniband/$d/ports/1/state" 2>/dev/null || continue
  gid="$(cat "/sys/class/infiniband/$d/ports/1/gids/$GID_INDEX" 2>/dev/null)"
  printf '  %-10s ACTIVE  gid[%s]=%s  numa=%s\n' "$d" "$GID_INDEX" "${gid:-?}" \
    "$(cat "/sys/class/infiniband/$d/device/numa_node" 2>/dev/null)"
  devs+=("$d")
done
[ ${#devs[@]} -gt 0 ] || echo "  NO ACTIVE RDMA PORT on the host"
n_ctr=$(docker run --rm --entrypoint bash --device=/dev/infiniband "$IMAGE" -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE; true')
echo "RDMA ports usable INSIDE the image: ${n_ctr:-0} (host: ${#devs[@]})"
[ "${n_ctr:-0}" -gt 0 ] || echo "  !! the image's RDMA provider cannot drive this NIC — KV would not move over RDMA"
echo
echo "=> use cluster/cluster.$mode.sh   ($why)"
echo "   export RDMA_IB_DEVICES=$(IFS=,; echo "${devs[*]}")"
echo "   export MC_GID_INDEX=$GID_INDEX   (the gid above must be an IPv4-mapped ::ffff: address, not fe80::)"
echo "   RDMA_GPU_NICS defaults to RDMA_IB_DEVICES: position k must be the NIC on GPU k's PCIe switch."
