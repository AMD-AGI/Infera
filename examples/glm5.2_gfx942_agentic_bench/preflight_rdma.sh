#!/usr/bin/env bash
# RDMA preflight. Run on each host before PD bring-up. If DUMP_PATH is set and
# one task runs per node, it also runs infera's cross-node fabric checks.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

echo "[preflight] active RDMA ports visible in $IMAGE (expect non-zero):"
IONIC_ARGS=()
if [[ -e "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}" ]]; then
  IONIC_ARGS=(-v "${HOST_LIBIONIC:-/usr/lib/x86_64-linux-gnu/libionic.so}:/host-libionic/libionic.so:ro")
fi
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
  "${IONIC_ARGS[@]}" "$IMAGE" bash -lc "ibv_devinfo | grep -c PORT_ACTIVE"

if [[ -z "${DUMP_PATH:-}" ]]; then
  echo "[preflight] DUMP_PATH not set; skipping cross-node netperf/mooncake checks."
  exit 0
fi

: "${SLURM_PROCID:?set SLURM_PROCID=<rank> or launch with srun}"
: "${SLURM_NNODES:?set SLURM_NNODES=<node count> or launch with srun}"
export SLURMD_NODENAME="${SLURMD_NODENAME:-$(hostname)}"

docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
  -e MC_GID_INDEX -e IB_DEVICE -e SLURM_PROCID -e SLURM_NNODES -e SLURMD_NODENAME \
  "${IONIC_ARGS[@]}" -v "$DUMP_PATH:$DUMP_PATH" "$IMAGE" \
  python -m infera.tools.preflight --dump-path "$DUMP_PATH" --netperf --mooncake
