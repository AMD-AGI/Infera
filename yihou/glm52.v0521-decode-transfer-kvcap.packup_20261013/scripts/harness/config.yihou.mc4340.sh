#!/usr/bin/env bash
# Overlay: run any v0521 E-config on infera-sglang:v0521-yihou-1010-mc4340
# (v0521 recipe + Mooncake #4340 HIP IPC sub-allocation relocation fix, faae8dd4 port)
# and add the YIHOU-REGBASE registration diagnostic mount.
#
# Use:  export CONFIG=$H/config.yihou.mc4340.sh
#       export MC4340_BASE_CONFIG=$H/config.yihou.E2b.sh   # any existing E-config (default E2-target)
#       export YIHOU_NODE=crsuse2-m2m-102|crsuse2-m2m-136  # picks the per-node image ID
#
# WIRE FORMAT: #4340 publishes a 72-byte IPC payload (handle + offset) for buffers that do
# not start their allocation. An old (1009) importer REJECTS it; an old exporter omits the
# offset and a new importer then relocates the old, wrong way. So BOTH legs must run the
# mc4340 build. engine.sh uses one $IMAGE for prefill and decode, so a single run is
# consistent by construction; never mix 1009 and mc4340 legs (e.g. a leg left running
# from an earlier round, or a cross-node pair with different images).
#
# The loaded engine.so is identical on both nodes:
#   sha256 3c96489a0ca4c8662a5f6512c1d9d44653835afc3d7cfaa9082b70f575bb289a
# (1009 image: f22b1364...). Image IDs differ per node (separate builds).

# HARD assignments (not :=) so an env file exporting the 1009 ID cannot win.
IMAGE=infera-sglang:v0521-yihou-1010-mc4340
case "${YIHOU_NODE:-}" in
    crsuse2-m2m-102) EXPECTED_IMAGE_ID=sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2 ;;
    crsuse2-m2m-136) EXPECTED_IMAGE_ID=sha256:9872bbdabfcb8f6676b04378d8ea75c922dfc8db6a8c5c87acd801e205b3b575 ;;
    *) echo "config.yihou.mc4340.sh: YIHOU_NODE='${YIHOU_NODE:-}' has no mc4340 image ID (built on 102 and 136 only)" >&2
       return 2 2>/dev/null || exit 2 ;;
esac
export IMAGE EXPECTED_IMAGE_ID

_mc4340_base="${MC4340_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.E2-target.sh}"
[[ -r "$_mc4340_base" ]] || { echo "config.yihou.mc4340.sh: MC4340_BASE_CONFIG not readable: $_mc4340_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_mc4340_base"

# After the base chain (v0521-base assigns YIHOU_MOUNTS unconditionally): v2 mounts + REGBASE.
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-regbase.tsv"
# The base chain must not have changed the image.
[[ "$IMAGE" == infera-sglang:v0521-yihou-1010-mc4340 ]] ||
    { echo "config.yihou.mc4340.sh: base config overrode IMAGE=$IMAGE" >&2; return 2 2>/dev/null || exit 2; }
YIHOU_EXP="${YIHOU_EXP:-E}+mc4340"
unset _mc4340_base
