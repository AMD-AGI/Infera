#!/usr/bin/env bash
# Resolve GLM-5.3 image knobs from GPU_PLATFORM.
# mi355x (default): gfx950 Mooncake, libionic ABI 4, re-anchor disagg patches.
# mi325x: gfx942 Mooncake, skip libionic (bnxt hosts), apply disagg patches as-is.
set -euo pipefail

p="$(printf '%s' "${GPU_PLATFORM:-mi355x}" | tr '[:upper:]' '[:lower:]')"
case "${p}" in
    mi325x|gfx942) p=mi325x ;;
    mi355x|gfx950|"") p=mi355x ;;
    *)
        echo "ERROR: GPU_PLATFORM=${GPU_PLATFORM} is not mi355x or mi325x" >&2
        exit 1
        ;;
esac
GPU_PLATFORM="${p}"

if [ "${p}" = mi325x ]; then
    [ -n "${MC_GPU_ARCH:-}" ] || MC_GPU_ARCH=gfx942
    [ -n "${INSTALL_LIBIONIC:-}" ] || INSTALL_LIBIONIC=0
    [ -n "${REANCHOR_SGLANG_DISAGG:-}" ] || REANCHOR_SGLANG_DISAGG=0
    LIBIONIC_REQUIRE_ABI="${LIBIONIC_REQUIRE_ABI:-}"
else
    [ -n "${MC_GPU_ARCH:-}" ] || MC_GPU_ARCH=gfx950
    [ -n "${INSTALL_LIBIONIC:-}" ] || INSTALL_LIBIONIC=1
    [ -n "${REANCHOR_SGLANG_DISAGG:-}" ] || REANCHOR_SGLANG_DISAGG=1
    [ -n "${LIBIONIC_REQUIRE_ABI:-}" ] || LIBIONIC_REQUIRE_ABI=4
fi

export GPU_PLATFORM MC_GPU_ARCH INSTALL_LIBIONIC LIBIONIC_REQUIRE_ABI REANCHOR_SGLANG_DISAGG
echo "[glm53-platform] GPU_PLATFORM=${GPU_PLATFORM} MC_GPU_ARCH=${MC_GPU_ARCH} INSTALL_LIBIONIC=${INSTALL_LIBIONIC} LIBIONIC_REQUIRE_ABI=${LIBIONIC_REQUIRE_ABI:-} REANCHOR_SGLANG_DISAGG=${REANCHOR_SGLANG_DISAGG}"
