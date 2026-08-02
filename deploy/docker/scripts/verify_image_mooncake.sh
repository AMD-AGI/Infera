#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Verify, AGAINST A BUILT IMAGE, that the Mooncake engine.so it will actually
# load carries the HIP-transport gate.
#
# WHY THIS EXISTS (and why the in-build check was not enough):
#   build_mooncake_sglang.sh already self-verifies the gate and `exit 1`s if the
#   rebuild did not take. That check can only fire if the build STEP RUNS. Both
#   rocm/infera:sglang-v0.1.1 and :sglang-v0.1.2 shipped an UNPATCHED engine.so
#   anyway, because the step was never in their build at all:
#     - v0.1.1 (built 2026-07-22) predates the Dockerfile stage that invokes the
#       script (added 2026-07-27, a546137c);
#     - v0.1.2 (built 2026-07-31) is v0.1.1 + exactly ONE appended layer (the
#       libionic deb) — i.e. it was built FROM the older image, not from
#       deploy/docker/Dockerfile.sglang, so the stage never re-ran.
#   A build-step assertion cannot catch "the build step is not in this image".
#   Only an assertion on the finished ARTIFACT can. That is this script.
#
# It is deliberately image-level and dependency-free: give it any image ref
# (local or registry) and it inspects the installed mooncake package the same
# way an engine would resolve it at runtime.
#
# USAGE:
#   deploy/docker/scripts/verify_image_mooncake.sh <image> [<image> ...]
#
# EXIT: 0 = every image carries the gate; 1 = at least one does not.
set -uo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <image> [<image> ...]" >&2
    exit 2
fi

# Runs inside the image. Resolves the mooncake package the way Python would,
# then asserts BOTH gate spellings are present in the binary that gets loaded:
#   MC_ENABLE_HIP_TRANSPORT  — opt back in for intra-node-only P2P
#   MC_DISABLE_HIP_TRANSPORT — hard veto (what infera/rocm_rdma_env.py sets)
# A binary with neither is upstream #2682 stock: it installs the HIP transport
# unconditionally and prefers it over RDMA, so cross-node PD cannot work.
read -r -d '' PROBE <<'SH' || true
set -u
SO_DIR=$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))' 2>/dev/null)
if [ -z "${SO_DIR:-}" ]; then
    echo "MOONCAKE_ABSENT"
    exit 0
fi
SO=$(ls "$SO_DIR"/engine*.so 2>/dev/null | head -1)
if [ -z "${SO:-}" ]; then
    echo "ENGINE_SO_ABSENT $SO_DIR"
    exit 0
fi
echo "ENGINE_SO $SO"
for v in MC_ENABLE_HIP_TRANSPORT MC_DISABLE_HIP_TRANSPORT; do
    echo "COUNT $v $(strings "$SO" | grep -c "$v")"
done
SH

rc=0
for image in "$@"; do
    echo "=== $image"
    out="$(docker run --rm --entrypoint sh "$image" -c "$PROBE" 2>/dev/null)" || {
        echo "  ERROR: could not run $image" >&2
        rc=1
        continue
    }

    case "$out" in
        *MOONCAKE_ABSENT*)
            # Not every image bundles Mooncake (e.g. the router/server images).
            # Nothing to gate, so nothing to fail.
            echo "  SKIP: no mooncake package in this image"
            continue
            ;;
        *ENGINE_SO_ABSENT*)
            echo "  ERROR: mooncake package present but no engine*.so" >&2
            rc=1
            continue
            ;;
    esac

    so="$(printf '%s\n' "$out" | awk '/^ENGINE_SO /{print $2}')"
    enable="$(printf '%s\n' "$out" | awk '/^COUNT MC_ENABLE_HIP_TRANSPORT /{print $3}')"
    disable="$(printf '%s\n' "$out" | awk '/^COUNT MC_DISABLE_HIP_TRANSPORT /{print $3}')"
    echo "  engine.so                 : $so"
    echo "  MC_ENABLE_HIP_TRANSPORT   : ${enable:-0}"
    echo "  MC_DISABLE_HIP_TRANSPORT  : ${disable:-0}"

    if [ "${enable:-0}" != "0" ] && [ "${disable:-0}" != "0" ]; then
        echo "  OK: HIP-transport gate present (cross-node PD will use RDMA)"
    else
        echo "  FAIL: engine.so has NO HIP-transport gate — this is stock upstream" >&2
        echo "        #2682, which installs the HIP transport unconditionally and" >&2
        echo "        prefers it over RDMA. Cross-node PD will die in KV transfer" >&2
        echo "        with: hipIpcOpenMemHandle failed (201 - invalid device context)." >&2
        echo "        Rebuild with deploy/docker/Dockerfile.sglang (BUILD_MOONCAKE=1)" >&2
        echo "        FROM THE PINNED BASE — not FROM a previously published image." >&2
        rc=1
    fi
done

if [ "$rc" -ne 0 ]; then
    echo
    echo "verify_image_mooncake: FAILED" >&2
fi
exit "$rc"
