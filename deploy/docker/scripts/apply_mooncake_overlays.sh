#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Apply ordered overlays to a fresh checkout; errors stop the build.
set -euo pipefail

root="${1:?usage: apply_mooncake_overlays.sh MOONCAKE_SOURCE}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patches=()
case "${APPLY_MOONCAKE_RDMA_PATCHES:-0}" in
    0) ;;
    1)
        pinned=faae8dd4a6309c3ecd47e0721a83b0250d686fa2
        if [[ "$(git -C "$root" rev-parse HEAD)" != "$pinned" ]]; then
            echo "[mc-build] bundled RDMA patches require Mooncake $pinned" >&2
            exit 1
        fi
        patches+=("$here/../patches/mooncake_rdma/destination-local-rail.patch"
                  "$here/../patches/mooncake_rdma/ack-timeout.patch")
        ;;
    *) echo "[mc-build] APPLY_MOONCAKE_RDMA_PATCHES must be 0 or 1" >&2; exit 1 ;;
esac
# No eval or glob expansion. Paths are absolute and whitespace-free.
extra="${MOONCAKE_PATCHES:-}"
if [[ "$extra" == *$'\n'* || "$extra" == *$'\r'* ]]; then
    echo "[mc-build] MOONCAKE_PATCHES must be a single whitespace-separated line" >&2
    exit 1
fi
read -r -a custom <<< "$extra"
patches+=("${custom[@]}")
manifest="$root/infera-build.txt"
printf 'revision %s\n' "$(git -C "$root" rev-parse HEAD)" > "$manifest"
for patch in "${patches[@]}"; do
    if [[ "$patch" != /* || ! -f "$patch" ]]; then
        echo "[mc-build] patch must be an existing absolute file: $patch" >&2
        exit 1
    fi
    git -C "$root" apply --check -- "$patch"
    git -C "$root" apply -- "$patch"
    printf 'patch %s\n' "$(sha256sum -- "$patch")" | tee -a "$manifest"
done
