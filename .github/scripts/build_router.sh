#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build infera-router in the manylinux_2_28 image, so the binary needs nothing
# newer than glibc 2.28 and the system libstdc++ -- what the tag of the wheel
# that bundles it promises (bundle_router_wheel.sh checks it). Needs docker.
#   .github/scripts/build_router.sh <out-dir>      -> <out-dir>/infera-router
#
# The checkout is mounted read-only and cargo's target dir lives in the
# container: a local run must neither leave root-owned files behind nor reuse
# objects a newer host compiler built for a newer CPU (see rust/.cargo/config.toml).
set -euo pipefail

out="${1:?usage: build_router.sh <out-dir>}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
repo="$(git -C "$(dirname "$(readlink -f "$0")")" rev-parse --show-toplevel)"
image="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"

docker run --rm \
  -v "$repo:/src:ro" -v "$out:/out" \
  -e CARGO_TARGET_DIR=/tmp/target -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -w /src/rust "$image" bash -euo pipefail -c '
    # onig_sys generates its bindings with bindgen, which needs libclang.
    dnf install -y -q clang-devel
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
    . "$HOME/.cargo/env"
    # From rust/, not via --manifest-path: cargo finds rust/.cargo/config.toml
    # by working directory, and the zeromq-src flags it sets are not optional.
    cargo build --release --locked --bin infera-router
    install -m 0755 -o "$HOST_UID" -g "$HOST_GID" "$CARGO_TARGET_DIR/release/infera-router" /out/
  '
ls -la "$out/infera-router"
