#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# One-time setup, run on EVERY node. The platform hands us a live SGLang container
# (no docker CLI, no image build), so Infera is installed into it at runtime.
# Override: INFERA_ROOT=/path/to/Infera bash install.sh
set -euo pipefail

INFERA_ROOT="${INFERA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ETCD_VERSION="${ETCD_VERSION:-v3.5.17}"
ETCD_DIR="${ETCD_DIR:-/wekafs/llying/opt/etcd-${ETCD_VERSION}-linux-amd64}"

# The SGLang image already carries fastapi/httpx/pyzmq/msgspec/transformers, so
# install only the gaps with --no-deps: a full resolve would upgrade torch/sglang.
pip install --no-cache-dir msgpack nats-py cryptography httptools watchfiles websockets
pip install --no-cache-dir --no-deps -e "$INFERA_ROOT"
(cd /tmp && python3 -c "import infera, infera._version as v; print('infera', v.version, infera.__file__)")

# etcd is a plain static binary on the shared filesystem — one download per cluster.
if [[ ! -x "$ETCD_DIR/etcd" ]]; then
    mkdir -p "$(dirname "$ETCD_DIR")"
    curl -fsSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz" \
        | tar xz -C "$(dirname "$ETCD_DIR")"
fi
echo "etcd: $("$ETCD_DIR/etcd" --version | head -1)"

echo "[install] done — next: bash patch_mooncake_hip.sh && bash patch_sglang.sh, then bash preflight_rdma.sh"
