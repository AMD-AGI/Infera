#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Gate Mooncake's HIP transport behind MC_ENABLE_HIP_TRANSPORT (default OFF), then
# rebuild the transfer engine IN PLACE against the sglang base image's own Mooncake
# checkout.
#
# WHY: the lmsysorg/sglang v0.5.15.post1 base bundles Mooncake at commit 01d1eb2a
# (upstream #2682 "Support rdma+hip multi-protocol segments for single-node
# disaggregation"), which installs a HIP transport UNCONDITIONALLY. It registers
# the KV pool under both "rdma" and "hip"; MultiTransport::selectTransport then
# prefers hip (priority 4) over rdma (priority 2) for EVERY transfer — including
# cross-node PD, where hipIpcOpenMemHandle can't open a peer's handle:
#   hip_transport.cpp:70 HipTransport: hipIpcOpenMemHandle failed (17 - invalid
#   device pointer) -> KVTransferError -> decode gets no KV -> 500.
# Older bases predated #2682 (no HIP transport at all), so cross-node PD used RDMA
# and worked. This restores that behavior: HIP off unless explicitly opted in for
# same-node P2P via MC_ENABLE_HIP_TRANSPORT=1.
#
# Idempotent + self-verifying: no-ops if already gated; fails the build if the
# rebuilt .so doesn't contain the gate.
set -euo pipefail

MC_ROOT="${MC_ROOT:-/sgl-workspace/Mooncake}"
SRC="$MC_ROOT/mooncake-transfer-engine/src/transfer_engine_impl.cpp"

[ -f "$SRC" ] || { echo "[mc-hip-gate] $SRC not found — base Mooncake layout changed; re-anchor this patch" >&2; exit 1; }

if grep -q 'MC_ENABLE_HIP_TRANSPORT' "$SRC"; then
    echo "[mc-hip-gate] already gated — skipping patch"
else
    echo "[mc-hip-gate] gating HIP transport install behind MC_ENABLE_HIP_TRANSPORT"
    python3 - "$SRC" <<'PY'
import sys
f = sys.argv[1]
s = open(f).read()
old = (
    "#ifdef USE_HIP\n"
    "        // HIP transport handles intra-node GPU P2P via XGMI/IPC and can\n"
    "        // coexist with the cross-node transport (RDMA/TCP) selected above.\n"
    "        {\n"
    "            Transport* hip_transport ="
)
new = (
    "#ifdef USE_HIP\n"
    "        // HIP transport handles intra-node GPU P2P via XGMI/IPC and can\n"
    "        // coexist with the cross-node transport (RDMA/TCP) selected above.\n"
    "        // FIX(infera): default OFF — upstream #2682 registers GPU bufs under a\n"
    "        // \"hip\" IPC segment that a CROSS-NODE peer cannot open, and\n"
    "        // selectTransport prefers hip over rdma; opt back in for same-node P2P\n"
    "        // with MC_ENABLE_HIP_TRANSPORT=1.\n"
    "        if (getenv(\"MC_ENABLE_HIP_TRANSPORT\")) {\n"
    "            Transport* hip_transport ="
)
if old not in s:
    sys.stderr.write("[mc-hip-gate] ERROR: anchor not found — Mooncake source changed; re-anchor\n")
    sys.exit(1)
assert s.count(old) == 1, "[mc-hip-gate] anchor not unique"
open(f, "w").write(s.replace(old, new, 1))
print("[mc-hip-gate] patched", f)
PY
fi

# ---- rebuild the transfer engine (USE_HIP=ON, matching the base's own build) ----
# `docker build` has NO GPU, so ROCm's amdgpu-arch probe fails ("Failed to get
# device count") and cmake would drop the HIP engine target from the build graph
# (leaving the base's unpatched .so). Pin the target arch explicitly so the build
# is GPU-independent. gfx950 = MI355X (this base's target).
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/opt/venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11:${CMAKE_PREFIX_PATH:-}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
export GPU_ARCHS="${GPU_ARCHS:-gfx950}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx950}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx950}"
cd "$MC_ROOT"
rm -rf build && mkdir build && cd build
echo "[mc-hip-gate] cmake configure"
cmake .. -DUSE_HIP=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
    -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_STORE_RUST=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja >/dev/null
# Build the python engine module explicitly. Plain `ninja` builds only the
# default target set, which does NOT include the pybind engine .so, so the
# prebuilt (unpatched) one would survive — target it by name.
echo "[mc-hip-gate] ninja build (engine module)"
ninja engine.cpython-310-x86_64-linux-gnu.so

# ---- install the rebuilt engine.so over the pip-installed one ----
SO="$(ls "$MC_ROOT"/build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so | head -1)"
DEST="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
cp "$SO" "$DEST"
ASIO="$(ls "$MC_ROOT"/build/mooncake-common/libasio.so 2>/dev/null | head -1 || true)"
[ -n "$ASIO" ] && { cp "$ASIO" /usr/local/lib/ && ldconfig 2>/dev/null || true; }

# ---- verify: gate string present in the installed binary + import works ----
# grep into a var (not `strings | grep -q`): under `set -o pipefail`, grep -q
# closing the pipe early sends SIGPIPE to strings -> non-zero -> false negative.
if ! strings "$DEST" | grep -c MC_ENABLE_HIP_TRANSPORT | grep -qv '^0$'; then
    echo "[mc-hip-gate] ERROR: rebuilt engine.so lacks the gate — build did not take" >&2
    exit 1
fi
python3 -c "from mooncake.engine import TransferEngine" || { echo "[mc-hip-gate] import failed" >&2; exit 1; }
echo "[mc-hip-gate] DONE — HIP transport gated, engine.so reinstalled at $DEST"

# trim the build tree to keep the image lean
rm -rf "$MC_ROOT/build"
