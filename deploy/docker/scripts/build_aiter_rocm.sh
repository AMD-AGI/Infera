#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# WHAT: build + install AITER from source (official ROCm/aiter) for DSv4 MXFP4 MoE.
#   Idempotent: skips if the flydsl fp4 path is present.
# WHY: stock base aiter (0.1.13.post1) lacks the flydsl fp4 a4w4 kernel DSv4 needs
#   (else cktile fallback -> "Unsupported scales/output dtype!"); prebuilt .so hits
#   GLIBCXX/GLIBC ABI mismatch, so we compile in-container for ABI match. The ref
#   also carries the Silu gate_mode fix + MHC accuracy fix ROCm/aiter#3033.
# ENV: AITER_GIT_REF (v0.1.16.post1), AITER_REPO, AITER_ROOT.
# USAGE: bash deploy/docker/scripts/build_aiter_rocm.sh   (ROCm container w/ hipcc/cmake/git)
set -euo pipefail

AITER_GIT_REF="${AITER_GIT_REF:-v0.1.16.post1}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter}"
AITER_ROOT="${AITER_ROOT:-/opt/aiter_build}"

echo "============================================"
echo "  AITER ROCm source build (DSv4 MXFP4 flydsl fp4)"
echo "  ref=${AITER_GIT_REF}  root=${AITER_ROOT}"
echo "============================================"

# ---- skip if a flydsl-fp4-capable aiter is already present ------------------
# Probe on-disk, NOT via `import aiter`: import inits the triton/GPU driver, absent
# in a `docker build` sandbox (no /dev/kfd) -> failure.
if python3 - <<'PY' 2>/dev/null
import importlib.util as u, os, sys
spec = u.find_spec("aiter")   # locate package dir without importing
if spec and spec.submodule_search_locations:
    root = list(spec.submodule_search_locations)[0]
    sys.exit(0 if os.path.exists(os.path.join(root, "ops/flydsl/moe_common.py")) else 1)
sys.exit(1)
PY
then
    echo "aiter with flydsl fp4 path already present — skipping build"
    exit 0
fi

# Drop the stock wheel so our editable source install takes over.
pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true

# ---- source tree -----------------------------------------------------------
if [ ! -d "$AITER_ROOT/.git" ]; then
    mkdir -p "$(dirname "$AITER_ROOT")"
    git clone "$AITER_REPO" "$AITER_ROOT"
fi
cd "$AITER_ROOT"
git checkout "$AITER_GIT_REF"
git submodule update --init --recursive

# ---- build + install (editable) --------------------------------------------
# No arch pin: this editable install does not compile kernels (PREBUILD_KERNELS
# is unset). Kernels are JIT-compiled at first import on the live GPU, whose arch
# aiter auto-detects via rocminfo (GPU_ARCHS defaults to "native"). To bake an
# AOT cache for a specific arch, export GPU_ARCHS=<gfxNNNN> PREBUILD_KERNELS=1.
echo "=== pip install -e . ==="
pip install -e . 2>&1 | tail -25

# ---- verify (on-disk; import needs GPU driver, absent in docker build) ----
python3 - <<'PY'
import importlib.util as u, os
spec = u.find_spec("aiter")
assert spec and spec.submodule_search_locations, "aiter package not found after build"
root = list(spec.submodule_search_locations)[0]
assert os.path.exists(os.path.join(root, "ops/flydsl/moe_common.py")), \
    "flydsl fp4 path missing after build"
print("AITER flydsl fp4 path OK ->", root)
PY
echo "AITER_BUILD_DONE"
