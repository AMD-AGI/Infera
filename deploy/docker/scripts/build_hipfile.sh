#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Build & install hipFile (AIS — AMD AI Storage) from the
# ROCm/rocm-systems monorepo into /opt/rocm of the current image.
#
# Source of truth:
#   https://github.com/ROCm/rocm-systems  (path: projects/hipfile)
#   License upstream: MIT (matches this repo's LICENSE).
#
# Why this script exists:
#   - hipFile is NOT shipped in the lmsysorg/sglang:v0.5.12.post1-rocm720-mi35x
#     base image. To exercise the SSD <-> GPU direct-DMA L3 path we need
#     libhipfile.so + headers installed alongside the existing rocm-7.2 stack.
#   - This is an OPT-IN layer: the base image without hipFile still serves;
#     hipFile only becomes the L3 backend when `--long-backend hipfile` is
#     passed to kvd (Phase 1 flag).
#   - The libhipfile python binding from LMCache requires libhipfile.so be
#     built first (A4 finding) — `pip install` is not a substitute.
#
# Findings this script encodes (from Task #8 / A1 research):
#   - lmsysorg/sglang:v0.5.12.post1-rocm720-mi35x ships rocm-7.2.0 full stack,
#     hsa-rocr-dev, rocm-hip, amdgpu-dkms 30.30.0, amdclang++ 22.0.0,
#     rocm-cmake, cmake 3.31.10, ninja, Boost 1.74 (all-dev). The ONLY apt
#     dep missing for the hipFile build is libmount-dev (libmount1 present,
#     headers not). We install it idempotently.
#   - No separate kernel module: hipFile uses the amdgpu/amdkfd KFD ioctl on
#     host kernels with CONFIG_PCI_P2PDMA=y. The L0/L1 host-libionic story
#     is unrelated to hipFile and is handled by the Mooncake fix elsewhere.
#   - Build recipe per A1: cmake -DCMAKE_HIP_PLATFORM=amd
#       -DCMAKE_INSTALL_PREFIX=/opt/rocm -DAIS_INSTALL_EXAMPLES=ON
#       -DAIS_INSTALL_TOOLS=ON -DCMAKE_BUILD_TYPE=Release ..
#   - AIS_INSTALL_TOOLS=ON produces `ais-check`, which prints
#     "Kernel P2PDMA support: True/False". We use that as the DMA-live
#     assertion (mirrors the Bench 0 sanity gate).
#
# Modes:
#   build_hipfile.sh                  # full clone (if absent) + build + install + probe
#   build_hipfile.sh --probe-only     # only run `ais-check` against an existing install
#
# Environment overrides:
#   HIPFILE_SRC_DIR          (default /opt/rocm-systems-src) — where to clone
#   HIPFILE_GIT_URL          (default https://github.com/ROCm/rocm-systems.git)
#   HIPFILE_GIT_REF          (default HEAD of the default branch) — set to a
#                            tag / commit for reproducible builds
#   HIPFILE_FORCE_RESYNC=1   — `git fetch` + `git reset --hard` the existing
#                              clone before building (default: keep as-is)
#   HIPFILE_INSTALL_PREFIX   (default /opt/rocm)
#   HIPFILE_BUILD_JOBS       (default $(nproc))
#   HIPFILE_SKIP_PROBE=1     — skip the final ais-check smoke test (CI use)
#
# Idempotency contract:
#   Re-running with no env changes after a successful install should be a
#   near-no-op: apt is a hit on installed lists, git clone is skipped because
#   the dir exists, cmake reconfigure detects no changes, ninja sees no work,
#   `cmake --install .` is a re-link, ais-check runs. Wall < 30 s.

set -euo pipefail

# --- Args -------------------------------------------------------------------

PROBE_ONLY=0
case "${1:-}" in
    --probe-only)
        PROBE_ONLY=1
        ;;
    "")
        ;;
    *)
        echo "ERROR: unknown argument: $1" >&2
        echo "Usage: $0 [--probe-only]" >&2
        exit 2
        ;;
esac

# --- Config -----------------------------------------------------------------

HIPFILE_SRC_DIR="${HIPFILE_SRC_DIR:-/opt/rocm-systems-src}"
HIPFILE_GIT_URL="${HIPFILE_GIT_URL:-https://github.com/ROCm/rocm-systems.git}"
HIPFILE_GIT_REF="${HIPFILE_GIT_REF:-}"
HIPFILE_FORCE_RESYNC="${HIPFILE_FORCE_RESYNC:-0}"
HIPFILE_INSTALL_PREFIX="${HIPFILE_INSTALL_PREFIX:-/opt/rocm}"
HIPFILE_BUILD_JOBS="${HIPFILE_BUILD_JOBS:-$(nproc 2>/dev/null || echo 4)}"
HIPFILE_SKIP_PROBE="${HIPFILE_SKIP_PROBE:-0}"

HIPFILE_SUBDIR="${HIPFILE_SRC_DIR}/projects/hipfile"
HIPFILE_BUILD_DIR="${HIPFILE_SUBDIR}/build"

log()  { echo "[build_hipfile] $*"; }
fail() { echo "[build_hipfile] ERROR: $*" >&2; exit 1; }

# --- Probe helper -----------------------------------------------------------
#
# Runs `ais-check` and verifies the output contains the
# "Kernel P2PDMA support" line. We deliberately do NOT require the value to
# be True — the host kernel may not have P2PDMA on the build box. What we
# DO require is that the tool ran and printed the line, which proves the
# install is wired (binary + .so + headers + libmount runtime dep).
#
# Set HIPFILE_REQUIRE_P2PDMA=1 to make a False result also fatal — useful
# for the post-install probe on a real MI355X node (Task #13's job).

probe_ais_check() {
    local ais_check_bin
    ais_check_bin="$(command -v ais-check || true)"
    if [ -z "${ais_check_bin}" ] && [ -x "${HIPFILE_INSTALL_PREFIX}/bin/ais-check" ]; then
        ais_check_bin="${HIPFILE_INSTALL_PREFIX}/bin/ais-check"
    fi
    if [ -z "${ais_check_bin}" ]; then
        fail "ais-check not found on PATH or in ${HIPFILE_INSTALL_PREFIX}/bin (install incomplete)"
    fi

    log "Running smoke probe: ${ais_check_bin}"
    local probe_log="/tmp/ais-check.out"
    # Tee to stdout AND a file so the build log shows what happened and we
    # have a stable artifact to grep + reference from the README.
    if ! "${ais_check_bin}" 2>&1 | tee "${probe_log}"; then
        # ais-check itself may exit non-zero on a box without P2PDMA — that
        # is NOT a build failure, so we ignore the exit code here and rely
        # on the output check below to decide.
        log "ais-check exited non-zero (may be expected on a box without P2PDMA)"
    fi

    if ! grep -q "Kernel P2PDMA support" "${probe_log}"; then
        fail "smoke probe failed: 'Kernel P2PDMA support' line not in ais-check output"
    fi
    log "Smoke probe OK — install + tooling work (see ${probe_log} for kernel support status)."

    if [ "${HIPFILE_REQUIRE_P2PDMA:-0}" = "1" ]; then
        if ! grep -q "Kernel P2PDMA support: True" "${probe_log}"; then
            fail "HIPFILE_REQUIRE_P2PDMA=1 set but probe reports DMA NOT live (compat-mode silent fallback)"
        fi
        log "HIPFILE_REQUIRE_P2PDMA=1: confirmed DMA path is live."
    fi
}

# --- Probe-only short-circuit ----------------------------------------------

if [ "${PROBE_ONLY}" = "1" ]; then
    log "Probe-only mode."
    probe_ais_check
    exit 0
fi

# --- Step 1: apt dep (libmount-dev) -----------------------------------------

# Idempotent: dpkg -s returns 0 if already installed; we skip apt entirely
# in that case so re-runs don't pay the apt-update cost.
if dpkg -s libmount-dev >/dev/null 2>&1; then
    log "libmount-dev already installed — skipping apt."
else
    log "Installing libmount-dev (only missing apt dep per A1)."
    if [ "$(id -u)" != "0" ]; then
        fail "must run as root to apt-get install libmount-dev (or pre-install it)"
    fi
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends libmount-dev
    rm -rf /var/lib/apt/lists/*
fi

# --- Step 2: source tree ----------------------------------------------------

if [ -d "${HIPFILE_SRC_DIR}/.git" ]; then
    if [ "${HIPFILE_FORCE_RESYNC}" = "1" ]; then
        log "HIPFILE_FORCE_RESYNC=1 — fetch + reset ${HIPFILE_SRC_DIR}"
        git -C "${HIPFILE_SRC_DIR}" fetch --depth=1 origin "${HIPFILE_GIT_REF:-HEAD}"
        git -C "${HIPFILE_SRC_DIR}" reset --hard FETCH_HEAD
    else
        log "Source tree already present at ${HIPFILE_SRC_DIR} — skipping clone (set HIPFILE_FORCE_RESYNC=1 to refresh)."
    fi
else
    log "Cloning ${HIPFILE_GIT_URL} -> ${HIPFILE_SRC_DIR}"
    if [ -n "${HIPFILE_GIT_REF}" ]; then
        # shallow clone with a specific ref
        git clone --depth=1 --branch "${HIPFILE_GIT_REF}" \
            "${HIPFILE_GIT_URL}" "${HIPFILE_SRC_DIR}"
    else
        git clone --depth=1 "${HIPFILE_GIT_URL}" "${HIPFILE_SRC_DIR}"
    fi
fi

if [ ! -d "${HIPFILE_SUBDIR}" ]; then
    fail "expected hipfile project not found at ${HIPFILE_SUBDIR} (upstream layout changed?)"
fi

# --- Step 3: cmake configure + ninja build ----------------------------------

mkdir -p "${HIPFILE_BUILD_DIR}"

# Use ninja if available (faster + better idempotency); fall back to default
# generator if not. The base image ships ninja per A1 so this almost always
# takes the fast path.
GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS+=("-G" "Ninja")
fi

log "Configuring (cmake) — prefix=${HIPFILE_INSTALL_PREFIX}"
# BUILD_TESTING=OFF: skip the cmake test target which has a hard
# dependency on boost-program-options-dev. lmsysorg/sglang ships
# Boost-all-dev so tests build; amdsiloai/vllm ships a leaner image
# without it. We don't actually need the cmake-side tests to ship
# libhipfile.so + ais-check (those are in the main + tools targets).
cmake \
    "${GENERATOR_ARGS[@]}" \
    -S "${HIPFILE_SUBDIR}" \
    -B "${HIPFILE_BUILD_DIR}" \
    -DCMAKE_HIP_PLATFORM=amd \
    -DCMAKE_INSTALL_PREFIX="${HIPFILE_INSTALL_PREFIX}" \
    -DAIS_INSTALL_EXAMPLES=ON \
    -DAIS_INSTALL_TOOLS=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release

log "Building (jobs=${HIPFILE_BUILD_JOBS})"
cmake --build "${HIPFILE_BUILD_DIR}" -j "${HIPFILE_BUILD_JOBS}"

log "Installing to ${HIPFILE_INSTALL_PREFIX}"
cmake --install "${HIPFILE_BUILD_DIR}"

# --- Step 4: sanity check the install ---------------------------------------

# Find the .so by walking common rocm lib paths; the upstream cmake puts it
# under lib/ or lib64/ depending on the platform.
INSTALLED_SO=""
for d in lib lib64; do
    candidate="${HIPFILE_INSTALL_PREFIX}/${d}/libhipfile.so"
    if [ -e "${candidate}" ]; then
        INSTALLED_SO="${candidate}"
        break
    fi
done
if [ -z "${INSTALLED_SO}" ]; then
    fail "libhipfile.so not found under ${HIPFILE_INSTALL_PREFIX}/{lib,lib64} after install"
fi
log "Installed: ${INSTALLED_SO}"

if [ ! -d "${HIPFILE_INSTALL_PREFIX}/include/hipfile" ] && \
   [ ! -e "${HIPFILE_INSTALL_PREFIX}/include/hipfile.h" ]; then
    # Not all upstream layouts use the same header directory; we warn but
    # don't fail because some versions ship headers under include/ais/.
    log "WARN: no hipfile headers found under ${HIPFILE_INSTALL_PREFIX}/include — verify upstream layout"
fi

# --- Step 4b: install the Python binding (separate from cmake install) -------
#
# The CMake install only ships libhipfile.so + headers; the Python binding
# is a sibling package (projects/hipfile/python/) built via scikit-build-core.
# Without this step `import hipfile` raises ImportError and
# infera.engine.{sglang,vllm}.hipfile_shim.is_available() returns False —
# the engine adapter then falls back to UDS / POSIX for every transfer.
#
# This step is idempotent: pip install -e re-resolves the egg-info but
# doesn't recompile the Cython binding unless _hipfile.pyx changed.

PY_BINDING_DIR="${HIPFILE_SRC_DIR}/projects/hipfile/python"
if [ ! -d "${PY_BINDING_DIR}" ]; then
    log "WARN: no Python binding source dir at ${PY_BINDING_DIR} — skipping pip install"
    log "      (older rocm-systems checkouts may not ship the Python wrapper)"
elif [ ! -f "${PY_BINDING_DIR}/pyproject.toml" ]; then
    log "WARN: ${PY_BINDING_DIR} has no pyproject.toml — skipping pip install"
else
    log "Installing Python binding (pip install -e ${PY_BINDING_DIR})"
    if ! python3 -m pip install -e "${PY_BINDING_DIR}" 2>&1 | tail -20; then
        fail "pip install hipfile Python binding failed — see output above"
    fi
    # Sanity import; failure here is a hard fail (we just installed it).
    if ! python3 -c "import hipfile; print(f'hipfile {hipfile.__version__}')" >/dev/null 2>&1; then
        fail "hipfile Python module didn't import after pip install — investigate"
    fi
    log "Python binding OK: $(python3 -c 'import hipfile; print(hipfile.__version__)')"
fi

# --- Step 5: smoke probe (ais-check) ----------------------------------------

if [ "${HIPFILE_SKIP_PROBE}" = "1" ]; then
    log "HIPFILE_SKIP_PROBE=1 — skipping ais-check (use --probe-only on a real MI355X node)."
else
    # Strict by default: missing binary or missing 'Kernel P2PDMA support'
    # line in output is a hard fail (install broken). DMA-live True/False is
    # informational here; gate it via HIPFILE_REQUIRE_P2PDMA=1 (Task #13).
    probe_ais_check
fi

log "hipFile build + install OK."
