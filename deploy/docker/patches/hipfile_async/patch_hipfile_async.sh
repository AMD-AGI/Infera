#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Apply the hipFile async-API binding patch + rebuild the editable
# install. Idempotent (greps for the marker before applying).
#
# Targets the editable install at /opt/rocm-systems-src/projects/hipfile
# in the upstream ROCm vLLM image; bails cleanly (rc 0, warns) when
# the path is absent so non-ROCm bases still build.
#
# The patch adds Python wrappers for:
#   hipFileWriteAsync, hipFileReadAsync,
#   hipFileStreamRegister, hipFileStreamDeregister,
#   + a Python `Stream` context manager + `supports_async()` probe
# in `hipfile/_chipfile.pxd`, `hipfile/_hipfile.pyx`, `hipfile/file.py`.
#
# After patching we re-run `pip install --no-build-isolation -e .`
# from the python/ dir; scikit-build-core picks up the .pyx change
# and rebuilds the Cython extension in place. No image-wide pip
# resolution — the editable install metadata is preserved.

set -euo pipefail

HIPFILE_ROOT="${HIPFILE_ROOT:-/opt/rocm-systems-src/projects/hipfile/python}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${PATCH_DIR}/hipfile_async.patch"
MARKER='hipFileStreamRegister'  # function name we add — grep target

if [[ ! -d "${HIPFILE_ROOT}/hipfile" ]]; then
    echo "[hipfile-async-patch] no hipfile binding at ${HIPFILE_ROOT} — skipping (non-ROCm base?)"
    exit 0
fi

if grep -q "${MARKER}" "${HIPFILE_ROOT}/hipfile/_chipfile.pxd" 2>/dev/null; then
    echo "[hipfile-async-patch] marker '${MARKER}' already present — skip (idempotent)"
    exit 0
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "[hipfile-async-patch] FATAL: ${PATCH_FILE} not found" >&2
    exit 1
fi

echo "[hipfile-async-patch] applying ${PATCH_FILE} at ${HIPFILE_ROOT}"
cd "${HIPFILE_ROOT}"
patch -p1 < "${PATCH_FILE}"

FRAGMENT="${PATCH_DIR}/file_async_fragment.py"
if [[ -f "${FRAGMENT}" ]]; then
    echo "[hipfile-async-patch] appending file_async_fragment.py to hipfile/file.py"
    if ! grep -q 'BEGIN hipfile-async-patch additions' "${HIPFILE_ROOT}/hipfile/file.py"; then
        echo '' >> "${HIPFILE_ROOT}/hipfile/file.py"
        cat "${FRAGMENT}" >> "${HIPFILE_ROOT}/hipfile/file.py"
    fi
else
    echo "[hipfile-async-patch] FATAL: fragment ${FRAGMENT} not found" >&2
    exit 1
fi

echo "[hipfile-async-patch] ensuring scikit-build-core + cython build deps present"
# Install build deps into the venv (need scikit-build-core + cython
# for the .pyx recompile). Pin to the versions declared in
# pyproject.toml so we don't drift. These are small + idempotent.
pip install --no-cache-dir 'scikit-build-core>=0.10' 'cython>=3.0'

echo "[hipfile-async-patch] rebuilding editable install (scikit-build-core will recompile the .pyx)"
# --no-build-isolation so scikit-build-core sees the deps we just
# installed into the venv (and the existing cython/ROCm/hipfile.h
# include paths). --force-reinstall ensures the .so is regenerated
# even when pip thinks the metadata is fresh.
pip install --no-build-isolation --force-reinstall --no-deps -e .

echo "[hipfile-async-patch] smoke-import"
python -c "
import hipfile
print('hipfile version:', hipfile.__version__ if hasattr(hipfile, '__version__') else '(no __version__)')
assert hasattr(hipfile, 'Stream'), 'Stream class missing — patch did not land'
print('  Stream class:', hipfile.Stream)
assert hasattr(hipfile, 'supports_async'), 'supports_async missing — patch did not land'
print('  supports_async():', hipfile.supports_async())
print('  FileHandle.write_async:', hipfile.FileHandle.write_async)
print('  FileHandle.read_async:', hipfile.FileHandle.read_async)
"

echo "[hipfile-async-patch] done"
