#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Bake the DSv4 FP4->FP8 MoE dequant source patch into the ATOM tree (gfx942).

WHAT: git-applies the sibling ``atom_dsv4_fp4_dequant_gfx942.patch`` to
  ``/app/ATOM``. That patch adds env ``ATOM_DSV4_FP4_DEQUANT`` (default 0) and,
  when set, dequantizes DeepSeek-V4-Pro's FP4 (e2m1 + ue8m0) routed experts to
  FP8 e4m3 128x128-block at load and swaps the layer to ATOM's ``Fp8MoEMethod``.

WHY: gfx942/CDNA3 (MI325X) has NO working FP4 MoE kernel (aiter's CK
  ``kernel_moe_mxgemm`` is gfx950-only; ATOM's triton MoE needs the absent
  ``triton_kernels.routing``). aiter's FP8 blockscale MoE
  (``fmoe_fp8_blockscale_g1u1``) DOES work on gfx942, so we dequant FP4->FP8 at
  load and run that. Verified on 8xMI325X: simple-prompt correct + conc=32
  stress 3958 tok/s (cudagraph). aiter is NOT rebuilt.

SAFETY (both requirements):
  * The code path is env-gated (``ATOM_DSV4_FP4_DEQUANT``, default OFF) AND
    guarded by a ``quant_dtype == fp4x2`` / ``Mxfp4MoEMethod`` check inside the
    patch. On gfx950 (MI355X) the launcher never sets the env -> native FP4 path
    runs unchanged. For non-FP4 models the fp4x2 guard is False -> no-op even if
    the env were set. So baking this in cannot break MI355X or other models.
  * The launcher (infera.engine.rocm_rdma_env) only sets the env when it detects
    gfx942 AND a DSv4-FP4 model; see apply_dsv4_gfx942_env_defaults.

DEPS: baked by Dockerfile.atom's ``patches/atom/patch_*.py`` loop. Idempotent
  (reverse-check), self-locating, and a NO-OP that never fails the build if the
  ATOM source drifted so the patch no longer applies (loud warning, exit 0) --
  same graceful-degradation convention as the other atom patches.
  VERSION: verified against atom @ 5837907f3 (v0.1.4.dev113); ROCm 7.2.4.
"""

import os
import subprocess
import sys

ATOM_DIR = os.environ.get("ATOM_DIR", "/app/ATOM")
PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atom_dsv4_fp4_dequant_gfx942.patch")
TAG = "atom-patch dsv4-fp4-dequant-gfx942"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", ATOM_DIR, *args],
        capture_output=True,
        text=True,
    )


def main() -> int:
    if not os.path.isfile(PATCH):
        print(f"[{TAG}] WARNING: patch file missing ({PATCH}) — skipped")
        return 0
    if not os.path.isdir(ATOM_DIR):
        print(f"[{TAG}] {ATOM_DIR} not found (no ATOM tree) — skipped")
        return 0
    if not os.path.isdir(os.path.join(ATOM_DIR, ".git")):
        print(f"[{TAG}] WARNING: {ATOM_DIR} is not a git tree — cannot git-apply, skipped")
        return 0

    # Already applied? (reverse-check succeeds only when the patch is present.)
    if _git("apply", "--reverse", "--check", PATCH).returncode == 0:
        print(f"[{TAG}] already applied — nothing to do")
        return 0

    # Will it apply cleanly against the current source?
    if _git("apply", "--check", PATCH).returncode != 0:
        chk = _git("apply", "--check", PATCH)
        print(
            f"[{TAG}] WARNING: patch does not apply to {ATOM_DIR} — ATOM source "
            f"likely drifted; SKIPPED (re-derive the patch). git said:\n{chk.stderr.strip()}"
        )
        return 0

    res = _git("apply", PATCH)
    if res.returncode != 0:
        print(f"[{TAG}] WARNING: git apply failed after check passed — skipped:\n{res.stderr.strip()}")
        return 0
    print(f"[{TAG}] patched {ATOM_DIR}")
    print(_git("diff", "--stat").stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
