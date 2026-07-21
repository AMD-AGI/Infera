#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Bake the DSv4 FP4->FP8 MoE dequant source patch into the SGLang tree (gfx942).

WHAT: git-applies the sibling ``sglang_fp8_fp4_dequant_gfx942.patch`` to the
  SGLang repo (``/sgl-workspace/sglang``). The patch edits
  ``python/sglang/srt/layers/quantization/fp8.py``
  (``Fp8MoEMethod.process_weights_after_loading_block_quant``): when
  ``SGLANG_DSV4_FP4_DEQUANT`` is set it skips the aiter-native-mxfp4 branch and
  dequantizes the FP4 experts to FP8 block-quant (``cast_e2m1fn_to_e4m3fn``),
  then runs the standard aiter FP8 blockscale MoE.

WHY: gfx942/CDNA3 (MI325X) has NO working aiter mxfp4 MoE kernel (fp4x2 quant is
  gfx950-only). aiter's FP8 blockscale MoE works on gfx942, so we dequant
  FP4->FP8 at load. Verified on 8xMI325X: simple-prompt correct + conc=32 stress
  2284 tok/s (cudagraph + GEMM tune). aiter is NOT rebuilt.

SAFETY (both requirements):
  * The added code path is gated by ``self.dequant_fp4_to_fp8`` (= env
    ``SGLANG_DSV4_FP4_DEQUANT``, default False) AND ``self.is_fp4_expert``. On
    gfx950 (MI355X) the launcher never sets the env -> the original
    aiter-native-mxfp4 path runs unchanged. For non-FP4-expert models the
    is_fp4_expert guard is False -> no-op. So baking this in cannot break
    MI355X or other models.
  * The launcher (infera.engine.rocm_rdma_env) only sets the env when it detects
    gfx942 AND a DSv4-FP4 model; see apply_dsv4_gfx942_env_defaults.

  Also installs the gfx942/cu_num=304 tuned aiter bf16 GEMM CSV
  (``tuned_dsv4_cu304.csv``) into the image so the launcher can point
  ``AITER_CONFIG_GEMM_BF16`` at it (decode-path torch fallback -> aiter,
  +3.8% throughput / -16% TTFT; pure perf, gfx942-only, no effect on gfx950).

DEPS: baked by Dockerfile.sglang.gfx942. Idempotent (reverse-check), self-locating,
  and a NO-OP that never fails the build if the SGLang source drifted so the
  patch no longer applies (loud warning, exit 0). VERSION: verified against
  sglang @ 9fec359a60 (0.5.15.dev); ROCm 7.2.0.
"""

import os
import shutil
import subprocess
import sys

TAG = "sglang-patch dsv4-fp4-dequant-gfx942"
HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "sglang_fp8_fp4_dequant_gfx942.patch")
CSV_SRC = os.path.join(HERE, "tuned_dsv4_cu304.csv")
# Where the launcher expects the tuned GEMM CSV (AITER_CONFIG_GEMM_BF16 target).
CSV_DST = os.environ.get(
    "INFERA_DSV4_GEMM_CSV", "/opt/infera/aiter_configs/tuned_dsv4_cu304.csv"
)


def _sglang_repo_root() -> str | None:
    """Locate the SGLang git repo root (holds ``python/sglang/...``)."""
    root = os.environ.get("SGLANG_DIR")
    candidates = [root] if root else []
    try:
        import sglang  # noqa: F401

        pkg = os.path.dirname(os.path.abspath(sglang.__file__))  # .../python/sglang
        # repo root = two levels up from the package dir (python/sglang -> repo)
        candidates.append(os.path.dirname(os.path.dirname(pkg)))
    except Exception:
        pass
    candidates.append("/sgl-workspace/sglang")
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "python/sglang/srt/layers/quantization/fp8.py")):
            return c
    return None


def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def _install_gemm_csv() -> None:
    if not os.path.isfile(CSV_SRC):
        print(f"[{TAG}] WARNING: tuned GEMM CSV missing ({CSV_SRC}) — GEMM tuning not installed")
        return
    os.makedirs(os.path.dirname(CSV_DST), exist_ok=True)
    shutil.copyfile(CSV_SRC, CSV_DST)
    print(f"[{TAG}] installed tuned GEMM CSV -> {CSV_DST}")


def _apply_patch() -> None:
    if not os.path.isfile(PATCH):
        print(f"[{TAG}] WARNING: patch file missing ({PATCH}) — skipped")
        return
    root = _sglang_repo_root()
    if not root:
        print(f"[{TAG}] SGLang tree not found — skipped")
        return
    if not os.path.isdir(os.path.join(root, ".git")):
        print(f"[{TAG}] WARNING: {root} is not a git tree — cannot git-apply, skipped")
        return

    if _git(root, "apply", "--reverse", "--check", PATCH).returncode == 0:
        print(f"[{TAG}] already applied to {root} — nothing to do")
        return
    if _git(root, "apply", "--check", PATCH).returncode != 0:
        chk = _git(root, "apply", "--check", PATCH)
        print(
            f"[{TAG}] WARNING: patch does not apply to {root} — SGLang source "
            f"likely drifted; SKIPPED (re-derive the patch). git said:\n{chk.stderr.strip()}"
        )
        return
    res = _git(root, "apply", PATCH)
    if res.returncode != 0:
        print(f"[{TAG}] WARNING: git apply failed after check passed — skipped:\n{res.stderr.strip()}")
        return
    print(f"[{TAG}] patched {root}")
    print(_git(root, "diff", "--stat").stdout.strip())


def main() -> int:
    _apply_patch()
    _install_gemm_csv()
    return 0


if __name__ == "__main__":
    sys.exit(main())
