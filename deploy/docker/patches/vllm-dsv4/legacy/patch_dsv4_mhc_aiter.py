###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""[DEPRECATED — NEW IMAGE NO-OP] Route DeepSeek-V4 MHC (multi-head compressor)
pre/post through the AITER kernel fast-path in vLLM's
``model_executor/layers/mhc.py`` (ROCm / gfx950 / MI355X).

WHAT: prefer the aiter MHC kernel (guarded by hidden_size%256==0), then tilelang
  if requested, else torch-native; selectable via VLLM_MHC_BACKEND.
WHY: stock vLLM uses mhc_*_tilelang, but tilelang 0.1.10 can't emit a gfx950 image
  -> EngineCore dies; the correct aiter path needs fix ROCm/aiter#3033 (b639cb63).
DEPS: NO-OP on dev748 (aiter MHC path upstream, old anchor absent) -> NOT applied
  by any Dockerfile, kept in legacy/ for older bases (see README.md + companion .diff).
  VERSION: verified vllm 0.23.1rc1.dev424, aiter must carry #3033.
"""

import os
import sys

try:
    import vllm
except Exception:
    print("dsv4-mhc-aiter: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "model_executor/layers/mhc.py",
)
if not os.path.exists(f):
    print(f"dsv4-mhc-aiter: {f} not found (no MHC in this vLLM) — skipping")
    sys.exit(0)

src = open(f).read()
MARK = "# yihou: MHC aiter fast-path"
if MARK in src:
    print("dsv4-mhc-aiter: already patched")
    sys.exit(0)

# 1) Insert the backend selector helper right after HAS_TILELANG_MHC is defined.
ANCHOR = "HAS_TILELANG_MHC = _has_tilelang_mhc()\n"
if src.count(ANCHOR) != 1:
    print(
        f"dsv4-mhc-aiter: anchor found {src.count(ANCHOR)}x (expected 1) — layout drifted, skipping"
    )
    sys.exit(0)
HELPER = (
    ANCHOR
    + "\n"
    + "import os as _os  "
    + MARK
    + "\n"
    + "def _mhc_backend() -> str:\n"
    + "    # aiter (default) | tilelang | native. aiter kernel needs\n"
    + "    # hidden_size%256==0; caller still guards and falls back.\n"
    + "    # tilelang broken on gfx950 here.\n"
    + "    return _os.getenv('VLLM_MHC_BACKEND', 'aiter').lower()\n"
)
src = src.replace(ANCHOR, HELPER, 1)

# 2) mhc_pre forward_hip: prepend the aiter/tilelang selection.
PRE_OLD = (
    "        if HAS_TILELANG_MHC:\n"
    "            return torch.ops.vllm.mhc_pre_tilelang(\n"
    "                residual,\n"
    "                fn,\n"
    "                hc_scale,\n"
    "                hc_base,\n"
    "                rms_eps,\n"
    "                hc_pre_eps,\n"
    "                hc_sinkhorn_eps,\n"
    "                hc_post_mult_value,\n"
    "                sinkhorn_repeat,\n"
    "                n_splits,\n"
    "                norm_weight,\n"
    "                norm_eps,\n"
    "            )\n"
)
PRE_NEW = (
    "        _be = _mhc_backend()  " + MARK + "\n"
    "        if _be == 'aiter' and residual.shape[-1] % 256 == 0:\n"
    "            return torch.ops.vllm.mhc_pre_aiter(\n"
    "                residual,\n"
    "                fn,\n"
    "                hc_scale,\n"
    "                hc_base,\n"
    "                rms_eps,\n"
    "                hc_pre_eps,\n"
    "                hc_sinkhorn_eps,\n"
    "                hc_post_mult_value,\n"
    "                sinkhorn_repeat,\n"
    "            )\n"
    "        if _be == 'tilelang' and HAS_TILELANG_MHC:\n"
    "            return torch.ops.vllm.mhc_pre_tilelang(\n"
    "                residual,\n"
    "                fn,\n"
    "                hc_scale,\n"
    "                hc_base,\n"
    "                rms_eps,\n"
    "                hc_pre_eps,\n"
    "                hc_sinkhorn_eps,\n"
    "                hc_post_mult_value,\n"
    "                sinkhorn_repeat,\n"
    "                n_splits,\n"
    "                norm_weight,\n"
    "                norm_eps,\n"
    "            )\n"
)
if src.count(PRE_OLD) != 1:
    print(f"dsv4-mhc-aiter: mhc_pre block found {src.count(PRE_OLD)}x (expected 1) — skipping")
    sys.exit(1)
src = src.replace(PRE_OLD, PRE_NEW, 1)

# 3) mhc_post forward_hip: prepend the aiter/tilelang selection.
POST_OLD = (
    "        if HAS_TILELANG_MHC:\n"
    "            return torch.ops.vllm.mhc_post_tilelang(\n"
    "                x, residual, post_layer_mix, comb_res_mix\n"
    "            )\n"
)
POST_NEW = (
    "        _be = _mhc_backend()  " + MARK + "\n"
    "        if _be == 'aiter' and residual.shape[-1] % 256 == 0:\n"
    "            return torch.ops.vllm.mhc_post_aiter(\n"
    "                x, residual, post_layer_mix, comb_res_mix\n"
    "            )\n"
    "        if _be == 'tilelang' and HAS_TILELANG_MHC:\n"
    "            return torch.ops.vllm.mhc_post_tilelang(\n"
    "                x, residual, post_layer_mix, comb_res_mix\n"
    "            )\n"
)
if src.count(POST_OLD) != 1:
    print(f"dsv4-mhc-aiter: mhc_post block found {src.count(POST_OLD)}x (expected 1) — skipping")
    sys.exit(1)
src = src.replace(POST_OLD, POST_NEW, 1)

open(f, "w").write(src)
print(f"dsv4-mhc-aiter: patched {f} (MHC -> aiter fast-path, pre+post)")
