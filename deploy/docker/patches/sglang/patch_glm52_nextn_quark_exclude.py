#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Temporary backport of sgl-project/sglang#30265 — GLM-5.2 MTP (nextn) quark-exclude fix.

WHAT: GLM-5.2's MTP layer (index = num_hidden_layers) is ENTIRELY bf16/unquantized —
the quark quantization_config lists `model.layers.<N>.eh_proj` (and the layer's other
submodules) in `exclude`. sglang's DeepseekV3ForCausalLMNextN disables nextn quant only
when the BARE layer prefix `model.layers.<N>` is in exclude_layers, but the exclude
entries are submodule-level (`...<N>.eh_proj`), so `should_ignore_layer()` returns False,
`eh_proj` is built as an MXFP4 (uint8) param, and draft weight-load dies:
  AssertionError: param.shape=[6144,6144] uint8 vs loaded_weight.shape=[6144,12288] bf16.

FIX: probe the `eh_proj` submodule (an exact exclude entry) instead of the bare layer, so
the match succeeds -> nextn_quant_config=None -> the whole (bf16) MTP layer is built bf16.
Verified: coherent GLM-5.2 MTP output on the v0.5.15.post1 base.

UPSTREAM: sgl-project/sglang#30265 (merged 2026-07-08) is the full fix (a dedicated
GlmMoeDsaForCausalLMNextN class). Our base image ships sglang v0.5.15.post1 which predates
it. DROP THIS PATCH once the base sglang carries #30265 (the anchor below will be gone and
this becomes a no-op).

Self-locating, idempotent, no-op if the anchor is absent (sglang refactored / already fixed).
"""

import importlib.util
import sys
from pathlib import Path


def _target():
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        return None
    f = Path(spec.origin).parent / "srt" / "models" / "deepseek_nextn.py"
    return f if f.exists() else None


def main():
    f = _target()
    if f is None:
        print("[glm52-nextn] sglang deepseek_nextn.py not found — skipping")
        return 0
    src = f.read_text()
    old = 'ckpt_prefix = f"model.layers.{config.num_hidden_layers}"'
    new = 'ckpt_prefix = f"model.layers.{config.num_hidden_layers}.eh_proj"'
    if new in src:
        print("[glm52-nextn] already patched — skipping")
        return 0
    if old not in src:
        print("[glm52-nextn] anchor absent (sglang has #30265 / refactored?) — skipping")
        return 0
    f.write_text(src.replace(old, new, 1))
    print(f"[glm52-nextn] patched {f} (GLM-5.2 MTP eh_proj quark-exclude; backport of #30265)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
