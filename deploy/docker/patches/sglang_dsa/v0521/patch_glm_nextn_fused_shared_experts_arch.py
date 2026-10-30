#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Backport the GLM NextN fused-shared-experts architecture name (v0.5.22).

WHAT: ModelConfig rewrites a GLM draft's architecture to
`GlmMoeDsaForCausalLMNextN`, but on v0.5.21 that class inherits
`fused_shared_experts_architecture = "DeepseekV3ForCausalLMNextN"`, so the gate
in deepseek_v2.py never matches and the NextN draft silently runs with the
shared experts unfused. v0.5.22 sets the attribute on the class; this script
inserts the same line, so `v0522/apply_sglang_dsa_patches_v0522.sh` (which
asserts it) holds on the rocm10.2 v0.5.21 base.

Self-locating and idempotent; exits non-zero when the anchor drifted.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[glm-nextn-fused-arch]"
OLD = "class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):\n"
NEW = OLD + (
    "    # ModelConfig rewrites a GLM draft's architecture to this class's own name,\n"
    "    # so the inherited DeepSeek name would never match what the gate compares.\n"
    '    fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"\n'
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        print(f"{TAG} cannot locate sglang")
        return 1
    path = Path(spec.origin).parent / "srt" / "models" / "glm4_moe.py"
    source = path.read_text()
    if NEW in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 1:
        print(f"{TAG} anchor found {source.count(OLD)}x, want 1x in {path} — re-cut the patch")
        return 1
    path.write_text(source.replace(OLD, NEW, 1))
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
