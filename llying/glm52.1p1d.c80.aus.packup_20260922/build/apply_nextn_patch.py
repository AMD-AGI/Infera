#!/usr/bin/env python3
"""Idempotently add the GLM-5.2 NextN shared-experts fusion architecture."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ.get("SGLANG_DIR", "/sgl-workspace/sglang"))
TARGET = ROOT / "python/sglang/srt/models/glm4_moe.py"
CLASS_LINE = "class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):\n"
MARKER = 'fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"'
INSERT = (
    CLASS_LINE
    + "    # The loader rewrites the draft arch to this name, so the inherited\n"
    + "    # DeepseekV3 value would never match and silently disable fusion.\n"
    + f"    {MARKER}\n\n"
)


def main() -> None:
    source = TARGET.read_text()
    if MARKER in source:
        if source.count(MARKER) != 1:
            raise SystemExit(f"unexpected duplicate marker in {TARGET}")
        print(f"[nextn] already applied: {TARGET}")
        return

    if source.count(CLASS_LINE) != 1:
        raise SystemExit(f"NextN class anchor missing or duplicated in {TARGET}")

    TARGET.write_text(source.replace(CLASS_LINE, INSERT, 1))
    pycache = TARGET.parent / "__pycache__"
    if pycache.is_dir():
        for path in pycache.glob("glm4_moe.*.pyc"):
            path.unlink()
    print(f"[nextn] applied: {TARGET}")


if __name__ == "__main__":
    main()
