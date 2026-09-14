#!/usr/bin/env python3
"""Backport GLM-5.2 fixes whose upstream diffs predate the v0.5.17 layout.

Targets exactly the source trees in deploy/docker/Dockerfile.sglang. Each
replacement is fail-closed and idempotent so source drift cannot silently build
an unpatched image.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SGLANG = Path(os.environ.get("SGLANG_ROOT", "/sgl-workspace/sglang"))
AITER = Path(os.environ.get("AITER_ROOT", "/sgl-workspace/aiter"))


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"[opt-patch] {label}: already applied")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor in {path}, found {count}")
    path.write_text(text.replace(old, new))
    print(f"[opt-patch] {label}: applied")


def patch_glm_correction_bias() -> None:
    # sglang#37133: HF stores this GLM-5.2 bias as fp32. Downcasting it at
    # construction and again at the aiter boundary collapses routing levels.
    model = SGLANG / "python/sglang/srt/models/deepseek_v2.py"
    replace_once(
        model,
        "\n\nclass MoEGate(nn.Module):\n",
        """

def _is_glm_moe_dsa(config) -> bool:
    \"\"\"Whether this is the GLM-5.2 main model or its NextN draft head.\"\"\"
    return any("GlmMoeDsa" in arch for arch in (config.architectures or []))


class MoEGate(nn.Module):
""",
        "sglang#37133 model predicate",
    )
    replace_once(
        model,
        """            if quant_config is not None:
                if _use_aiter and quant_config.get_name() in (
""",
        """            # GLM-5.2's offset correction bias must stay fp32.
            if quant_config is not None and not _is_glm_moe_dsa(config):
                if _use_aiter and quant_config.get_name() in (
""",
        "sglang#37133 parameter dtype",
    )

    topk = SGLANG / "python/sglang/srt/layers/moe/topk.py"
    replace_once(
        topk,
        """        aiter_biased_grouped_topk(
            gating_output,
            correction_bias.to(dtype=gating_output.dtype),
            topk_weights,
""",
        """        if correction_bias.dtype == torch.float32:
            aiter_gating_output = gating_output.to(torch.float32)
            aiter_bias = correction_bias
        else:
            aiter_gating_output = gating_output
            aiter_bias = correction_bias.to(dtype=gating_output.dtype)
        aiter_biased_grouped_topk(
            aiter_gating_output,
            aiter_bias,
            topk_weights,
""",
        "sglang#37133 aiter boundary",
    )


def patch_aiter_large_logits_gate() -> None:
    # aiter#5121: the kernel re-bases per row/tile. Gate buffer ops on the
    # largest element offset, not total tensor bytes. The pinned aiter predates
    # the later BLOCK_M=2 change, so only this dependency-free hunk is ported.
    logits = AITER / "aiter/ops/triton/attention/fp8_mqa_logits.py"
    replace_once(
        logits,
        """        # Buffer ops use a 32-bit byte offset (2 GiB resource descriptor cap).
        # Fall back to plain global load/store when a tensor exceeds that.
        BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
        use_buffer_load = KV.numel() * KV.element_size() < BUFFER_LIMIT_BYTES
        use_buffer_store = logits.numel() * logits.element_size() < BUFFER_LIMIT_BYTES
""",
        """        # aiter#5121: buffer descriptors are re-based per row/tile, so
        # only the largest element offset formed by the kernel must fit int32.
        INT32_MAX = 2**31 - 1
        max_kv_offset = (seq_len_kv - 1) * stride_kv_s + (head_size - 1) * stride_kv_d
        max_logits_offset = (seq_len - 1) * stride_logits_s + (
            seq_len_kv - 1
        ) * stride_logits_k
        use_buffer_load = max_kv_offset <= INT32_MAX
        use_buffer_store = max_logits_offset <= INT32_MAX
""",
        "aiter#5121 offset gate",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm-correction-bias", action="store_true")
    parser.add_argument("--aiter-large-logits-gate", action="store_true")
    args = parser.parse_args()
    if not args.glm_correction_bias and not args.aiter_large_logits_gate:
        args.glm_correction_bias = True
        args.aiter_large_logits_gate = True
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.glm_correction_bias:
        patch_glm_correction_bias()
    if args.aiter_large_logits_gate:
        patch_aiter_large_logits_gate()
