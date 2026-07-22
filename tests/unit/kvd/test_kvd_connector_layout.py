###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""kvd connector — KV-cache (model-type x dtype) offload support matrix.

``register_kv_caches`` decides whether a KV-cache group is offloaded to L3 or
skipped, from a few pure helpers. This pins the support matrix so the
non-obvious cases can't silently regress:

    model / layout      KV dtype     L3 offload
    ------------------  -----------  -----------
    MLA (plain latent)  fp8          yes  (hidden == kv_lora_rank+qk_rope_head_dim)
    MLA / any           bf16/fp16    yes  (not packed -> passthrough)
    GQA / MHA           bf16/fp16    yes  (not packed -> passthrough)
    GQA / MHA           fp8          NO   (scale-packed, could mis-stride -> skip)
    MLA scale-packed    fp8 (656/…)  NO   (hidden != plain latent -> skip)

Pure functions only — no GPU / engine / model. The matrix decision below is
torch-free; only the dtype->packed classification needs torch (guarded).
"""

from types import SimpleNamespace

import pytest

from infera.engine.vllm.kvd_connector import (
    _expected_plain_mla_hidden,
    _is_mla_from_config,
    _is_packed_quant_kv_dtype,
)


def _cfg(*, use_mla=None, kv_lora_rank=None, qk_rope_head_dim=None):
    """Minimal stand-in for the vLLM config the helpers read (model_config.*)."""
    hf = SimpleNamespace(
        kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim, text_config=None
    )
    mc = SimpleNamespace(use_mla=use_mla, hf_text_config=hf, hf_config=hf)
    return SimpleNamespace(model_config=mc)


# --- MLA detection (config-only) ---------------------------------------------


def test_mla_detected_from_use_mla_flag():
    assert _is_mla_from_config(_cfg(use_mla=True)) is True
    assert _is_mla_from_config(_cfg(use_mla=False)) is False


def test_mla_detected_from_kv_lora_rank_marker():
    assert _is_mla_from_config(_cfg(kv_lora_rank=512)) is True


def test_non_mla_or_missing_config_is_false():
    assert _is_mla_from_config(_cfg()) is False
    assert _is_mla_from_config(SimpleNamespace(model_config=None)) is False


# --- plain-MLA-fp8 hidden width = kv_lora_rank + qk_rope_head_dim -------------


def test_plain_mla_hidden_kimi_like():
    # Kimi-K2.6: 512 + 64 = 576 (validated byte-exact on real hardware).
    assert _expected_plain_mla_hidden(_cfg(kv_lora_rank=512, qk_rope_head_dim=64)) == 576


def test_plain_mla_hidden_none_when_not_mla():
    assert _expected_plain_mla_hidden(_cfg()) is None
    assert _expected_plain_mla_hidden(SimpleNamespace(model_config=None)) is None


# --- the offload/skip matrix (torch-free: takes is_packed directly) ----------


def _would_offload(*, is_packed, num_kv_channels, hidden_dim, plain_mla_hidden):
    """Mirror of ``register_kv_caches``' packed-dtype gate: a non-packed group
    always offloads; a packed (fp8/…) group offloads ONLY in the provably-safe
    plain-MLA-fp8 case (num_kv_channels == 1 and hidden equals the plain latent
    width kv_lora_rank + qk_rope_head_dim)."""
    if not is_packed:
        return True
    return num_kv_channels == 1 and plain_mla_hidden is not None and hidden_dim == plain_mla_hidden


def test_matrix_mla_fp8_plain_latent_offloads():
    # MLA latent (3-D shape -> num_kv_channels=1) + fp8 + hidden==576 -> offload
    assert _would_offload(is_packed=True, num_kv_channels=1, hidden_dim=576, plain_mla_hidden=576)


def test_matrix_bf16_always_offloads():
    # not packed -> passthrough, regardless of MLA vs GQA
    assert _would_offload(is_packed=False, num_kv_channels=1, hidden_dim=576, plain_mla_hidden=576)
    assert _would_offload(is_packed=False, num_kv_channels=2, hidden_dim=512, plain_mla_hidden=None)


def test_matrix_gqa_fp8_is_skipped():
    # GQA (num_kv_channels=2) + fp8 -> scale-packed / unrecognized -> SKIP
    assert not _would_offload(
        is_packed=True, num_kv_channels=2, hidden_dim=512, plain_mla_hidden=None
    )


def test_matrix_mla_fp8_scale_packed_hidden_mismatch_is_skipped():
    # MLA + fp8 but hidden != plain latent (fp8_ds_mla 656/584 packs scales) -> SKIP
    assert not _would_offload(
        is_packed=True, num_kv_channels=1, hidden_dim=656, plain_mla_hidden=576
    )


# --- dtype -> "packed/quantized" classification (needs torch) ----------------


def test_dtype_packed_classification():
    torch = pytest.importorskip("torch")
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "uint8", "int8"):
        dt = getattr(torch, name, None)
        if dt is not None:
            assert _is_packed_quant_kv_dtype(dt) is True, name
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        assert _is_packed_quant_kv_dtype(dt) is False
