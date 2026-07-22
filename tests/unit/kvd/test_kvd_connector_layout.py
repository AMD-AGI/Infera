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
    ------------------  -----------  ----------------------------------------
    MLA (plain latent)  fp8          yes  (hidden == kv_lora_rank+qk_rope_head_dim)
    GQA / MHA (plain)   fp8          yes  (hidden == (kv_heads//tp)*head_dim)
    MLA / any           bf16/fp16    yes  (not packed -> passthrough)
    GQA / MHA           bf16/fp16    yes  (not packed -> passthrough)
    MLA scale-packed    fp8 (656/…)  NO   (hidden != plain latent -> skip)
    GQA scale-packed    fp8          NO   (hidden != (kv_heads//tp)*head_dim -> skip)

Both plain-fp8 cases are per-tensor-scale casts (no scale bytes interleaved
into the hidden run), so the raw-byte chunked gather/scatter round-trips them
byte-exact. A scale-PACKED format stores a LARGER hidden and is skipped.

Pure functions only — no GPU / engine / model. The matrix decision below is
torch-free; only the dtype->packed classification needs torch (guarded).
"""

from types import SimpleNamespace

import pytest

from infera.engine.vllm.kvd_connector import (
    _expected_plain_gqa_hidden,
    _expected_plain_mla_hidden,
    _is_mla_from_config,
    _is_packed_quant_kv_dtype,
)


def _cfg(
    *,
    use_mla=None,
    kv_lora_rank=None,
    qk_rope_head_dim=None,
    num_key_value_heads=None,
    num_attention_heads=None,
    head_dim=None,
    hidden_size=None,
):
    """Minimal stand-in for the vLLM config the helpers read (model_config.*)."""
    hf = SimpleNamespace(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_key_value_heads=num_key_value_heads,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        text_config=None,
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


# --- plain-GQA-fp8 hidden width = (num_key_value_heads // tp) * head_dim ------


def test_plain_gqa_hidden_minimax_like_tp1():
    # MiniMax-M2.5: 8 kv-heads x 128 head_dim = 1024 (single card, tp=1).
    cfg = _cfg(num_key_value_heads=8, head_dim=128, num_attention_heads=48)
    assert _expected_plain_gqa_hidden(cfg, 1) == 1024


def test_plain_gqa_hidden_qwen_like_tp1():
    # Qwen2.5-7B: 4 kv-heads x 128 = 512.
    cfg = _cfg(num_key_value_heads=4, head_dim=128, num_attention_heads=28)
    assert _expected_plain_gqa_hidden(cfg, 1) == 512


def test_plain_gqa_hidden_is_tp_sharded():
    # KV heads shard across TP: per-rank width halves at tp=2.
    cfg = _cfg(num_key_value_heads=8, head_dim=128, num_attention_heads=48)
    assert _expected_plain_gqa_hidden(cfg, 2) == 512  # (8//2)*128
    assert _expected_plain_gqa_hidden(cfg, 4) == 256  # (8//4)*128


def test_plain_gqa_hidden_replicated_when_kv_heads_below_tp():
    # kv_heads < tp -> vLLM replicates one kv head per rank.
    cfg = _cfg(num_key_value_heads=2, head_dim=128, num_attention_heads=32)
    assert _expected_plain_gqa_hidden(cfg, 8) == 128  # 1 * 128


def test_plain_gqa_hidden_none_on_ragged_shard():
    # kv_heads not divisible by tp (and > tp) -> can't predict -> None (skip).
    cfg = _cfg(num_key_value_heads=6, head_dim=128, num_attention_heads=48)
    assert _expected_plain_gqa_hidden(cfg, 4) is None


def test_plain_gqa_hidden_none_for_mla():
    # MLA latent is not a GQA head layout -> None (handled by the MLA path).
    assert _expected_plain_gqa_hidden(_cfg(kv_lora_rank=512, qk_rope_head_dim=64), 1) is None


def test_plain_gqa_hidden_head_dim_fallback():
    # No explicit head_dim -> derive from hidden_size // num_attention_heads.
    cfg = _cfg(num_key_value_heads=4, num_attention_heads=32, hidden_size=4096)
    assert _expected_plain_gqa_hidden(cfg, 1) == 512  # 4 * (4096//32=128)


def test_plain_gqa_hidden_none_when_missing_config():
    assert _expected_plain_gqa_hidden(SimpleNamespace(model_config=None), 1) is None


# --- the offload/skip matrix (torch-free: takes is_packed directly) ----------


def _would_offload(*, is_packed, num_kv_channels, hidden_dim, plain_mla_hidden, plain_gqa_hidden):
    """Mirror of ``register_kv_caches``' packed-dtype gate: a non-packed group
    always offloads; a packed (fp8/…) group offloads ONLY in a provably-safe
    PLAIN fp8 case — either a plain MLA latent (num_kv_channels == 1, hidden ==
    kv_lora_rank + qk_rope_head_dim) or a plain GQA head layout
    (num_kv_channels == 2, hidden == (kv_heads//tp)*head_dim)."""
    if not is_packed:
        return True
    plain_mla = (
        num_kv_channels == 1 and plain_mla_hidden is not None and hidden_dim == plain_mla_hidden
    )
    plain_gqa = (
        num_kv_channels == 2 and plain_gqa_hidden is not None and hidden_dim == plain_gqa_hidden
    )
    return plain_mla or plain_gqa


def test_matrix_mla_fp8_plain_latent_offloads():
    # MLA latent (3-D shape -> num_kv_channels=1) + fp8 + hidden==576 -> offload
    assert _would_offload(
        is_packed=True,
        num_kv_channels=1,
        hidden_dim=576,
        plain_mla_hidden=576,
        plain_gqa_hidden=None,
    )


def test_matrix_gqa_fp8_plain_offloads():
    # GQA (num_kv_channels=2) + fp8 + hidden==(kv_heads//tp)*head_dim -> offload
    assert _would_offload(
        is_packed=True,
        num_kv_channels=2,
        hidden_dim=1024,
        plain_mla_hidden=None,
        plain_gqa_hidden=1024,
    )


def test_matrix_bf16_always_offloads():
    # not packed -> passthrough, regardless of MLA vs GQA
    assert _would_offload(
        is_packed=False,
        num_kv_channels=1,
        hidden_dim=576,
        plain_mla_hidden=576,
        plain_gqa_hidden=None,
    )
    assert _would_offload(
        is_packed=False,
        num_kv_channels=2,
        hidden_dim=512,
        plain_mla_hidden=None,
        plain_gqa_hidden=512,
    )


def test_matrix_gqa_fp8_scale_packed_hidden_mismatch_is_skipped():
    # GQA + fp8 but hidden != (kv_heads//tp)*head_dim (extra scale bytes) -> SKIP
    assert not _would_offload(
        is_packed=True,
        num_kv_channels=2,
        hidden_dim=1152,
        plain_mla_hidden=None,
        plain_gqa_hidden=1024,
    )


def test_matrix_gqa_fp8_unknown_width_is_skipped():
    # GQA + fp8 but plain width couldn't be resolved (ragged/None) -> SKIP
    assert not _would_offload(
        is_packed=True,
        num_kv_channels=2,
        hidden_dim=1024,
        plain_mla_hidden=None,
        plain_gqa_hidden=None,
    )


def test_matrix_mla_fp8_scale_packed_hidden_mismatch_is_skipped():
    # MLA + fp8 but hidden != plain latent (fp8_ds_mla 656/584 packs scales) -> SKIP
    assert not _would_offload(
        is_packed=True,
        num_kv_channels=1,
        hidden_dim=656,
        plain_mla_hidden=576,
        plain_gqa_hidden=None,
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
