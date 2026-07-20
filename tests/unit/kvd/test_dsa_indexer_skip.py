###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""kvd DSA (DeepSeek Sparse Attention) indexer-cache skip — glm_moe_dsa / deepseek_v32.

The DSA indexer registers as an MLAAttentionSpec (looks like a normal MLA latent), so the
shape-based path would offload it. It is an auxiliary recomputable index, NOT reusable KV, and
must be skipped from L3 offload. Detected by layer-name substring; strict no-op for non-DSA models.
"""

from infera.engine.vllm.kvd_connector import _is_dsa_indexer_group


def test_detects_dsa_indexer_layers():
    assert _is_dsa_indexer_group(["model.layers.5.self_attn.indexer.k_cache"]) is True
    assert _is_dsa_indexer_group(["a.indexer", "b.indexer"]) is True


def test_ignores_normal_and_recurrent_layers():
    # main MLA latent / regular attention / mamba — never matched (no-op for non-DSA models)
    assert _is_dsa_indexer_group(["model.layers.0.self_attn.attn"]) is False
    assert _is_dsa_indexer_group(["model.layers.0.mixer.mamba"]) is False
    assert _is_dsa_indexer_group([]) is False
    assert _is_dsa_indexer_group(None) is False
