###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A model file describes the checkpoint it is named for.

A field a model file leaves out falls back to its base file or a default, and
nothing fails: the replay just runs a slightly different model. Missing
vocabularies gave every LM head 100352 columns, and an inherited QK norm gave
gpt-oss a kernel it doesn't run. Each model file is checked against its
checkpoint's ``config.json`` where that checkpoint is on the host.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from infera.projection.core.config.preset_loader import PresetLoader
from infera.projection.core.projection.training_config import padded_vocab_size

CHECKPOINTS = os.environ.get("INFERA_CHECKPOINT_ROOT", "/shared_nfs/hyperloom/models")

PAIRS = {
    "llama3.1_8B": "Llama-3.1-8B-Instruct",
    "qwen3_8B": "Qwen3-8B",
    "qwen3_14B": "Qwen3-14B-FP8",
    "gpt_oss_120B": "gpt-oss-120b",
    "mixtral_8x7B_v0.1": "Mixtral-8x7B-Instruct-v0.1",
    "deepseek_v4_pro": "DeepSeek-V4-Pro",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "minimax_m3": "MiniMax-M3-MXFP4",
}

FIELDS = {
    "vocab_size": ("vocab_size",),
    "hidden_size": ("hidden_size",),
    "num_layers": ("num_hidden_layers",),
    "num_attention_heads": ("num_attention_heads",),
    "num_query_groups": ("num_key_value_heads",),
    "num_experts": ("num_local_experts", "n_routed_experts", "num_experts"),
    "moe_router_topk": ("num_experts_per_tok", "experts_per_token"),
}

QK_NORM = {
    "LlamaForCausalLM": False,
    "MixtralForCausalLM": False,
    "GptOssForCausalLM": False,
    "Qwen3ForCausalLM": True,
    "Qwen3MoeForCausalLM": True,
}


def _hf(name):
    path = os.path.join(CHECKPOINTS, name, "config.json")
    if not os.path.exists(path):
        pytest.skip(f"checkpoint {name} is not on this host")
    with open(path) as f:
        cfg = json.load(f)
    return cfg if "hidden_size" in cfg else cfg.get("text_config", cfg)


@pytest.mark.parametrize("preset,checkpoint", sorted(PAIRS.items()))
def test_model_file_matches_checkpoint(preset, checkpoint):
    hf = _hf(checkpoint)
    model = PresetLoader.load(preset, framework="megatron")
    wrong = {}
    for field, keys in FIELDS.items():
        want = next((hf[k] for k in keys if k in hf), None)
        if want is None or (field == "num_query_groups" and model.get("multi_latent_attention")):
            continue
        if model.get(field) != want:
            wrong[field] = (model.get(field), want)
    arch = (hf.get("architectures") or [None])[0]
    if arch in QK_NORM and bool(model.get("qk_layernorm")) != QK_NORM[arch]:
        wrong["qk_layernorm"] = (model.get("qk_layernorm"), QK_NORM[arch])
    assert not wrong, f"{preset} vs {checkpoint} (model file, checkpoint): {wrong}"


@pytest.mark.parametrize(
    "vocab,multiple,padded",
    [(201088, 128, 201088), (129280, 64, 129280), (50257, 128, 50304), (None, 128, 100352)],
)
def test_vocab_padded_from_the_model(vocab, multiple, padded):
    args = SimpleNamespace(vocab_size=vocab, make_vocab_size_divisible_by=multiple)
    assert padded_vocab_size(args) == padded
