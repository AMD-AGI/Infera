###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Qwen3.8-Flash-Next: 51.2B of its 180B parameters are n-gram embedding tables.

A lookup reads a few rows of them per token, so they cost no time, but a copy
holds all of them: a preset that dropped them would fit the model on a GPU
that cannot hold it.
"""

from __future__ import annotations

from .conftest import project_spec

QWEN = dict(
    model="qwen3_8_flash_next", tp=1, concurrency=4, weight_dtype="fp8", kv_cache_dtype="fp8"
)
NGRAM_PARAMS = 51_200_245_760


def test_ngram_tables_are_weights():
    full = project_spec(**QWEN)
    bare = project_spec(**QWEN, model_overrides=("ngram_embedding_params=0",))
    gib = (full["memory_gb"] - bare["memory_gb"]) * 1024.0**3
    assert abs(gib - NGRAM_PARAMS) / NGRAM_PARAMS < 0.02


def test_ngram_tables_cost_no_time():
    full = project_spec(**QWEN)
    bare = project_spec(**QWEN, model_overrides=("ngram_embedding_params=0",))
    assert full["decode_step_ms"] == bare["decode_step_ms"]
