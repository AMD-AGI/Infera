###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Architectural features declared by a model preset must reach the projection.

Module configs merge over model configs, so a key the module also declares wins
even when the model sets it deliberately. That silently reverted gpt-oss to full
attention -- its KV cache and long-context attention cost were projected at
roughly twice the truth -- and it is invisible in the output, since the
projection simply reports a plausible-looking dense number.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec

GPT_OSS_WINDOW = 128


@pytest.mark.parametrize("model", ["gpt_oss_120B", "gpt_oss_20B"])
def test_gpt_oss_window_survives_the_module_merge(model):
    assert project_spec(model=model)["sliding_window"] == GPT_OSS_WINDOW


def test_gpt_oss_has_no_shared_expert():
    """Inherited from the DeepSeek-V2 base config; the real model has none."""
    assert not project_spec()["shared_expert_size"]


def test_window_halves_the_long_context_kv_cache():
    """Every other layer is windowed to 128 tokens, so at 64k it holds ~nothing."""
    windowed = project_spec(input_len=65536, output_len=8, concurrency=8)
    dense = project_spec(input_len=65536, output_len=8, concurrency=8, sliding_window=0)
    ratio = windowed["kv_cache_gb"] / dense["kv_cache_gb"]
    assert 0.45 < ratio < 0.55, f"windowed KV is {ratio:.2f} of dense, expected ~0.5"


def test_window_is_a_no_op_inside_the_window():
    """A context that never leaves the window costs the same as full attention."""
    windowed = project_spec(input_len=64, output_len=8, concurrency=8)
    dense = project_spec(input_len=64, output_len=8, concurrency=8, sliding_window=0)
    assert windowed["kv_cache_gb"] == pytest.approx(dense["kv_cache_gb"], rel=1e-6)
