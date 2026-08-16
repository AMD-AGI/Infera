###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reusing a warmup is only safe if the anchor describes what it measured.

The measure-once design turns on matching a deployment to a previous warmup, so
the ways that match can go wrong are the ways the whole thing goes wrong: an
anchor that mislabels its own dtype can never serve its deployment, and one that
is silently accepted for a different model produces a confident wrong answer.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.inference_projection.search.regime import (
    recipe_from_meta,
    regime_distance,
)


def test_a_natively_quantised_checkpoint_is_not_recorded_as_bf16():
    """An unset --quantization does not mean bf16.

    vLLM resolves the dtype from the checkpoint, and gpt-oss ships mxfp4 while
    DeepSeek-R1 ships fp8. Reading the unset flag as bf16 mislabelled those runs,
    so an anchor could not match the deployment it was measured for.
    """
    resolved = recipe_from_meta(
        {"model": "openai/gpt-oss-120b", "quantization": None, "weight_dtype": "mxfp4"}
    )
    assert resolved["weight_dtype"] == "mxfp4"

    target = dict(resolved, tp=8)
    assert regime_distance(resolved, target) == 0, (
        "an anchor must match the deployment it was measured for"
    )


def test_an_unrecorded_dtype_stays_unknown_rather_than_guessed():
    """Unknown is skipped when matching; a wrong guess forces a false mismatch."""
    old = recipe_from_meta({"model": "m", "quantization": None})
    assert old["weight_dtype"] is None
    # An explicit request still wins when there is no resolved dtype.
    assert recipe_from_meta({"quantization": "fp8"})["weight_dtype"] == "fp8"


def test_different_dtypes_are_still_different_regimes():
    """The guard has to keep working: mxfp4 and fp8 run different kernels."""
    a = recipe_from_meta({"model": "m", "weight_dtype": "mxfp4"})
    b = recipe_from_meta({"model": "m", "weight_dtype": "fp8"})
    assert regime_distance(a, b) >= 1


class _FakeStore:
    def __init__(self, models):
        self._models = models

    def entries(self):
        return [{"model": m} for m in self._models]


def test_a_multi_model_store_refuses_to_guess_which_model_to_use(monkeypatch):
    """Calibrating gpt-oss against DeepSeek's warmup must not be possible."""
    from infera.projection.core.projection.inference_projection.launcher import (
        _anchor_model_filter,
    )

    one = _FakeStore(["openai/gpt-oss-120b"])
    assert _anchor_model_filter(one) is None, "a single-model store needs no filter"

    many = _FakeStore(["openai/gpt-oss-120b", "deepseek-ai/DeepSeek-R1"])
    monkeypatch.setenv("INFERASIM_MODEL", "gpt_oss_120B")
    assert _anchor_model_filter(many) == "openai/gpt-oss-120b", (
        "the preset spelling should resolve against the artifact's id"
    )

    monkeypatch.delenv("INFERASIM_MODEL", raising=False)
    with pytest.raises(ValueError, match="could not be identified"):
        _anchor_model_filter(many)
