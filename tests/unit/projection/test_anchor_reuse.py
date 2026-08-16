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


# ---------------------------------------------------------------------------
# Finding the anchors at all.
#
# A store that cannot see a measurement still answers -- just uncalibrated, and
# without saying so. On an engine the model was not fitted to that is the
# difference between 17.6% error and 2.3%, which makes a silently empty store
# the most expensive failure in this path rather than a cosmetic one.
# ---------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402

from infera.projection.core.projection.inference_projection.search.anchor_store import (  # noqa: E402
    AnchorStore,
)


def _artifact(tmp_path, name, *, tp=8, sweep=True, sub=None):
    doc = {
        "meta": {"model": "openai/gpt-oss-120b", "tp": tp, "weight_dtype": "mxfp4"},
        "sweep": [{"batch": 1, "decode_ms": 3.7}] if sweep else [],
    }
    d = tmp_path / sub if sub else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(doc))
    return d / name


def test_a_directory_of_measurements_is_not_empty_just_because_nobody_indexed_it(tmp_path):
    """The store reads its own directory instead of trusting a manifest.

    Hyperloom points the store at a warmup output directory and never calls
    add_artifact, so before discovery every projection there ran uncalibrated
    while reporting success.
    """
    _artifact(tmp_path, "real_tp8.json")
    _artifact(tmp_path, "real_tp4.json", tp=4, sub="run-2")

    store = AnchorStore(str(tmp_path))
    assert len(store.entries()) == 2, "artifacts at the root and one level down"
    assert os.path.exists(store.index_path), "discovery should persist what it found"


def test_discovery_is_idempotent_and_does_not_duplicate(tmp_path):
    _artifact(tmp_path, "real_tp8.json")
    first = AnchorStore(str(tmp_path))
    assert len(first.entries()) == 1
    assert AnchorStore(str(tmp_path)).discover() == 0
    assert len(AnchorStore(str(tmp_path)).entries()) == 1


def test_json_that_is_not_a_measurement_is_left_alone(tmp_path):
    """A store root is full of reports and configs; only curves can anchor.

    Indexing one of those yields an anchor with nothing to calibrate against,
    which then wins a lookup and degrades the answer with no way to trace it
    back here.
    """
    (tmp_path / "benchmark_report.json").write_text(json.dumps({"success": True}))
    (tmp_path / "config.json").write_text(json.dumps({"tp": 8}))
    (tmp_path / "broken.json").write_text("{not json")
    _artifact(tmp_path, "empty_sweep.json", sweep=False)
    good = _artifact(tmp_path, "real_tp8.json")

    entries = AnchorStore(str(tmp_path)).entries()
    assert [e["path"] for e in entries] == [str(good)]


def test_discovery_stops_before_walking_an_entire_filesystem(tmp_path):
    """A store root pointed somewhere broad should cost a bounded walk."""
    deep = "a/b/c/d/e"
    _artifact(tmp_path, "far.json", sub=deep)
    assert AnchorStore(str(tmp_path)).entries() == []
