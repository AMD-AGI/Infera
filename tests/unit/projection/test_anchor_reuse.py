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
    aiter_ops_axis,
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


def test_turning_off_one_aiter_kernel_is_a_different_regime():
    """The master switch alone does not say which kernels ran.

    A recipe can leave ``VLLM_ROCM_USE_AITER=1`` on and still turn off a single
    kernel family, so keying on the master switch made every such recipe hash
    equal to base -- and an anchor measured on base was handed back for a
    deployment that never runs those kernels.
    """
    on = {"VLLM_ROCM_USE_AITER": "1"}
    mha_off = dict(on, VLLM_ROCM_USE_AITER_MHA="0")
    assert aiter_ops_axis(on) == "default"
    assert aiter_ops_axis(mha_off) == "mha=0"

    base = recipe_from_meta({"model": "m", "aiter_ops": aiter_ops_axis(on)})
    swapped = recipe_from_meta({"model": "m", "aiter_ops": aiter_ops_axis(mha_off)})
    assert regime_distance(base, swapped) >= 1


def test_two_different_kernel_swaps_do_not_share_an_anchor():
    """Turning off MHA and turning off MoE are not the same measurement."""
    env = {"VLLM_ROCM_USE_AITER": "1"}
    a = recipe_from_meta({"model": "m",
                          "aiter_ops": aiter_ops_axis(dict(env, VLLM_ROCM_USE_AITER_MHA="0"))})
    b = recipe_from_meta({"model": "m",
                          "aiter_ops": aiter_ops_axis(dict(env, VLLM_ROCM_USE_AITER_MOE="0"))})
    assert regime_distance(a, b) >= 1


def test_a_switch_added_later_is_tracked_without_a_code_change():
    """Every ``VLLM_ROCM_USE_AITER_*`` var is folded in, not a fixed list."""
    invented = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_SOMETHING_NEW": "0"}
    assert aiter_ops_axis(invented) == "something_new=0"


def test_an_unrecorded_op_set_matches_defaults_but_not_a_swap():
    """Unknown is not the same as "no overrides".

    Artifacts predating this tracking ran on the default kernel set, so unknown
    may serve a target that is also on defaults. It may not serve one that swaps
    a kernel: that would be the original bug, reintroduced through the old
    artifacts.
    """
    old = recipe_from_meta({"model": "m"})
    assert old["aiter_ops"] is None

    env = {"VLLM_ROCM_USE_AITER": "1"}
    defaults = recipe_from_meta({"model": "m", "aiter_ops": aiter_ops_axis(env)})
    swapped = recipe_from_meta(
        {"model": "m", "aiter_ops": aiter_ops_axis(dict(env, VLLM_ROCM_USE_AITER_MOE="0"))}
    )
    assert regime_distance(old, defaults) == 0
    assert regime_distance(old, swapped) >= 1


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
# without saying so. On an engine the model was not fitted to, calibration is
# the difference between a large error and a small one, which makes a silently
# empty store the most expensive failure in this path rather than a cosmetic
# one.
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


@pytest.mark.parametrize(
    "target, artifact, same",
    [
        # A local mount and an org prefix are provenance, not identity.
        ("/models/DeepSeek-R1", "deepseek-ai/DeepSeek-R1", True),
        ("/models/gpt-oss-120b", "openai/gpt-oss-120b", True),
        # A preset drops decoration the id keeps.
        ("gpt_oss_120B", "openai/gpt-oss-120b", True),
        ("qwen3_8B", "Qwen/Qwen3-8B", True),
        # Different checkpoints stay different, including same-family ones: an
        # anchor measures routing behaviour, which is weights, not architecture.
        ("deepseek_v3", "deepseek-ai/DeepSeek-R1", False),
        ("/models/Qwen3-8B", "Qwen/Qwen3-30B-A3B", False),
        ("/models/qwen3_8B", "openai/gpt-oss-120b", False),
        # An unnamed side cannot be claimed to match.
        ("", "openai/gpt-oss-120b", False),
    ],
)
def test_which_names_refer_to_the_same_model(target, artifact, same):
    """One definition of model identity, shared by Infera and the bridge.

    Both sides had their own and the bridge's was inert: it compared a checkout
    path to a HuggingFace id for equality, which never holds, so every workload
    matched nothing and then fell back to using any anchor at all.
    """
    from infera.projection.core.projection.inference_projection.search.regime import (
        models_match,
    )

    assert models_match(target, artifact) is same
    assert models_match(artifact, target) is same, "matching is symmetric"


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
