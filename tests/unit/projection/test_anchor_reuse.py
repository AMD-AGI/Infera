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

from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection.search.regime import (
    aiter_ops_axis,
    recipe_from_inference_config,
    recipe_from_meta,
    regime_distance,
    regime_signature,
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
    a = recipe_from_meta(
        {"model": "m", "aiter_ops": aiter_ops_axis(dict(env, VLLM_ROCM_USE_AITER_MHA="0"))}
    )
    b = recipe_from_meta(
        {"model": "m", "aiter_ops": aiter_ops_axis(dict(env, VLLM_ROCM_USE_AITER_MOE="0"))}
    )
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


def test_two_engines_measuring_one_checkpoint_are_two_regimes():
    """An SGLang anchor does not describe vLLM, and the axes have to say so.

    The engine was named as regime-defining in the design split from the start
    but was missing from ``REGIME_AXES``, so every engine's anchor for a given
    checkpoint hashed to one signature: the bench cache would hand a vLLM
    measurement to an SGLang run, and ``regime_distance`` called the pair a
    perfect match.
    """
    vllm = recipe_from_meta({"model": "m"}, engine="vllm")
    sglang = recipe_from_meta({"model": "m"}, engine="sglang")

    assert regime_distance(vllm, sglang) >= 1
    assert regime_signature(vllm) != regime_signature(sglang)


def test_a_rebuilt_engine_does_not_inherit_the_engine_it_forked_from():
    """``mori-sglang`` runs SGLang's launch script and is not SGLang.

    It is the case that motivates the axis: the two share a recipe script, and
    their reported KV pools differ 16x because one shards the MLA latent across
    ranks where the other replicates it. Keyed on anything coarser than the
    engine string, the fork silently inherits the parent's anchor.
    """
    parent = recipe_from_meta({"model": "m"}, engine="sglang")
    fork = recipe_from_meta({"model": "m"}, engine="mori-sglang")

    assert regime_distance(parent, fork) >= 1


def test_the_engine_is_read_off_the_artifact_not_only_the_meta_block():
    """``benchmark_serving`` records the backend beside ``meta``, not inside it.

    So the store passes it explicitly. An artifact that does record it inline
    is still honoured, which is what lets a future harness stop threading it.
    """
    assert recipe_from_meta({"model": "m"}, engine="atom")["engine"] == "atom"
    assert recipe_from_meta({"model": "m", "backend": "atom"})["engine"] == "atom"
    assert recipe_from_meta({"model": "m", "engine": "atom"})["engine"] == "atom"


def test_an_anchor_indexed_before_the_axis_existed_still_matches():
    """Unknown stays skippable, or adding the axis would orphan the store.

    Every artifact already in the store predates the axis. Counting unknown as
    a difference would refuse all of them at once, which is a worse failure
    than the one being fixed -- and the engine filter in the harness's own
    ``resolve`` is what actually gates reuse today.
    """
    old = recipe_from_meta({"model": "m"})
    assert old["engine"] is None
    assert regime_distance(old, recipe_from_meta({"model": "m"}, engine="vllm")) == 0


def test_a_projection_names_the_engine_it_is_asking_about():
    """The target side is unknown unless ``--serving-engine`` says otherwise.

    A projection has no engine of its own, so without this the axis could never
    bite on the path that actually consumes anchors.
    """
    from infera.projection.core.projection.training_config import InferenceRequestConfig

    cfg = SimpleNamespace(
        request_config=InferenceRequestConfig(serving_engine="sglang"),
        model_config=None,
        model_parallel_config=None,
    )
    assert recipe_from_inference_config(cfg)["engine"] == "sglang"
    assert InferenceRequestConfig().serving_engine is None


def test_naming_an_engine_does_not_change_what_simulate_projects():
    """It is a matching key, not a cost input.

    Simulate mode prices an architecture on a device analytically; if naming an
    engine moved the number, the axis would be quietly inventing engine
    performance it has no measurement for.
    """
    from infera.projection.core.projection.training_config import InferenceRequestConfig

    base = InferenceRequestConfig()
    named = InferenceRequestConfig(serving_engine="sglang")
    assert (
        base.resolved_attention_backend_multiplier()
        == named.resolved_attention_backend_multiplier()
    )


def test_the_store_rehashes_rather_than_trusting_a_recorded_signature(tmp_path):
    """A signature in ``meta`` was hashed over whichever axes that build had.

    Preferring it pins the store to an older definition of "same regime", so
    adding an axis cannot separate anchors that the older one merged -- which
    is what happened to seven DeepSeek-V4-Flash anchors sharing one recorded
    signature across SGLang and vLLM.
    """
    import json

    from infera.projection.core.projection.inference_projection.search.anchor_store import (
        AnchorStore,
    )

    paths = []
    for engine in ("sglang", "vllm"):
        p = tmp_path / f"{engine}.json"
        p.write_text(
            json.dumps({
                "backend": engine,
                # The same stale hash on both, as the real artifacts carry.
                "meta": {"model": "m", "regime_signature": "staleaaaaaaaaaaa"},
                "sweep": [{"batch": 1, "decode_ms": 1.0}],
            })
        )
        paths.append(str(p))

    store = AnchorStore(str(tmp_path / "store"))
    sigs = [store.add_artifact(p)["regime_signature"] for p in paths]

    assert "staleaaaaaaaaaaa" not in sigs, "a recorded signature is provenance, not the key"
    assert sigs[0] != sigs[1], "two engines must not land in one regime"


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


def test_a_target_may_name_its_checkpoint_when_the_preset_does_not(monkeypatch, tmp_path):
    """DeepSeek-R1's own anchor must be usable for a DeepSeek-R1 projection.

    R1 runs on the V3 architecture preset, so ``INFERASIM_MODEL`` says
    "deepseek_v3" while the artifact says "deepseek-ai/DeepSeek-R1" and identity
    by preset alone refused R1's own measurement for R1's own projection. The
    answer is not to let the architecture stand in for the checkpoint -- an
    anchor measures expert routing, which is weights -- but to let the target
    name its weights, which ``--bench-model`` already does elsewhere.
    """
    from infera.projection.core.projection.inference_projection.launcher import (
        _anchor_model_filter,
        _assert_anchor_is_this_model,
    )

    artifact = {"meta": {"model": "deepseek-ai/DeepSeek-R1"}}
    monkeypatch.setenv("INFERASIM_MODEL", "deepseek_v3")
    monkeypatch.delenv("INFERASIM_BENCH_MODEL", raising=False)
    monkeypatch.delenv("INFERASIM_ALLOW_FOREIGN_ANCHOR", raising=False)

    with pytest.raises(ValueError, match="was measured on"):
        _assert_anchor_is_this_model("a.json", artifact)

    _assert_anchor_is_this_model(
        "a.json", artifact, SimpleNamespace(bench_model="deepseek-ai/DeepSeek-R1")
    )
    monkeypatch.setenv("INFERASIM_BENCH_MODEL", "deepseek-ai/DeepSeek-R1")
    _assert_anchor_is_this_model("a.json", artifact)

    # Naming the checkpoint widens nothing else: a foreign anchor is still refused.
    with pytest.raises(ValueError, match="was measured on"):
        _assert_anchor_is_this_model("b.json", {"meta": {"model": "openai/gpt-oss-120b"}})

    # The store filter resolves against the same identity.
    many = _FakeStore(["openai/gpt-oss-120b", "deepseek-ai/DeepSeek-R1"])
    assert _anchor_model_filter(many) == "deepseek-ai/DeepSeek-R1"


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


def _artifact(tmp_path, name, *, tp=8, ep=1, sweep=True, sub=None):
    doc = {
        "meta": {
            "model": "openai/gpt-oss-120b",
            "tp": tp,
            "ep": ep,
            "weight_dtype": "mxfp4",
        },
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


@pytest.mark.parametrize(
    "target_tp,target_ep,expected",
    [
        (1, 1, "tp1.json"),
        (2, 1, "tp2.json"),
        (2, 2, "tp2ep2.json"),
        (4, 1, "tp4.json"),
        (8, 1, "tp4.json"),
        # The TP4 policy takes precedence over crossing to a narrower EP anchor.
        (8, 8, "tp4.json"),
        (16, 1, "tp4.json"),
    ],
)
def test_anchor_tp_policy_uses_tp4_at_four_and_above(tmp_path, target_tp, target_ep, expected):
    _artifact(tmp_path, "tp1.json", tp=1)
    _artifact(tmp_path, "tp2.json", tp=2)
    _artifact(tmp_path, "tp2ep2.json", tp=2, ep=2)
    _artifact(tmp_path, "tp4.json", tp=4)
    _artifact(tmp_path, "tp8.json", tp=8)

    store = AnchorStore(str(tmp_path))
    entry, distance = store.nearest({"tp": target_tp, "ep": target_ep, "weight_dtype": "mxfp4"})

    assert distance == 0
    assert os.path.basename(entry["path"]) == expected


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


def test_an_anchor_from_another_part_is_not_the_same_regime():
    """gfx950 timings do not describe a gfx942 deployment.

    Nothing in the store recorded which accelerator produced a timing, so the
    axis could not be compared and every anchor matched every target. The one
    place that bit in this matrix is GLM-5.2: its only anchors are the
    MXFP4 Quark build, which loads on gfx950 alone, and they were being used
    to price an MI325X deployment -- a part with roughly half the compute.
    It read 4.3x the measured throughput while reporting itself calibrated.
    """
    anchor = recipe_from_meta(
        {"model": "/models/GLM-5.2-MXFP4", "gpu_arch": "mi355x"}, engine="sglang"
    )
    assert anchor["gpu_arch"] == "mi355x"
    assert regime_distance(anchor, dict(anchor, gpu_arch="mi325x")) == 1
    assert regime_distance(anchor, dict(anchor, gpu_arch="mi355x")) == 0


def test_an_anchor_that_never_recorded_its_part_is_still_usable():
    """Adding the axis must not retire every anchor harvested before it.

    An unrecorded part is unknown rather than wrong, and ``regime_distance``
    skips an axis neither side pins. The store is what closes the gap, by
    stamping the part at build time for the pairs whose part can be argued
    for -- so silence here means "nobody has said yet".
    """
    legacy = recipe_from_meta({"model": "/models/DeepSeek-V4-Pro"}, engine="vllm")
    assert legacy["gpu_arch"] is None
    assert regime_distance(legacy, dict(legacy, gpu_arch="mi300x")) == 0
