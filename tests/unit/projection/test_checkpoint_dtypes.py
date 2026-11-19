###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Each op is priced at the format its checkpoint stores, not the precision label's.

The fixtures are the safetensors headers of real checkpoints, trimmed to the
tensors that decide every role (a layer or two, two experts, the globals).
"""

from __future__ import annotations

import json
import os
import struct
from types import SimpleNamespace

import pytest

from infera.projection.core.projection.inference_projection import checkpoint_dtypes as cd
from infera.projection.core.projection.inference_projection import gemm_shapes as gs
from infera.projection.core.projection.inference_projection import launcher
from infera.projection.core.projection.inference_projection import pareto as P
from infera.projection.core.projection.training_config import ModelConfig, ModelParallelConfig

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "checkpoint_headers")


def _fixture(name):
    with open(os.path.join(FIXTURES, f"{name}.json")) as f:
        return json.load(f)


def _classify(name):
    fx = _fixture(name)
    return cd.classify(map(tuple, fx["tensors"]), fx["quantization_config"], fx["checkpoint"])


def _write_checkpoint(tmp_path, name):
    """A checkpoint directory whose one shard has the fixture's header and no data."""
    fx = _fixture(name)
    header = {n: {"dtype": d, "shape": s, "data_offsets": [0, 0]} for n, d, s in fx["tensors"]}
    blob = json.dumps(header).encode()
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": fx["quantization_config"]})
    )
    return str(tmp_path)


def _save(tmp_path, ckpt):
    path = tmp_path / "dtypes.json"
    path.write_text(json.dumps(ckpt.to_json()))
    return str(path)


def test_gpt_oss_ships_mxfp4_experts_behind_bf16_attention_and_router():
    d = _classify("gpt_oss_120b")
    assert d.roles == {
        "attention": "bf16",
        "embedding": "bf16",
        "lm_head": "bf16",
        "routed_expert": "mxfp4",  # U8 blocks with U8 scales
        "router": "bf16",
    }
    assert d.fields() == {"linear_weight_dtype": "bf16", "moe_expert_dtype": "mxfp4"}
    assert d.unpriceable() == []


def test_deepseek_v4_keeps_bf16_compressors_and_fp32_mixers_inside_fp8_attention():
    d = _classify("deepseek_v4_pro")
    assert d.roles == {
        "attention": "fp8",
        "compressor": "bf16",
        "embedding": "bf16",
        "indexer_q": "fp8",
        "indexer_weights": "bf16",
        "lm_head": "bf16",
        "mixer": "fp32",
        "routed_expert": "mxfp4",  # I8 with E8M0 scales
        "router": "bf16",
        "shared_expert": "fp8",
    }
    assert d.fields() == {"linear_weight_dtype": "fp8", "moe_expert_dtype": "mxfp4"}


def test_reader_parses_headers_from_disk(tmp_path):
    d = cd.read(_write_checkpoint(tmp_path, "gpt_oss_120b"))
    assert d.roles == _classify("gpt_oss_120b").roles
    assert cd.main([str(tmp_path), "--out", str(tmp_path / "out.json")]) == 0
    again = cd.CheckpointDtypes.load(str(tmp_path / "out.json"))
    assert again.roles == d.roles and again.checkpoint == os.path.basename(str(tmp_path))


def test_packed_bytes_take_their_format_from_the_scales():
    t = [
        ("model.layers.0.mlp.experts.0.down_proj.weight", "U8", [64, 32]),
        ("model.layers.0.mlp.experts.0.down_proj.weight_scale", "F8_E4M3", [64, 4]),
    ]
    d = cd.classify(t)
    assert d.roles == {"routed_expert": "nvfp4"}
    assert d.unpriceable() == ["moe_expert_dtype=nvfp4 (InferaSim prices bf16, fp8, mxfp4)"]
    with pytest.raises(cd.CheckpointMismatch, match="cannot price"):
        cd.check_flags(d, {})


def test_flags_in_reads_both_spellings_and_the_last_wins():
    argv = ["--moe-expert-dtype", "fp8", "--linear-weight-dtype=bf16", "--moe-expert-dtype=mxfp4"]
    assert cd.flags_in(argv) == {"moe_expert_dtype": "mxfp4", "linear_weight_dtype": "bf16"}


def test_guard_names_the_op_and_both_formats():
    d = _classify("gpt_oss_120b")
    cd.check_flags(d, {"linear_weight_dtype": "bf16", "moe_expert_dtype": "fp4"})  # fp4 is mxfp4
    with pytest.raises(cd.CheckpointMismatch) as e:
        cd.check_flags(d, {"linear_weight_dtype": "mxfp4"})
    assert "linear_weight_dtype=mxfp4" in str(e.value)
    assert "stores attention as bf16" in str(e.value)
    assert "self_attn.k_proj.weight BF16" in str(e.value)


def _moe_cfg(roles, linear="fp8", expert="fp8"):
    mc = ModelConfig(
        num_layers=2,
        hidden_size=2880,
        padded_vocab_size=201088,
        ffn_hidden_size=2880,
        num_attention_heads=64,
        kv_channels=64,
        group_query_attention=True,
        num_query_groups=8,
        swiglu=True,
        num_experts=128,
        moe_router_topk=4,
        moe_ffn_hidden_size=2880,
        moe_pattern=[1, 1],
        moe_shared_expert_intermediate_size=2048,
    )
    mp = ModelParallelConfig(tensor_model_parallel_size=1)
    req = SimpleNamespace(
        linear_weight_dtype=linear, moe_expert_dtype=expert, checkpoint_dtypes=roles
    )
    return SimpleNamespace(model_config=mc, model_parallel_config=mp, request_config=req)


def test_kernel_shapes_take_each_role_from_the_checkpoint():
    roles = {
        "attention": "bf16",
        "routed_expert": "mxfp4",
        "router": "fp8",
        "shared_expert": "fp8",
        "lm_head": "fp8",
    }
    ops = {g.op: g for g, _ in gs.step_gemms(_moe_cfg(roles, linear="mxfp4"), 16, 16)}
    assert ops["attn_qkv"].dtype == ops["attn_o"].dtype == "bf16"
    assert ops["moe_gate_up"].dtype == ops["moe_down"].dtype == "mxfp4"
    assert ops["moe_router"].dtype == "fp8"
    assert ops["shared_gate_up"].dtype == "fp8"
    assert ops["lm_head"].dtype == "fp8"


def test_without_a_map_the_fields_and_bf16_router_stand():
    ops = {g.op: g for g, _ in gs.step_gemms(_moe_cfg(None, linear="fp8", expert="mxfp4"), 16, 16)}
    assert ops["attn_qkv"].dtype == ops["shared_gate_up"].dtype == "fp8"
    assert ops["moe_gate_up"].dtype == "mxfp4"
    assert ops["moe_router"].dtype == ops["lm_head"].dtype == "bf16"


def test_launcher_sets_the_cost_fields_and_refuses_a_contradicting_flag(tmp_path):
    path = _save(tmp_path, _classify("gpt_oss_120b"))
    got = launcher._collect_inference_overrides(SimpleNamespace(checkpoint_dtypes=path))
    assert got["linear_weight_dtype"] == "bf16" and got["moe_expert_dtype"] == "mxfp4"
    assert got["checkpoint_dtypes"]["router"] == "bf16"
    with pytest.raises(cd.CheckpointMismatch):
        launcher._collect_inference_overrides(
            SimpleNamespace(checkpoint_dtypes=path, linear_weight_dtype="mxfp4")
        )


def _point(weights, precision="mxfp4", extra=()):
    return P.point_argv(
        scenario=P.Scenario("1k1k", 1024, 1024),
        trace_path=None,
        precision=precision,
        engine="vllm",
        par=P.Parallelism(1),
        concurrency=16,
        gpu_arch="mi355x",
        hbm_gb=288.0,
        num_requests=160,
        anchor=None,
        checkpoint="gpt-oss-120b",
        extra=extra,
        weights=weights,
    )


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_pareto_prices_the_checkpoint_not_the_precision_label(tmp_path):
    ckpt = _classify("gpt_oss_120b")
    ckpt.path = _save(tmp_path, ckpt)
    argv = _point(ckpt)
    assert _flag(argv, "--linear-weight-dtype") == "bf16"  # the "mxfp4" label said mxfp4
    assert _flag(argv, "--weight-dtype") == "bf16"
    assert _flag(argv, "--moe-expert-dtype") == "mxfp4"
    assert _flag(argv, "--kv-cache-dtype") == "fp8"  # a runtime choice, still the label's
    assert _flag(argv, "--checkpoint-dtypes") == ckpt.path
    plain = _point(None)
    assert _flag(plain, "--linear-weight-dtype") == "mxfp4" and "--checkpoint-dtypes" not in plain


def test_pareto_accepts_an_agreeing_flag_and_refuses_a_contradicting_one(tmp_path):
    ckpt = _classify("deepseek_v4_pro")
    _point(ckpt, precision="fp8", extra=["--moe-expert-dtype=mxfp4"])
    with pytest.raises(cd.CheckpointMismatch, match="moe_expert_dtype=fp8"):
        _point(ckpt, precision="fp8", extra=["--moe-expert-dtype", "fp8"])


def test_pareto_main_exits_on_a_contradiction(tmp_path, capsys):
    path = _save(tmp_path, _classify("gpt_oss_120b"))
    rc = P.main(
        [
            "--model",
            "gpt_oss_120b",
            "--engine",
            "vllm",
            "--precision",
            "mxfp4",
            "--parallelism",
            "tp1",
            "--checkpoint-dtypes",
            path,
            "--out-dir",
            str(tmp_path / "out"),
            "--dry-run",
            "--",
            "--linear-weight-dtype=mxfp4",
        ]
    )
    assert rc == 2
    assert "stores attention as bf16" in capsys.readouterr().err
