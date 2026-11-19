###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Op bring-up: context pack, coder retry on validator feedback, acknowledge path, evaluator hook."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from infera.projection.agents.tuning_agent import op_bringup as ob  # noqa: E402
from infera.projection.agents.tuning_agent.evaluator import EvalResult, Evaluator  # noqa: E402
from infera.projection.core.projection.op_models.coverage import OpGap  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
_WORKLOAD = _ROOT / "infera/projection/examples/workloads/lfm2_8B_A1B-BF16-serving.yaml"
_LFM2 = _ROOT / "infera/projection/core/projection/op_models/generated/lfm2_short_conv.py"


@pytest.fixture(autouse=True)
def _workload_default_model(monkeypatch):
    monkeypatch.delenv("INFERASIM_MODEL", raising=False)


def _gap(**keys):
    return OpGap(op_id="test_op", severity="gap", slot="attention", reason="unit test", keys=keys)


def _variant(miscount: bool) -> str:
    src = _LFM2.read_text().replace('name = "lfm2_short_conv"', 'name = "conv_bringup_test"')
    src = src.replace("class Lfm2ShortConv(", "class ConvBringupTest(")
    if miscount:
        src = src.replace(
            "        return 2.0 * m * 4 * h * h", "        return 6.0 * m * 4 * h * h"
        )
    return "Here is the module.\n```python\n" + src + "```\n"


def test_reply_parsing():
    assert ob.extract_module("```python\nclass A(OpModel):\n    pass\n```").startswith("class A")
    assert ob.extract_module("no code here") is None
    assert ob.parse_acknowledgements(
        "ACKNOWLEDGE a_key: tiny\nnoise\nACKNOWLEDGE b: 1% of layer"
    ) == {
        "a_key": "tiny",
        "b": "1% of layer",
    }


def test_context_pack_has_the_pieces(monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: '{"model_type": "lfm2_moe"}')
    from infera.projection.agents.tuning_agent.workload import _find_config_root, resolve_workload

    arch = resolve_workload(_WORKLOAD, config_root=_find_config_root(_WORKLOAD))
    text = ob.build_context(
        _gap(lfm_layer_types=["conv", "full_attention"]),
        model_path=Path(arch.model_path),
        raw_model=arch.raw_model,
        papers=["y = C * conv(B * x)"],
    )
    for needle in (
        "## The gap",
        "lfm2_8B_A1B.yaml",
        "lfm2_moe",
        "y = C * conv",
        "class OpContext",
        "generated/lfm2_short_conv.py",
        "_simulate_tile_level",
        "is_decode",
        "ACKNOWLEDGE",
    ):
        assert needle in text, needle


def test_failed_validation_is_fed_back_then_the_fix_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    prompts = []

    def coder(prompt, target):
        prompts.append(prompt)
        return _variant(miscount=len(prompts) == 1)

    res = ob.bring_up_op(
        _gap(lfm_layer_types=["conv"]),
        workload=_WORKLOAD,
        out_dir=tmp_path,
        coder=coder,
        gpu_arch="mi355x",
        register=False,
        log=lambda s: None,
    )
    assert res.status == "registered" and res.attempts == 2, res.message
    assert res.op_name == "conv_bringup_test"
    assert "Feedback on your previous attempt" in prompts[1] and "reference_flops" in prompts[1]
    assert (tmp_path / "op_bringup/test_op/report.md").is_file()
    assert not (_LFM2.parent / "conv_bringup_test.py").exists()


def test_acknowledge_must_cover_exactly_the_gap_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    replies = iter(
        ["ACKNOWLEDGE unrelated: x", "ACKNOWLEDGE fancy_gate: hidden->1 GEMV, <1% of layer"]
    )
    res = ob.bring_up_op(
        _gap(fancy_gate=True),
        workload=_WORKLOAD,
        out_dir=tmp_path,
        coder=lambda p, t: next(replies),
        register=False,
        log=lambda s: None,
    )
    assert res.status == "acknowledged" and res.attempts == 2
    assert res.acknowledged == {"fancy_gate": "hidden->1 GEMV, <1% of layer"}


def test_no_coder_leaves_a_context_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "fetch_hf_config", lambda *a, **k: None)
    res = ob.bring_up_op(
        _gap(x_gate=True), workload=_WORKLOAD, out_dir=tmp_path, coder=None, log=lambda s: None
    )
    assert res.status == "context_only" and Path(res.context_path).is_file()


def test_inference_evaluator_reruns_once_after_new_op_models():
    ev = Evaluator.__new__(Evaluator)
    ev.op_gap_handler, ev.handled_op_gaps, ev.op_bringup = None, set(), None
    gap = {
        "op_id": "new_mixer",
        "severity": "gap",
        "slot": "attention",
        "reason": "r",
        "keys": {"k": 1},
    }
    calls = []

    def fake_eval(cfg, tag):
        calls.append(tag)
        return EvalResult(op_gaps=[gap] if len(calls) == 1 else [], ttft_ms=float(len(calls)))

    ev._evaluate_inference = fake_eval
    handled = []
    ev.op_gap_handler = lambda gaps: handled.append([g["op_id"] for g in gaps]) or True
    r = ev.evaluate_inference(None, "t0")
    assert calls == ["t0", "t0_ops"] and handled == [["new_mixer"]] and r.ttft_ms == 2.0
    calls.clear()
    ev.evaluate_inference(None, "t1")  # same gap again: no second bring-up
    assert calls == ["t1"] and len(handled) == 1
