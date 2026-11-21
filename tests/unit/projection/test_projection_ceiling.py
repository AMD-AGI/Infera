###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The ceiling bound sits above every expected projection of the same deployment.

A better kernel, a fused one or full overlap can beat the expected projection;
none may beat the ceiling. These are inequalities between the two bounds, so
they keep holding as either gets more accurate.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.inference_projection.ceiling import (
    LEVERS,
    check_bound,
    parse_levers,
)
from infera.projection.core.projection.training_config import InferenceRequestConfig

from .conftest import project_spec

_ALL_FUSIONS = "add_rmsnorm,norm_quant,act_quant,rope_kvstore,moe_routing,moe_expert"
_DENSE = {"model": "qwen3_8B", "weight_dtype": "bf16", "tp": 4}
_MOE = {"model": "gpt_oss_120B", "weight_dtype": "mxfp4", "tp": 8, "ep": 8}


def _run(spec: dict, *flags) -> dict:
    return project_spec(**spec, extra_args=list(flags))


def _ceiling(spec: dict, levers: str | None = None) -> dict:
    flags = ["--bound", "ceiling"] + (["--ceiling-levers", levers] if levers else [])
    return _run(spec, *flags)


@pytest.mark.parametrize("spec", [_DENSE, _MOE], ids=["dense", "moe"])
def test_every_lever_only_speeds_the_step_up(spec):
    exp = _run(spec)
    full = _ceiling(spec)
    for lever in LEVERS:
        one = _ceiling(spec, lever)
        assert one["decode_step_ms"] <= exp["decode_step_ms"] + 1e-9, lever
        assert one["decode_step_ms"] >= full["decode_step_ms"] - 1e-9, lever
    assert full["decode_tps"] > exp["decode_tps"]
    assert full["ttft_ms"] <= exp["ttft_ms"]


def test_launch_ceiling_does_not_depend_on_measured_overheads():
    """Per-kernel occupancy is today's stack; the ceiling has none to charge."""
    slow, fast = ({**_DENSE, "decode_kernel_occupancy_us": us} for us in (20.0, 1.0))
    assert _run(slow)["decode_step_ms"] > _run(fast)["decode_step_ms"]
    assert _ceiling(slow, "launch")["decode_step_ms"] == pytest.approx(
        _ceiling(fast, "launch")["decode_step_ms"]
    )


def test_overlap_ceiling_is_full_overlap_for_a_dense_model():
    full = _run(_DENSE, "--decode-comm-overlap", "1", "--prefill-comm-overlap", "1")
    assert _ceiling(_DENSE, "overlap")["decode_step_ms"] == pytest.approx(full["decode_step_ms"])
    assert _ceiling(_DENSE, "overlap")["decode_step_ms"] < _run(_DENSE)["decode_step_ms"]


def test_fusion_ceiling_beats_every_named_fusion():
    """The named fusions keep some passes; the ceiling folds every one away.
    They also remove kernels, which the ceiling counts under ``launch``."""
    named = _run(_MOE, "--fusions", _ALL_FUSIONS)
    assert _ceiling(_MOE, "fusion")["decode_step_ms"] < _run(_MOE)["decode_step_ms"]
    assert _ceiling(_MOE, "fusion,launch")["decode_step_ms"] <= named["decode_step_ms"]


def test_headroom_report_stacks_to_the_ceiling_and_flags_a_measurement_above_it():
    res = _run(_MOE, "--headroom-measured", "1e9")
    rep = res["headroom"]
    assert rep["ceiling"]["output_tput_per_gpu"] >= rep["expected"]["output_tput_per_gpu"]
    assert rep["waterfall"][-1]["output_tput_per_gpu"] == pytest.approx(
        rep["ceiling"]["output_tput_per_gpu"], rel=1e-6
    )
    gains = [row["gain_pct"] for row in rep["levers"]]
    assert gains == sorted(gains, reverse=True)
    assert rep["measured"]["above_ceiling"] is True
    # The report leaves the projection it ran inside at the bound it was asked for.
    assert res["decode_step_ms"] == pytest.approx(_run(_MOE)["decode_step_ms"])


def test_ceiling_refuses_a_measured_anchor():
    req = InferenceRequestConfig(projection_bound="ceiling")
    with pytest.raises(ValueError, match="simulate mode"):
        check_bound(req, benchmark=True)
    check_bound(req, benchmark=False)
    check_bound(InferenceRequestConfig(), benchmark=True)


def test_bound_and_lever_names_are_checked():
    assert parse_levers("fusion, overlap,fusion") == ("fusion", "overlap")
    with pytest.raises(ValueError):
        parse_levers("overlap,kernels")
    with pytest.raises(ValueError):
        parse_levers("")
    with pytest.raises(ValueError):
        check_bound(InferenceRequestConfig(projection_bound="optimistic"), benchmark=False)
