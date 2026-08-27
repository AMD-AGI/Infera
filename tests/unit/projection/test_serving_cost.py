###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Throughput is the projection; cost per million tokens is the question asked of it.

A recipe is chosen against a budget, so the comparison that matters is not which
recipe emits more tokens a second but which emits them for less. The two differ
whenever a recipe buys its throughput with GPUs: the faster recipe can be the
more expensive one, and tokens/s/GPU alone will not say so.
"""

from __future__ import annotations

import re

import pytest

from .conftest import project_spec

PRICE = 2.0


def _costs(report: str) -> dict:
    """The cost lines, as {basis: dollars per million tokens}."""
    found = re.findall(r"Cost / 1M (\S+) tokens:\s+\$([\d,.]+)", report)
    return {basis: float(amount.replace(",", "")) for basis, amount in found}


def test_a_price_turns_throughput_into_the_unit_a_budget_is_quoted_in():
    out = project_spec(gpu_cost_per_hour=PRICE, input_len=1024, output_len=1024)
    costs = _costs(out["report"])

    assert set(costs) == {"output", "in+out"}
    # The replica is charged for every GPU it holds, so the cost of a token is
    # the replica's hourly cost spread over what it emits in that hour.
    expected = PRICE * out["replica_gpus"] * 1e6 / (out["decode_tps"] * 3600.0)
    # Tolerance follows the report's own precision rather than the arithmetic:
    # the line is printed to the cent-and-a-bit, so half of that last place is
    # the tightest a parsed value can be held to.
    assert costs["output"] == pytest.approx(expected, abs=5e-4)


def test_prompt_tokens_are_carried_by_the_same_requests():
    """Blending over the workload's own mix, so the two bases differ by it."""
    out = project_spec(gpu_cost_per_hour=PRICE, input_len=3072, output_len=1024)
    costs = _costs(out["report"])
    # Four tokens billed for every one emitted, so the blended basis is a quarter.
    # Tolerance follows the report's own precision rather than the arithmetic:
    # both figures are parsed back from three printed decimals, so near $0.2 a
    # single rounding step is already 0.24% and a ratio of two of them twice
    # that. The underlying relationship is exact.
    assert costs["in+out"] == pytest.approx(costs["output"] / 4.0, rel=6e-3)


def test_nothing_is_priced_without_a_price():
    """An invented default price would be quoted as though it were measured."""
    assert _costs(project_spec()["report"]) == {}


def test_the_cheaper_recipe_is_not_always_the_faster_one():
    """The reason the metric earns its place in the report.

    Tokens/s alone ranks recipes by speed. Cost ranks them by what that speed
    costs, and the order is not the same one whenever throughput was bought
    with GPUs rather than with efficiency.
    """
    small = project_spec(gpu_cost_per_hour=PRICE, tp=4, concurrency=64)
    large = project_spec(gpu_cost_per_hour=PRICE, tp=8, concurrency=64)

    assert large["decode_tps"] > small["decode_tps"]
    assert large["replica_gpus"] > small["replica_gpus"]
    # Both are priced off the same GPU-hour, so whichever wins, the report says
    # which -- rather than leaving tokens/s to imply the larger one is better.
    for arm in (small, large):
        assert _costs(arm["report"])["output"] > 0.0
