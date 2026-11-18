###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The DP identity a vLLM worker registers, and INFERA_VLLM_DP_ENGINE_LB.

A DP engine registers dp_size so the router fans it out into one target per
rank and pins each request with a rank header. With INFERA_VLLM_DP_ENGINE_LB=1
it registers as a single worker instead, and vLLM's own DP balancer picks the
rank (P4D4 decode: the router's input-token booking left one rank KV-full 60%
of the time while another sat at 78%).
"""

from __future__ import annotations

import pytest

from infera.engine.vllm.args import _dp_identity


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("INFERA_VLLM_DP_ENGINE_LB", raising=False)


def test_dp_engine_registers_its_size_and_rank():
    assert _dp_identity(4, None) == (4, None)
    assert _dp_identity(4, 2) == (4, 2)


def test_single_rank_registers_no_dp_identity():
    assert _dp_identity(1, None) == (None, None)
    assert _dp_identity(None, None) == (None, None)


def test_engine_lb_registers_a_dp_engine_as_one_worker(monkeypatch):
    monkeypatch.setenv("INFERA_VLLM_DP_ENGINE_LB", "1")
    assert _dp_identity(4, None) == (None, None)
    assert _dp_identity(4, 3) == (None, None)


def test_engine_lb_needs_the_exact_value(monkeypatch):
    monkeypatch.setenv("INFERA_VLLM_DP_ENGINE_LB", "0")
    assert _dp_identity(4, None) == (4, None)
