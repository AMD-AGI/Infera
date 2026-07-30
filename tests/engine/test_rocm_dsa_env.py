###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera.engine.rocm_dsa_env (DSA topk_v2 ROCm opt-out)."""

from __future__ import annotations

import os

import infera.engine.rocm_dsa_env as rde


def test_disables_topk_v2_on_rocm(monkeypatch):
    monkeypatch.delenv("SGLANG_OPT_USE_TOPK_V2", raising=False)
    monkeypatch.setattr(rde, "_is_rocm", lambda: True)
    assert rde.apply_rocm_dsa_env_defaults() == {"SGLANG_OPT_USE_TOPK_V2": "0"}
    assert os.environ["SGLANG_OPT_USE_TOPK_V2"] == "0"


def test_operator_override_wins(monkeypatch):
    """Set-if-unset is how this workaround self-retires once upstream fixes the
    kernel — an explicit opt-in must survive."""
    monkeypatch.setattr(rde, "_is_rocm", lambda: True)
    monkeypatch.setenv("SGLANG_OPT_USE_TOPK_V2", "1")
    assert rde.apply_rocm_dsa_env_defaults() == {}
    assert os.environ["SGLANG_OPT_USE_TOPK_V2"] == "1"
