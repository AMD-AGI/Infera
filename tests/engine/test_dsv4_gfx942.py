###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera.engine.dsv4_gfx942 (gfx942 DSv4 support policy)."""

from __future__ import annotations

import json

import pytest  # noqa: F401  (used by Task 2 enforcement tests)

import infera.engine.dsv4_gfx942 as d


def _write_config(tmp_path, cfg: dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


# Minimal representative configs (dims are the discriminators).
_PRO = {"model_type": "deepseek_v4", "hidden_size": 7168, "num_hidden_layers": 61}
_FLASH = {"model_type": "deepseek_v4", "hidden_size": 4096, "num_hidden_layers": 43}
_FP4_QC = {"quantization_config": {"quant_method": "mxfp4", "fmt": "e2m1"}}
_FP8_QC = {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}


def test_detect_pro_fp4(tmp_path):
    p = _write_config(tmp_path, {**_PRO, **_FP4_QC})
    m = d.detect_dsv4(p)
    assert m is not None and m.variant == "pro" and m.quant == "fp4"


def test_detect_pro_fp8(tmp_path):
    p = _write_config(tmp_path, {**_PRO, **_FP8_QC})
    m = d.detect_dsv4(p)
    assert m is not None and m.variant == "pro" and m.quant == "fp8"


def test_detect_flash_fp4(tmp_path):
    p = _write_config(tmp_path, {**_FLASH, **_FP4_QC})
    m = d.detect_dsv4(p)
    assert m is not None and m.variant == "flash" and m.quant == "fp4"


def test_detect_flash_fp8(tmp_path):
    p = _write_config(tmp_path, {**_FLASH, **_FP8_QC})
    m = d.detect_dsv4(p)
    assert m is not None and m.variant == "flash" and m.quant == "fp8"


def test_detect_non_dsv4_returns_none(tmp_path):
    p = _write_config(tmp_path, {"model_type": "llama", "hidden_size": 4096})
    assert d.detect_dsv4(p) is None


def test_detect_bare_repo_id_returns_none():
    # Not a local dir -> never downloads, returns None.
    assert d.detect_dsv4("deepseek-ai/DeepSeek-V4-Pro") is None


def test_detect_missing_config_returns_none(tmp_path):
    assert d.detect_dsv4(str(tmp_path)) is None


def test_detect_none_path_returns_none():
    assert d.detect_dsv4(None) is None
