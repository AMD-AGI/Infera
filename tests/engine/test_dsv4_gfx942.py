###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Unit tests for infera.engine.dsv4_gfx942 (gfx942 DSv4 support policy)."""

from __future__ import annotations

import json

import pytest

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


def test_detect_quant_config_as_string(tmp_path):
    # quantization_config given as a non-dict (string) -> _detect_quant's
    # str() branch must still classify it.
    p = _write_config(
        tmp_path, {**_PRO, "quantization_config": "fp8 e4m3 blockwise"}
    )
    m = d.detect_dsv4(p)
    assert m is not None and m.quant == "fp8"


def test_detect_empty_config_returns_none(tmp_path):
    # Empty config -> not dsv4 -> None.
    assert d.detect_dsv4(_write_config(tmp_path, {})) is None


def test_detect_model_type_casing_normalized(tmp_path):
    # model_type is lowercased before the deepseek_v4 check.
    p = _write_config(
        tmp_path, {"model_type": "DeepSeek_V4", "hidden_size": 7168,
                   "num_hidden_layers": 61, **_FP8_QC}
    )
    m = d.detect_dsv4(p)
    assert m is not None and m.variant == "pro" and m.quant == "fp8"


def test_detect_dsv4_via_index_topk_without_model_type(tmp_path):
    # dsv4 detected via index_topk sparse-attn config even when model_type
    # is not deepseek_v4.
    p = _write_config(
        tmp_path, {"model_type": "custom", "index_topk": 1024,
                   "hidden_size": 7168, "num_hidden_layers": 61, **_FP8_QC}
    )
    m = d.detect_dsv4(p)
    assert m is not None and m.variant == "pro" and m.quant == "fp8"


def test_detect_quant_fp4_wins_when_both_present(tmp_path):
    # A real mxfp4 checkpoint often carries fp8/e4m3 SCALE fields too; fp4 must
    # win (checked first) so we don't misclassify an fp4 model as fp8.
    p = _write_config(
        tmp_path,
        {**_PRO, "quantization_config": {"quant_method": "mxfp4",
                                         "fmt": "e2m1", "scale_fmt": "e4m3"}},
    )
    m = d.detect_dsv4(p)
    assert m is not None and m.quant == "fp4"


# --------------------------------------------------------------------------
# apply_gfx942_dsv4: matrix enforcement + env/CLI injection
# --------------------------------------------------------------------------

_ALL_ENV = [
    "HSA_NO_SCRATCH_RECLAIM",
    "SGLANG_USE_ROCM700A",
    "SGLANG_HACK_FLASHMLA_BACKEND",
    "AITER_BF16_FP8_MOE_BOUND",
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for v in _ALL_ENV:
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def _force_gfx942(monkeypatch):
    monkeypatch.setattr(d, "is_gfx942", lambda: True)


def _cfg(tmp_path, base, qc):
    return _write_config(tmp_path, {**base, **qc})


# ---- no-op paths ----

def test_noop_when_not_gfx942(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "is_gfx942", lambda: False)
    p = _cfg(tmp_path, _PRO, _FP8_QC)
    argv = ["--tp", "8"]
    assert d.apply_gfx942_dsv4(p, engine="sglang", argv=argv) == ["--tp", "8"]
    assert "HSA_NO_SCRATCH_RECLAIM" not in __import__("os").environ


def test_noop_when_not_dsv4(tmp_path, _force_gfx942):
    p = _write_config(tmp_path, {"model_type": "llama", "hidden_size": 4096})
    argv = ["--tp", "8"]
    assert d.apply_gfx942_dsv4(p, engine="sglang", argv=argv) == ["--tp", "8"]


# ---- fail-fast (unsupported cells) ----

@pytest.mark.parametrize(
    "engine,base,qc",
    [
        ("sglang", _PRO, _FP4_QC),
        ("sglang", _FLASH, _FP4_QC),
        ("atom", _PRO, _FP4_QC),
        ("atom", _FLASH, _FP4_QC),
        ("vllm", _PRO, _FP8_QC),
        ("vllm", _FLASH, _FP8_QC),
    ],
)
def test_unsupported_combos_raise(tmp_path, _force_gfx942, engine, base, qc):
    p = _cfg(tmp_path, base, qc)
    with pytest.raises(d.Dsv4UnsupportedError):
        d.apply_gfx942_dsv4(p, engine=engine, argv=[])


# ---- supported cells do not raise ----

@pytest.mark.parametrize(
    "engine,base,qc",
    [
        ("vllm", _PRO, _FP4_QC),
        ("vllm", _FLASH, _FP4_QC),
        ("sglang", _PRO, _FP8_QC),
        ("sglang", _FLASH, _FP8_QC),
        ("atom", _PRO, _FP8_QC),
        ("atom", _FLASH, _FP8_QC),
    ],
)
def test_supported_combos_ok(tmp_path, _force_gfx942, engine, base, qc):
    p = _cfg(tmp_path, base, qc)
    d.apply_gfx942_dsv4(p, engine=engine, argv=[])  # must not raise


# ---- sglang fp8 env + CLI ----

def test_sglang_pro_fp8_env_and_cli(tmp_path, _force_gfx942):
    import os

    p = _cfg(tmp_path, _PRO, _FP8_QC)
    out = d.apply_gfx942_dsv4(p, engine="sglang", argv=["--tp", "8"])
    assert os.environ["HSA_NO_SCRATCH_RECLAIM"] == "1"
    assert os.environ["SGLANG_HACK_FLASHMLA_BACKEND"] == "unified_kv_triton"
    assert os.environ["SGLANG_USE_ROCM700A"] == "0"
    assert os.environ["AITER_BF16_FP8_MOE_BOUND"] == "0"
    assert "--attention-backend" in out and "dsv4" in out
    assert "--disable-shared-experts-fusion" in out
    # Pro does NOT get MTP.
    assert "--speculative-algorithm" not in out


def test_sglang_flash_fp8_injects_mtp(tmp_path, _force_gfx942):
    p = _cfg(tmp_path, _FLASH, _FP8_QC)
    out = d.apply_gfx942_dsv4(p, engine="sglang", argv=[])
    assert "--speculative-algorithm" in out
    i = out.index("--speculative-algorithm")
    assert out[i + 1] == "EAGLE"
    assert "--speculative-num-steps" in out


# ---- atom fp8 env + MTP ----

def test_atom_pro_fp8_env_only(tmp_path, _force_gfx942):
    import os

    p = _cfg(tmp_path, _PRO, _FP8_QC)
    out = d.apply_gfx942_dsv4(p, engine="atom", argv=["-tp", "8"])
    assert os.environ["HSA_NO_SCRATCH_RECLAIM"] == "1"
    # atom does not take the sglang-specific env.
    assert "SGLANG_HACK_FLASHMLA_BACKEND" not in os.environ
    assert "--method" not in out  # Pro: no MTP


def test_atom_flash_fp8_injects_mtp(tmp_path, _force_gfx942):
    p = _cfg(tmp_path, _FLASH, _FP8_QC)
    out = d.apply_gfx942_dsv4(p, engine="atom", argv=[])
    assert "--method" in out
    i = out.index("--method")
    assert out[i + 1] == "mtp"
    assert "--num-speculative-tokens" in out


# ---- set-if-unset: operator override wins ----

def test_env_operator_override_preserved(tmp_path, _force_gfx942, monkeypatch):
    import os

    monkeypatch.setenv("SGLANG_HACK_FLASHMLA_BACKEND", "custom_backend")
    p = _cfg(tmp_path, _PRO, _FP8_QC)
    d.apply_gfx942_dsv4(p, engine="sglang", argv=[])
    assert os.environ["SGLANG_HACK_FLASHMLA_BACKEND"] == "custom_backend"


def test_cli_not_duplicated_when_present(tmp_path, _force_gfx942):
    p = _cfg(tmp_path, _PRO, _FP8_QC)
    out = d.apply_gfx942_dsv4(
        p, engine="sglang", argv=["--attention-backend", "flashinfer"]
    )
    assert out.count("--attention-backend") == 1
    i = out.index("--attention-backend")
    assert out[i + 1] == "flashinfer"  # operator value kept
