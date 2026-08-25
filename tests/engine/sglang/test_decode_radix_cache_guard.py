###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The hybrid SWA/SSM guard on --disaggregation-decode-enable-radix-cache.

infera appends that flag to the decode leg's argv when KV events are on.
SGLang rejects it for hybrid SWA/SSM models, and does so inside
build_kv_cache -- after the weights are loaded -- so the guard has to catch
it here or the decode leg dies ~2.5 minutes into startup.

Guarded by importorskip("sglang") like the other args tests: the module
imports sglang.srt.server_args at load time.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("sglang")

from infera.engine.sglang import args as args_mod  # noqa: E402
from infera.engine.sglang.args import parse_sglang_args  # noqa: E402

_DECODE = [
    "--model-path",
    "Qwen/Qwen3-0.6B",
    "--etcd-endpoint",
    "127.0.0.1:2379",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "mooncake",
]

_FLAG = "--disaggregation-decode-enable-radix-cache"


def _reason(monkeypatch, value):
    monkeypatch.setattr(args_mod, "_decode_radix_cache_unsupported_reason", lambda _sa: value)


def test_appends_for_a_supported_model(monkeypatch):
    _reason(monkeypatch, None)
    assert _FLAG in parse_sglang_args(_DECODE).sglang_argv


def test_skips_for_a_mamba_ssm_model(monkeypatch):
    _reason(monkeypatch, "Mamba/SSM")
    assert _FLAG not in parse_sglang_args(_DECODE).sglang_argv


def test_skips_for_a_hybrid_swa_model(monkeypatch):
    _reason(monkeypatch, "sliding window attention (SWA)")
    assert _FLAG not in parse_sglang_args(_DECODE).sglang_argv


def test_says_which_family_it_skipped_for(monkeypatch, caplog):
    _reason(monkeypatch, "Mamba/SSM")
    with caplog.at_level("INFO", logger=args_mod.logger.name):
        parse_sglang_args(_DECODE)
    assert "Mamba/SSM" in caplog.text


def test_an_explicit_flag_is_left_alone(monkeypatch):
    """The operator asked for it; the guard is only for what infera appends."""
    _reason(monkeypatch, "Mamba/SSM")
    argv = parse_sglang_args([*_DECODE, _FLAG]).sglang_argv
    assert argv.count(_FLAG) == 1


def test_prefill_leg_is_untouched(monkeypatch):
    _reason(monkeypatch, None)
    prefill = [*_DECODE[:-4], "--disaggregation-mode", "prefill", *_DECODE[-2:]]
    assert _FLAG not in parse_sglang_args(prefill).sglang_argv


def test_guard_never_raises_when_the_model_config_is_unreadable(monkeypatch):
    """A config we cannot read must not take the worker down here -- the
    engine fails for real a few lines later if it is genuinely broken."""

    class _Boom:
        def get_model_config(self):
            raise RuntimeError("no config.json")

    assert args_mod._decode_radix_cache_unsupported_reason(_Boom()) is None


def test_guard_never_raises_when_sglangs_private_module_moves(monkeypatch):
    """``sglang.srt.configs.hybrid_arch`` is private API that moves between
    releases. A rename must degrade to the warning path like any other
    unreadable config, not propagate an ImportError out of argv parsing."""
    # A None entry in sys.modules is what makes `import x` raise ImportError.
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.hybrid_arch", None)

    class _Unused:
        def get_model_config(self):
            raise AssertionError("the import fails before the config is read")

    assert args_mod._decode_radix_cache_unsupported_reason(_Unused()) is None
