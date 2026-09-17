###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine startup timeout environment contract."""

from __future__ import annotations

import importlib.util
import sys
from importlib import import_module
from types import ModuleType

import pytest

from infera.engine.vllm.worker import _ready_timeout as vllm_ready_timeout


def _load_sglang_ready_timeout():
    # The dev/unit-test extra intentionally does not install the vendor SGLang
    # package. The worker only needs this type at import time for these tests.
    if importlib.util.find_spec("sglang") is not None:
        return import_module("infera.engine.sglang.worker")._ready_timeout

    sglang = ModuleType("sglang")
    sglang.__path__ = []
    srt = ModuleType("sglang.srt")
    srt.__path__ = []
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = object
    fake_modules = {
        "sglang": sglang,
        "sglang.srt": srt,
        "sglang.srt.server_args": server_args,
    }
    touched = [*fake_modules, "infera.engine.sglang.worker"]
    previous = {name: sys.modules.get(name) for name in touched}
    try:
        sys.modules.update(fake_modules)
        return import_module("infera.engine.sglang.worker")._ready_timeout
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


sglang_ready_timeout = _load_sglang_ready_timeout()


@pytest.mark.parametrize("reader", [sglang_ready_timeout, vllm_ready_timeout])
def test_ready_timeout_defaults_to_1800(monkeypatch, reader):
    monkeypatch.delenv("INFERA_ENGINE_READY_TIMEOUT", raising=False)
    assert reader() == 1800.0


@pytest.mark.parametrize("reader", [sglang_ready_timeout, vllm_ready_timeout])
def test_ready_timeout_accepts_override(monkeypatch, reader):
    monkeypatch.setenv("INFERA_ENGINE_READY_TIMEOUT", "3600")
    assert reader() == 3600.0


@pytest.mark.parametrize("reader", [sglang_ready_timeout, vllm_ready_timeout])
def test_ready_timeout_invalid_value_falls_back(monkeypatch, reader):
    monkeypatch.setenv("INFERA_ENGINE_READY_TIMEOUT", "not-a-number")
    assert reader() == 1800.0
