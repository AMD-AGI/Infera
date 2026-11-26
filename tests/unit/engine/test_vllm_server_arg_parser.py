###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Where `vllm serve`'s argument parser comes from: vLLM >= 0.34 moved it to
``vllm.entrypoints.common.cli_args`` (and expects its FlexibleArgumentParser),
earlier releases have it in ``vllm.entrypoints.openai.cli_args``. Stub modules
stand in for both layouts, so this runs without vLLM installed."""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from infera.engine.vllm.args import _vllm_server_arg_parser


def _install(monkeypatch: pytest.MonkeyPatch, modules: dict[str, object]) -> None:
    for name in ("vllm", "vllm.entrypoints", "vllm.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _cli_args_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)

    def make_arg_parser(parser):
        parser.add_argument("--model")
        return parser

    module.make_arg_parser = make_arg_parser
    return module


def test_vllm_034_layout_uses_flexible_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlexibleArgumentParser(argparse.ArgumentParser):
        pass

    argparse_utils = types.ModuleType("vllm.utils.argparse_utils")
    argparse_utils.FlexibleArgumentParser = FlexibleArgumentParser
    _install(
        monkeypatch,
        {
            "vllm.entrypoints.common": types.ModuleType("vllm.entrypoints.common"),
            "vllm.entrypoints.common.cli_args": _cli_args_module(
                "vllm.entrypoints.common.cli_args"
            ),
            "vllm.utils.argparse_utils": argparse_utils,
            # 0.34 removed the old module.
            "vllm.entrypoints.openai.cli_args": None,
        },
    )

    parser = _vllm_server_arg_parser()

    assert type(parser) is FlexibleArgumentParser
    assert parser.parse_args(["--model", "m"]).model == "m"


def test_pre_034_layout_uses_argparse(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            "vllm.entrypoints.common.cli_args": None,
            "vllm.entrypoints.openai": types.ModuleType("vllm.entrypoints.openai"),
            "vllm.entrypoints.openai.cli_args": _cli_args_module(
                "vllm.entrypoints.openai.cli_args"
            ),
        },
    )

    parser = _vllm_server_arg_parser()

    assert type(parser) is argparse.ArgumentParser
    assert parser.parse_args(["--model", "m"]).model == "m"
