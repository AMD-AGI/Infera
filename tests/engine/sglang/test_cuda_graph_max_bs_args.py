###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""--cuda-graph-max-bs keeps working on SGLang builds that split it into
--cuda-graph-max-bs-decode / --cuda-graph-max-bs-prefill (v0.5.22), where argparse
would otherwise reject it as an ambiguous prefix.

Guarded by importorskip("sglang") because infera.engine.sglang.args
imports sglang.srt.server_args at module load; runs in the SGLang
container, skips on bare dev boxes.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs  # noqa: E402

from infera.engine.sglang.args import parse_sglang_args  # noqa: E402

_BASE = ["--model-path", "Qwen/Qwen3-0.6B", "--etcd-endpoint", "127.0.0.1:2379"]


def _sglang_has(flag: str) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    return flag in parser._option_string_actions


@pytest.mark.parametrize("tokens", [["--cuda-graph-max-bs", "8"], ["--cuda-graph-max-bs=8"]])
def test_cuda_graph_max_bs_sizes_the_decode_graphs(tokens):
    args = parse_sglang_args([*_BASE, *tokens])
    if _sglang_has("--cuda-graph-max-bs"):
        assert args.sglang_argv[-len(tokens) :] == tokens
    else:
        old = [t for t in args.sglang_argv if t.split("=")[0] == "--cuda-graph-max-bs"]
        assert old == []
        assert args.server_args.cuda_graph_max_bs_decode == 8


def test_the_decode_flag_passes_through_unchanged():
    if not _sglang_has("--cuda-graph-max-bs-decode"):
        pytest.skip("this SGLang has no --cuda-graph-max-bs-decode")
    tokens = ["--cuda-graph-max-bs-decode", "16"]
    args = parse_sglang_args([*_BASE, *tokens])
    assert args.sglang_argv[-2:] == tokens
