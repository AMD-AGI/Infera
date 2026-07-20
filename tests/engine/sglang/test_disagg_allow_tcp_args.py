###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Phase 1: --disaggregation-allow-tcp plumbing for the SGLang launcher.

Guarded by importorskip("sglang") because infera.engine.sglang.args
imports sglang.srt.server_args at module load; runs in the SGLang
container, skips on bare dev boxes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sglang")

from infera.engine.sglang.args import parse_sglang_args  # noqa: E402

_BASE = ["--model-path", "Qwen/Qwen3-0.6B", "--etcd-endpoint", "127.0.0.1:2379"]


def test_allow_tcp_defaults_false():
    args = parse_sglang_args(_BASE)
    assert args.disaggregation_allow_tcp is False


def test_allow_tcp_flag_sets_true():
    args = parse_sglang_args([*_BASE, "--disaggregation-allow-tcp"])
    assert args.disaggregation_allow_tcp is True
