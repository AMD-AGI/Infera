###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""CLI plumbing for --wait-for-decode on the SGLang worker."""

from __future__ import annotations

import pytest

pytest.importorskip("sglang")

from infera.engine.sglang.args import parse_sglang_args  # noqa: E402

_PREFILL = [
    "--model-path",
    "Qwen/Qwen3-0.6B",
    "--served-model-name",
    "glm-5-3",
    "--disaggregation-mode",
    "prefill",
]


def test_wait_for_decode_defaults_none_on_prefill():
    args = parse_sglang_args(_PREFILL)
    assert args.wait_for_decode is None
    assert args.decode_ready_timeout is None
    assert args.k8s_label_selector is None


def test_no_wait_for_decode_flag():
    args = parse_sglang_args([*_PREFILL, "--no-wait-for-decode"])
    assert args.wait_for_decode is False


def test_wait_for_decode_timeout_and_selector():
    args = parse_sglang_args(
        [
            *_PREFILL,
            "--wait-for-decode",
            "--decode-ready-timeout",
            "90",
            "--k8s-label-selector",
            "infera.amd.com/deployment=x",
        ]
    )
    assert args.wait_for_decode is True
    assert args.decode_ready_timeout == 90.0
    assert args.k8s_label_selector == "infera.amd.com/deployment=x"
