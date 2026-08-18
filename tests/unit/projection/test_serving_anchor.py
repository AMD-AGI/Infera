###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""An anchor has to be measured on the engine it is going to predict.

Offline ``LLM()`` and a real server do not resolve the same kernels, so the
anchor path launches a server by default and records which kernels it got. The
launch is the only part that differs between engines: both expose ``/health``
and an OpenAI-compatible completions route, so readiness and load generation
are shared.
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from infera.projection.core.projection.inference_projection.benchmark_serving import (
    _server_command,
    resolved_kernels,
)

VLLM_LOG = """
Overriding with ROCM_AITER_FA out of potential backends: ['ROCM_AITER_FA']
Using 'AITER_MXFP4_BF16' Mxfp4 MoE backend
"""

SERVER_LOG = """
Overriding with ROCM_AITER_UNIFIED_ATTN out of potential backends: ['TRITON_ATTN']
Using 'TRITON' Mxfp4 MoE backend
"""


def spec(**over) -> Namespace:
    base = dict(model="openai/gpt-oss-120b", serving_backend="vllm",
                max_model_len=8192, enable_expert_parallel=False,
                enforce_eager=False, quantization=None, kv_cache_dtype=None,
                server_args="")
    base.update(over)
    return Namespace(**base)


def test_the_two_entrypoints_do_not_run_the_same_kernels():
    """The finding the serving default exists to fix, pinned as a test."""
    offline = resolved_kernels(VLLM_LOG)
    served = resolved_kernels(SERVER_LOG)
    assert offline["resolved_attention_backend"] == "ROCM_AITER_FA"
    assert offline["resolved_moe_backend"] == "AITER_MXFP4_BF16"
    assert served["resolved_attention_backend"] == "ROCM_AITER_UNIFIED_ATTN"
    assert served["resolved_moe_backend"] == "TRITON"
    assert offline != served


def test_kernels_are_absent_rather_than_wrong_for_another_engine():
    """A log that never names a vLLM backend must not report one."""
    assert resolved_kernels("Loading weights...\nServer started") == {
        "resolved_attention_backend": None,
        "resolved_moe_backend": None,
    }


@pytest.mark.parametrize("backend, head, tp_flag", [
    ("vllm", ["vllm", "serve", "openai/gpt-oss-120b"], "--tensor-parallel-size"),
    ("sglang", ["python", "-m", "sglang.launch_server"], "--tp"),
])
def test_each_engine_is_launched_the_way_it_expects(backend, head, tp_flag):
    cmd = _server_command(spec(serving_backend=backend), port=8123, tp=4)
    assert cmd[:len(head)] == head
    assert cmd[cmd.index(tp_flag) + 1] == "4"
    assert cmd[cmd.index("--port") + 1] == "8123"


def test_the_context_length_flag_follows_the_engine():
    """The same intent, spelled differently by each engine."""
    assert "--max-model-len" in _server_command(spec(), port=1, tp=1)
    assert "--context-length" in _server_command(spec(serving_backend="sglang"),
                                                 port=1, tp=1)


def test_caller_server_args_always_win_by_coming_last():
    cmd = _server_command(spec(server_args="--gpu-memory-utilization 0.85"),
                          port=1, tp=1)
    assert cmd[-2:] == ["--gpu-memory-utilization", "0.85"]
