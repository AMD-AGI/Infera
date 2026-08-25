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
    _engine_argv,
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


@pytest.mark.parametrize("backend, model_flag, tp_flag, http_flag", [
    ("vllm", None, "--tensor-parallel-size", "--port"),
    ("sglang", "--model-path", "--tp", "--port"),
    ("atom", "--model", "--tensor-parallel-size", "--server-port"),
])
def test_each_engine_is_launched_the_way_it_expects(backend, model_flag, tp_flag,
                                                    http_flag):
    argv = _engine_argv(spec(serving_backend=backend), port=8123, tp=4)
    if model_flag is None:
        assert argv[0] == "openai/gpt-oss-120b"  # vLLM takes the model positionally
    else:
        assert argv[argv.index(model_flag) + 1] == "openai/gpt-oss-120b"
    assert argv[argv.index(tp_flag) + 1] == "4"
    assert argv[argv.index(http_flag) + 1] == "8123"


def test_atom_keeps_its_rendezvous_port_off_the_http_port():
    """ATOM's --port is MASTER_PORT, not the API; sharing one number deadlocks."""
    argv = _engine_argv(spec(serving_backend="atom"), port=8123, tp=1)
    assert argv[argv.index("--server-port") + 1] == "8123"
    assert argv[argv.index("--port") + 1] != "8123"


def test_the_context_length_flag_follows_the_engine():
    """The same intent, spelled differently by each engine."""
    assert "--max-model-len" in _engine_argv(spec(), port=1, tp=1)
    assert "--context-length" in _engine_argv(spec(serving_backend="sglang"),
                                              port=1, tp=1)


def test_caller_server_args_always_win_by_coming_last():
    argv = _engine_argv(spec(server_args="--gpu-memory-utilization 0.85"),
                        port=1, tp=1)
    assert argv[-2:] == ["--gpu-memory-utilization", "0.85"]
