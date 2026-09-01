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

from infera.projection.core.projection.inference_projection import benchmark_serving
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


# --- what one launch brings back --------------------------------------------
# A served point costs a whole client run, so which batches a launch sweeps is
# the difference between an anchor that answers off its measured point and one
# that does not.

def _sweep_batches(monkeypatch, **over):
    """Run the serving anchor with the engine and load generator stubbed."""
    measured: list[int] = []

    class FakeEngine:
        async def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(benchmark_serving, "_build_engine", lambda *a, **k: FakeEngine())
    monkeypatch.setattr(
        benchmark_serving, "_measure_concurrency",
        lambda port, batch, args, out_dir: measured.append(batch) or float(batch),
    )
    fields = dict(tp=8, pp=1, benchmark_gpus=4, batch=16, batches=None,
                  concurrency=None, input_len=1024, output_len=128, env=[],
                  quantization="mxfp4")
    fields.update(over)
    return measured, benchmark_serving.run_serving_benchmark(spec(**fields))


def test_a_concurrency_anchor_sweeps_the_ladder_rather_than_one_batch(monkeypatch):
    """Asking for a concurrency must not collapse to the default batch.

    ``--concurrency`` is what benchmark mode requests, and honouring it only on
    the offline path left the served sweep measuring a single point at whatever
    ``--batch`` happened to default to -- neither the shape asked for, nor a
    curve. The projector then held that one point flat across every batch while
    wearing a measurement's credibility.
    """
    measured, artifact = _sweep_batches(monkeypatch, concurrency=32)

    assert len(measured) > 1
    assert measured == sorted(measured)
    # The ladder has to reach the concurrency, so any batch up to it buckets up
    # to something actually measured.
    assert max(measured) >= 32
    assert artifact["meta"]["decode_pad_to_capture"] is True
    assert artifact["meta"]["concurrency"] == 32
    # The anchor point is the bucket covering the concurrency, not --batch.
    assert artifact["meta"]["batch"] == 32


def test_explicit_batches_are_measured_as_given(monkeypatch):
    """The ladder is for --concurrency; a caller naming batches still gets them,
    and no claim that decode may be bucketed."""
    measured, artifact = _sweep_batches(monkeypatch, batches="8,32")
    assert measured == [8, 32]
    assert artifact["meta"]["decode_pad_to_capture"] is False
