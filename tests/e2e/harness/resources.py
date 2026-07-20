###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Environment/resource guards for the e2e parameter matrix.

The intended matrix (tp=4, dp-attention, expert-parallel, ...) needs more
GPUs than a default 2-GPU e2e box exposes, and expert-parallel needs an MoE
model. Rather than fail on under-provisioned hosts, tests call these guards
up front so unsupported combinations *skip* with a clear reason and the
suite stays green on whatever hardware is available.
"""

from __future__ import annotations

import os

import pytest

from .params import EngineParams


def visible_gpu_count() -> int:
    """Number of GPUs visible to this process.

    Prefers the explicit HIP/CUDA visibility list the test harness runs
    under (run_tests.sh passes ``HIP_VISIBLE_DEVICES``); falls back to torch.
    """
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        val = os.environ.get(var)
        if val:
            return len([x for x in val.split(",") if x.strip() != ""])
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def require_gpus(params: EngineParams) -> None:
    """Skip unless enough GPUs are visible for the worker's tp_size."""
    need = max(1, params.tensor_parallel_size)
    have = visible_gpu_count()
    if have < need:
        pytest.skip(f"needs {need} GPUs (tp{params.tensor_parallel_size}); only {have} visible")


def require_supported(params: EngineParams) -> None:
    """Skip param combinations the harness can't yet honour end-to-end."""
    if params.expert_parallel and not params.is_moe:
        pytest.skip("expert_parallel needs an MoE model (this model is dense)")
