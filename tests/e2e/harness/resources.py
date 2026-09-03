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

from . import arch
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
    if params.skip_reason:
        pytest.skip(params.skip_reason)
    if params.expert_parallel and not params.is_moe:
        pytest.skip("expert_parallel needs an MoE model (this model is dense)")


def require_model_staged(params: EngineParams) -> None:
    """Fail when a pre-staged model directory exists but its ``config.json`` does not.

    ``INFERA_E2E_MODEL_DIR`` is bind-mounted read-only at the same path, and a tree
    staged as an HF cache (``hf download`` with no ``--local-dir``) is a farm of
    symlinks into a sibling ``blobs/`` that sits OUTSIDE that mount. The directory
    then lists correctly while every file in it dangles, and the run dies minutes
    later in ``transformers`` with "Should have a model_type key in its config.json"
    — which names neither the mount nor the link. Fails rather than skips: the case
    named this checkpoint, so a staging mistake is to be fixed, not passed over.

    Quiet when the path is not a local directory at all: that is either an HF repo
    id (loaded from the Hub) or the PD-disagg orchestrator, which resolves a path
    that only exists on the compute nodes.
    """
    path = params.model
    if not os.path.isdir(path):
        return
    config = os.path.join(path, "config.json")
    if os.path.isfile(config):
        return
    dangling = os.path.islink(config)
    detail = (
        f"it is a symlink to {os.readlink(config)!r}, which does not resolve here"
        if dangling
        else "it is missing"
    )
    hint = (
        "the tree is staged as an HF cache whose blobs sit outside the read-only "
        "mount; re-stage it self-contained, e.g. "
        "`cp -r -L -l <cache>/snapshots/<sha> $INFERA_E2E_MODEL_DIR/<org>/<name>` "
        "(hard links, so no second copy)"
        if dangling
        else "check INFERA_E2E_MODEL_DIR and that the checkpoint finished downloading"
    )
    pytest.fail(f"{path}/config.json is unreadable: {detail}. {hint}.", pytrace=False)


def require_arch() -> None:
    """Fail when a declared target arch contradicts the GPU present. Fails rather than
    skips: the image and knobs came from that declaration, so the run is misconfigured."""
    problem = arch.check_arch()
    if problem:
        pytest.fail(problem, pytrace=False)
