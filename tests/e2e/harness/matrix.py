###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Shared building blocks for the per-engine e2e "mixed" parametrize grids.

Engine-agnostic only: per-model trait maps + the helpers that turn a compact,
**declarative case table** into :class:`EngineParams`. Each engine composes its
OWN grid from these in its own directory (``tests/e2e/pd_mixed/{sglang,vllm,atom}/
matrix.py``), so per-engine model/knob choices stay local to that engine.

Declarative case table
-----------------------
Each engine grid is a list of rows, ``[model, tp, ep, dp_attn]`` with an
optional trailing ``opts`` dict::

    CASES = [
        # model, tp, ep, dp_attn
        ["openai/gpt-oss-120b", 2, False, False],
        # ...with per-case extra launch args / extra env:
        ["openai/gpt-oss-120b", 2, [False, True], False, {
            "args": ["--kv-cache-dtype", "fp8_e4m3",   # verbatim launch args
                     "--attention-backend", "aiter"],
            "env":  {"SGLANG_USE_AITER": "1"},         # worker subprocess env
        }],
    ]

and is expanded with :func:`expand_cases`. Row axes:

- ``model``    HF repo id (the logical id — used for the pytest id and the MoE
               lookup; the *launched* path is resolved via :func:`resolve_model`).
- ``tp``       tensor-parallel size — an ``int`` or a list of ints.
- ``ep``       expert-parallel     — a bool, or a list/tuple to enumerate.
- ``dp_attn``  dp-attention        — a bool, or a list/tuple to enumerate.
- ``opts``     optional dict:
               - ``args`` (list[str]): appended verbatim to the worker launch
                 argv — where per-case flags like ``--kv-cache-dtype fp8_e4m3`` go.
               - ``env`` (dict[str, str]): set on the worker subprocess.
               - ``setup`` (list[str]): shell commands run in the engine
                 container once before the worker launches (e.g.
                 ``["pip install amd-quark"]`` for an extra runtime dep).
               - ``server_ready_timeout`` (int): seconds to wait for the worker
                 to become active (default 300; raise for big MXFP4 MoE models).

An axis given a **list/tuple** enumerates each element (the cartesian product
across axes); a scalar is a single value. To cover both settings of a boolean
axis, pass it explicitly as ``[False, True]``.

Model location
--------------
The whole e2e suite resolves model *locations* through a single env var,
``INFERA_E2E_MODEL_DIR``: if it is set and ``<dir>/<model_id>`` exists on disk,
that local path is launched (offline, no HF pull); otherwise the ``model_id`` is
loaded from the HF Hub. ``tests/run_tests.sh`` mounts that dir (and any nested
per-model dirs) read-only into every e2e container and forwards the env var.

Extensibility:
- New test case: add ONE row to an engine's ``CASES`` table. MoE-ness is
  detected from the model's ``config.json`` (see :func:`is_moe`), so an
  ``ep=True`` row on a dense model self-skips (see
  :func:`tests.e2e.harness.resources.require_supported`) without any registry.
"""

from __future__ import annotations

import functools
import itertools
import json
import os

from .params import EngineParams

# Base dir for locally pre-staged models. Set/forwarded by run_tests.sh (see
# module docstring). Read once at import; unset ⇒ always load from the HF Hub.
MODEL_DIR = os.environ.get("INFERA_E2E_MODEL_DIR") or None

# Model references (HF repo ids). The actual launch path is resolved via
# resolve_model() (local copy under MODEL_DIR, else the HF Hub). MoE-ness is
# detected from the model's config (is_moe), not hardcoded.
GPT_OSS = "openai/gpt-oss-120b"
QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN3_8B = "Qwen/Qwen3-8B"
KIMI_K25_MXFP4 = "amd/Kimi-K2.5-MXFP4"
KIMI_K26_MXFP4 = "amd/Kimi-K2.6-MXFP4"
DEEPSEEK_V4_PRO = "deepseek-ai/DeepSeek-V4-Pro"
GLM_5_1_FP8 = "zai-org/GLM-5.1-FP8"

EXTRA_ARGS: dict[str, tuple[str, ...]] = {}  # default verbatim extra launch args

# config.json keys that mark a Mixture-of-Experts model (any present with an
# expert count > 1). Covers gpt-oss (num_local_experts), Mixtral (num_local_experts),
# Qwen-MoE (num_experts), DeepSeek (n_routed_experts), etc.
_MOE_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def resolve_model(model_id: str) -> str:
    """Resolve a model reference to a launchable location.

    If :data:`MODEL_DIR` is set and ``<MODEL_DIR>/<model_id>`` is a directory,
    return that local path (offline load); otherwise return ``model_id``
    unchanged (loaded from the HF Hub by id).
    """
    if MODEL_DIR:
        local = os.path.join(MODEL_DIR, model_id)
        if os.path.isdir(local):
            return local
        # The PD-disagg orchestrator runs on a host that may NOT have MODEL_DIR
        # mounted (the pre-staged tree lives on the compute nodes). When the base
        # dir isn't visible here, trust that "<dir>/<model_id>" exists on the node
        # and use it — otherwise we'd needlessly pull a staged model from the Hub.
        if not os.path.isdir(MODEL_DIR):
            return local
    return model_id


def _load_config(model_id: str) -> dict:
    """Best-effort read of a model's ``config.json`` (local resolved path first,
    else fetched from the HF Hub). Returns ``{}`` if it can't be read."""
    path = resolve_model(model_id)
    local = os.path.join(path, "config.json")
    if os.path.isfile(local):
        try:
            with open(local) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(model_id, "config.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def _has_moe_key(obj) -> bool:
    """Recursively scan a config dict for an expert-count key > 1, so nested
    sub-configs (``text_config`` / ``thinker_config`` for multimodal, or a
    model's own nested language config like Kimi) are all covered."""
    if not isinstance(obj, dict):
        return False
    for key, val in obj.items():
        if key in _MOE_CONFIG_KEYS and isinstance(val, int) and val > 1:
            return True
        if isinstance(val, dict) and _has_moe_key(val):
            return True
    return False


@functools.cache
def is_moe(model_id: str) -> bool:
    """Detect a Mixture-of-Experts model by inspecting its ``config.json``
    (any nested expert-count key > 1) instead of a hardcoded list.
    Best-effort: returns False when the config can't be read."""
    return _has_moe_key(_load_config(model_id))


def make_params(
    model_id: str, *, extra_args=None, extra_env=None, setup=None, server_ready_timeout=None, **kw
) -> EngineParams:
    """EngineParams for ``model_id`` with per-model + per-case traits filled.

    Traits (MoE flag, default extra args) are keyed off the *logical*
    ``model_id``; the ``model`` field stores the resolved launch location
    (:func:`resolve_model`). Per-case ``extra_args`` / ``extra_env`` / ``setup``
    / ``server_ready_timeout`` override the per-model / dataclass defaults.
    """
    if server_ready_timeout is not None:
        kw["server_ready_timeout"] = server_ready_timeout
    return EngineParams(
        model=resolve_model(model_id),
        is_moe=is_moe(model_id),
        extra_args=tuple(extra_args) if extra_args else EXTRA_ARGS.get(model_id, ()),
        extra_env=tuple(extra_env.items()) if extra_env else (),
        setup=tuple(setup) if setup else (),
        **kw,
    )


def _axis(value) -> tuple:
    """Normalise one case-table axis into the tuple of values to enumerate.

    A list/tuple enumerates each element verbatim; anything else is a single
    value. To enumerate a boolean axis, pass it explicitly as ``[False, True]``.
    """
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def expand_cases(table) -> list[EngineParams]:
    """Expand a declarative case table (see the module docstring) to params.

    Each row is ``[model, tp, ep, dp_attn]`` with an optional trailing ``opts``
    dict (``args`` / ``env``). Each axis is normalised via :func:`_axis`
    (a list/tuple enumerates), then the cartesian product of the axes
    yields one :class:`EngineParams` per combination.
    """
    params: list[EngineParams] = []
    for row in table:
        model_id, tp, ep, dp_attn = row[0], row[1], row[2], row[3]
        opts = row[4] if len(row) > 4 and row[4] else {}
        for t, e, d in itertools.product(_axis(tp), _axis(ep), _axis(dp_attn)):
            params.append(
                make_params(
                    model_id,
                    tensor_parallel_size=t,
                    expert_parallel=e,
                    dp_attention=d,
                    extra_args=opts.get("args"),
                    extra_env=opts.get("env"),
                    setup=opts.get("setup"),
                    server_ready_timeout=opts.get("server_ready_timeout"),
                )
            )
    return params
