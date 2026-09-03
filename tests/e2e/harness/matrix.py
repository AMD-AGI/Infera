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
Each engine grid is a list of rows, ``[enable, model, tp, ep, dp_attn]`` with an
optional trailing ``opts`` dict::

    CASES = [
        # enable, model, tp, ep, dp_attn
        [True, "openai/gpt-oss-120b", 2, False, False],
        # ...with per-case extra launch args / extra env:
        [True, "openai/gpt-oss-120b", 2, [False, True], False, {
            "args": ["--kv-cache-dtype", "fp8_e4m3",   # verbatim launch args
                     "--attention-backend", "aiter"],
            "env":  {"SGLANG_USE_AITER": "1"},         # worker subprocess env
        }],
        # kept in the tree, not run until someone flips it:
        [False, "deepseek-ai/DeepSeek-V4-Pro", 8, False, False],
    ]

and is expanded with :func:`expand_cases`. Row axes:

- ``enable``   ``True`` collects the row; ``False`` keeps the (often long and
               hard-won) launch recipe in the tree without running it, so
               parking a case no longer means commenting the block out.
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
               - ``skip`` (str): non-empty skips the case with that reason.
               - ``"gfx942"`` / ``"gfx950"``: a per-architecture overlay — see
                 below.
An axis given a **list/tuple** enumerates each element (the cartesian product
across axes); a scalar is a single value. To cover both settings of a boolean
axis, pass it explicitly as ``[False, True]``.

Per-architecture overrides
--------------------------
A case usually needs different launch knobs on gfx942 (MI300X / MI325X) than on
gfx950 (MI355X). Rather than a second case table — which duplicates every row
and drifts on the first one-sided edit — a row keeps its single entry, and its
``opts`` gains a key named after the architecture holding only the delta::

    [True, GPT_OSS, 2, True, False, {
        "args": ["--attention-backend", "triton"],
        "env":  {"SGLANG_USE_AITER": "1"},
        "gfx942": {                                    # applied on gfx942 only
            "args": ["--attention-backend", "aiter"],
            "env":  {"HSA_NO_SCRATCH_RECLAIM": "1"},
            "server_ready_timeout": 2400,
        },
    }]

:func:`expand_cases` merges the overlay for :func:`.arch.target_arch` into the
base, so the row, its axes and its pytest id are identical on both
architectures and only the knobs differ. Merge rules:

- ``args`` / ``setup`` **replace** the base list outright. Engines disagree on
  how a repeated flag resolves, so an append-merge would not be predictable;
  these lists are short, and a full list reads as a config rather than a patch.
- ``env`` merges per key, and a value of ``None`` **deletes** the base key (env
  is nearly always additive, so ``None`` is the escape hatch for a removal).
- Everything else replaces.
- ``skip`` keeps an unrunnable case *visible* in the report with its reason,
  which ``enable=False`` and a deleted row do not.

An unknown key — in the base ``opts`` or in an overlay — raises at collection
time, because a typo like ``"gfx940"`` or ``"arg"`` otherwise does nothing at
all, silently.

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

from .arch import SUPPORTED_ARCHS, target_arch
from .params import EngineParams

# Base dir for locally pre-staged models. Set/forwarded by run_tests.sh (see
# module docstring). Read once at import; unset ⇒ always load from the HF Hub.
MODEL_DIR = os.environ.get("INFERA_E2E_MODEL_DIR") or None

# Knobs a case's ``opts`` may carry, besides the per-architecture overlays.
_OPT_KEYS = frozenset({"args", "env", "setup", "server_ready_timeout", "skip"})

# Model references (HF repo ids). The actual launch path is resolved via
# resolve_model() (local copy under MODEL_DIR, else the HF Hub). MoE-ness is
# detected from the model's config (is_moe), not hardcoded.
GPT_OSS = "openai/gpt-oss-120b"
QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN3_8B = "Qwen/Qwen3-8B"
KIMI_K25_MXFP4 = "amd/Kimi-K2.5-MXFP4"
KIMI_K26_MXFP4 = "amd/Kimi-K2.6-MXFP4"
DEEPSEEK_V4_PRO = "deepseek-ai/DeepSeek-V4-Pro"
DEEPSEEK_V4_FLASH = "deepseek-ai/DeepSeek-V4-Flash"
DEEPSEEK_V4_FLASH_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
GLM_5_1_FP8 = "zai-org/GLM-5.1-FP8"
GLM_5_2_FP8 = "zai-org/GLM-5.2-FP8"

# Which of GLM-5.2's 78 layers own a DSA lightning indexer and which reuse one.
# The checkpoint's `indexer_types` marks layers 0, 1, 2 and then every 4th "full"
# and the rest "shared", and it ships indexer weights for the "full" ones only.
# ATOM spells the same thing as `index_topk_pattern`, where "S" means "skip the
# top-k and reuse the last selection"; it is spelled out per layer because ATOM's
# own index_topk_freq shorthand derives a different, off-by-one set of owners
# (see the GLM-5.2 row in pd_mixed/atom/matrix.py). Consumed by the ATOM rows.
# Hand-copied, so check it against the checkpoint's indexer_types, not by eye.
GLM_5_2_INDEXER_PATTERN = "FFFSSS" + "FSSS" * 18

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
    model_id: str,
    *,
    extra_args=None,
    extra_env=None,
    setup=None,
    server_ready_timeout=None,
    skip_reason=None,
    **kw,
) -> EngineParams:
    """EngineParams for ``model_id`` with per-model + per-case traits filled.

    Traits (MoE flag, default extra args) are keyed off the *logical*
    ``model_id``; the ``model`` field stores the resolved launch location
    (:func:`resolve_model`). Per-case ``extra_args`` / ``extra_env`` / ``setup``
    / ``server_ready_timeout`` override the per-model / dataclass defaults.
    """
    if skip_reason:
        kw["skip_reason"] = skip_reason
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


def apply_arch_overlay(opts: dict, arch: str) -> dict:
    """Collapse a case's ``opts`` (base knobs + per-arch overlays) into plain knobs for
    ``arch``. Merge rules and the reasoning behind them are in the module docstring."""
    _reject_unknown_keys(opts, "opts")
    base = {k: v for k, v in opts.items() if k not in SUPPORTED_ARCHS}
    overlay = opts.get(arch) or {}
    if not overlay:
        return base
    _reject_unknown_keys(overlay, f"the {arch!r} overlay", allow_arch_keys=False)

    merged = {**base, **overlay}
    if "env" in merged:
        merged["env"] = _merge_env(base.get("env"), overlay.get("env"))
    return merged


def _merge_env(base: dict | None, overlay: dict | None) -> dict:
    """Per-key env overlay, where a ``None`` value drops the base key."""
    merged = dict(base or {})
    for key, value in (overlay or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _reject_unknown_keys(opts: dict, where: str, *, allow_arch_keys: bool = True) -> None:
    """Reject a key that is neither a known knob nor an allowed arch overlay."""
    known = _OPT_KEYS | (set(SUPPORTED_ARCHS) if allow_arch_keys else set())
    unknown = sorted(set(opts) - known)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; expected any of {sorted(known)}")


def expand_cases(table, *, arch: str | None = None) -> list[EngineParams]:
    """Expand a declarative case table (see the module docstring) to params.

    Each row is ``[enable, model, tp, ep, dp_attn]`` with an optional trailing
    ``opts`` dict (``args`` / ``env``). Rows with ``enable`` false are dropped
    here, so a parked case is never collected. Each axis is normalised via
    :func:`_axis` (a list/tuple enumerates), then the cartesian product of the
    axes yields one :class:`EngineParams` per combination.

    ``opts`` is first collapsed for ``arch`` (default: the run's target
    architecture), which is why the same table yields the same ids and the same
    case count on gfx942 as it does on gfx950.
    """
    arch = arch or target_arch()
    params: list[EngineParams] = []
    for row in table:
        enable, model_id, tp, ep, dp_attn = row[0], row[1], row[2], row[3], row[4]
        opts = apply_arch_overlay(row[5] if len(row) > 5 and row[5] else {}, arch)
        if not enable:
            continue
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
                    skip_reason=opts.get("skip"),
                )
            )
    return params
