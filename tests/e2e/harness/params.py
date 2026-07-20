###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine-agnostic parameter set for e2e tests.

An :class:`EngineParams` captures the high-level knobs a test exercises; each
engine's adapter (see :mod:`.adapter`) translates them into that engine's
launcher flags, so adding a backend means writing an adapter — not touching the
tests or this dataclass.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class DisaggRole(enum.Enum):
    """A worker's role in a prefill/decode-disaggregated deployment.

    ``value`` is the human label used in logs / container names; :meth:`kv_role`
    is the vLLM ``kv_transfer_config`` role string, and :meth:`is_prefill` gates
    role-only launch flags (e.g. Mooncake's bootstrap server runs on prefill).
    Engine adapters translate this into their own disaggregation flags.
    """

    PREFILL = "prefill"
    DECODE = "decode"

    @property
    def is_prefill(self) -> bool:
        return self is DisaggRole.PREFILL

    def kv_role(self) -> str:
        """vLLM ``kv_transfer_config.kv_role`` (see infera/engine/vllm/args.py)."""
        return "kv_producer" if self is DisaggRole.PREFILL else "kv_consumer"


@dataclass(frozen=True)
class EngineParams:
    model: str = DEFAULT_MODEL
    tensor_parallel_size: int = 1
    expert_parallel: bool = False
    dp_attention: bool = False
    # Whether `model` is MoE — gates the expert-parallel case (ep is a no-op /
    # error on a dense model). Set by the matrix.
    is_moe: bool = False
    # Verbatim extra launch args / worker env (as (key, value) pairs since a
    # frozen dataclass can't hold a dict), set per-case by the matrix.
    extra_args: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = ()
    # Shell commands run (in the engine container) once before the worker is
    # launched, e.g. ("pip install amd-quark",) for a model needing an extra
    # runtime dep. Set per-case by the matrix.
    setup: tuple[str, ...] = ()
    # Seconds to wait for the worker to register active before failing. Per-case
    # (big MXFP4 MoE models at tp>1 need longer); default suits small/medium ones.
    server_ready_timeout: int = 300

    def id(self) -> str:
        """Compact, pytest-friendly parametrize id (e.g. ``Qwen3-0.6B-tp2``)."""
        bits = [self.model.split("/")[-1], f"tp{self.tensor_parallel_size}"]
        if self.expert_parallel:
            bits.append("ep")
        if self.dp_attention:
            bits.append("dpattn")
        return "-".join(bits)
