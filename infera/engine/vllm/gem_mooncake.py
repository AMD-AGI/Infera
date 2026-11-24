###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""vLLM's MooncakeConnector with two opt-in AMD deployment features.

Select it with ``"kv_connector": "MooncakeConnector"`` and
``"kv_connector_module_path": "infera.engine.vllm.gem_mooncake"``; the class
keeps the upstream name so the router protocol (``vllm-mooncake``) is unchanged.
With neither feature enabled it behaves exactly like upstream.

KV cache in amdgpu GEM buffer objects (``INFERA_AMD_GEM_ALLOCATOR``).
amdgpu-dkms 6.16 .. 7.1 charges a KFD allocation pinned by an RDMA NIC through
dma-buf twice in the KFD limit (fixed upstream in torvalds/linux 9131cc85d0de),
and hosts without a peer-memory client can only register VRAM through dma-buf.
There a large KV cache leaves the engine "out of memory" well below its
utilization target. A GEM BO imported into the process is charged once.
``INFERA_AMD_GEM_ALLOCATOR`` names the allocator library, which must also be in
``LD_PRELOAD``: it answers HIP's pointer and address-range queries and the
dma-buf export for its own buffers, and refuses ``ibv_reg_mr`` on them.

Canonical KV (``"canonical_kv": true`` in ``kv_connector_extra_config``).
vLLM lets attention backends keep fast paths that assume no foreign bytes
(e.g. NaNs in unwritten slots) reach the KV cache only for connectors whose
``only_restores_canonical_kv`` is true; upstream Mooncake does not claim it, so
on ROCm DeepSeek-V4.1 a P/D pair loses the aiter sparse MLA kernel and, with it,
``--kv-cache-dtype fp8_e4m3``. When prefill and decode run one model with one KV
cache layout, every block Mooncake moves is a whole block the producer's own
cache writer wrote, so the claim holds. Set it only for such a pair.
"""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector as _UpstreamMooncakeConnector,
)

logger = logging.getLogger(__name__)

ALLOCATOR_ENV = "INFERA_AMD_GEM_ALLOCATOR"
CANONICAL_KV_KEY = "canonical_kv"


class MooncakeConnector(_UpstreamMooncakeConnector):
    _gem_pool: torch.cuda.MemPool | None = None

    @classmethod
    def only_restores_canonical_kv(cls, kv_transfer_config) -> bool:
        extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
        return bool(extra.get(CANONICAL_KV_KEY, False))

    def get_mem_pool_context(self) -> AbstractContextManager | None:
        so = os.environ.get(ALLOCATOR_ENV)
        if not so:
            return super().get_mem_pool_context()
        if so not in os.environ.get("LD_PRELOAD", ""):
            raise RuntimeError(f"{ALLOCATOR_ENV}={so} must also be in LD_PRELOAD")
        if self._gem_pool is None:
            allocator = torch.cuda.memory.CUDAPluggableAllocator(so, "hip_gem_alloc", "hip_gem_free")
            self._gem_pool = torch.cuda.MemPool(allocator.allocator())
            logger.info("KV cache allocated from amdgpu GEM BOs (%s)", so)
        return torch.cuda.use_mem_pool(self._gem_pool)
