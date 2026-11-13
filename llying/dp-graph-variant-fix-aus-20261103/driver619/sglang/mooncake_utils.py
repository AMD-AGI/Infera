# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mooncake-specific utilities for custom memory pool management."""

import logging
import os
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Global constants for custom memory pool types
SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES = ["NVLINK", "BAREX", "INTRA_NODE_NVLINK"]


def _cuda_expandable_segments_enabled() -> Optional[str]:
    """Return the allocator env var that enables CUDA expandable segments."""
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        for field in os.environ.get(var, "").split(","):
            key, _, value = field.partition(":")
            if key.strip() == "expandable_segments" and value.strip().lower() == "true":
                return var
    return None


def _validate_efa_allocator_compatibility(
    enable_custom_mem_pool: bool, custom_mem_pool_type: Optional[str]
) -> None:
    """Reject CUDA VMM allocators unsupported by the current libfabric EFA path."""
    if envs.MOONCAKE_PROTOCOL.get().lower() != "efa":
        return

    if enable_custom_mem_pool and custom_mem_pool_type in ("NVLINK", "BAREX"):
        raise ValueError(
            f"SGLANG_MOONCAKE_CUSTOM_MEM_POOL={custom_mem_pool_type} is "
            "incompatible with MOONCAKE_PROTOCOL=efa. Mooncake custom memory "
            "pools use CUDA VMM allocations, which the current libfabric EFA "
            "provider cannot transfer. Unset SGLANG_MOONCAKE_CUSTOM_MEM_POOL."
        )

    expandable_segments_var = _cuda_expandable_segments_enabled()
    if expandable_segments_var is not None:
        raise ValueError(
            f"{expandable_segments_var} enables expandable_segments, which is "
            "incompatible with MOONCAKE_PROTOCOL=efa because the current "
            "libfabric EFA provider cannot transfer CUDA VMM allocations. "
            "Disable expandable_segments."
        )


_amd_gem_allocator = None


def _init_amd_gem_mem_pool(device: str) -> Tuple[bool, Any, str]:
    """KV pools in amdgpu GEM BOs instead of KFD BOs.

    amdgpu-dkms 6.18.4 through 7.1.9 charge KFD VRAM that an RDMA NIC pins twice in the KFD allocation limit, so
    registering a KV cache larger than the remaining headroom makes every later device allocation fail. GEM BOs are
    not in kfd.vram_used and are charged once. AMD_GEM is deliberately not in
    SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES: the transfer path stays the default one. The allocator library must
    also be in LD_PRELOAD (it answers the HIP / HSA queries Mooncake makes about its buffers) and the process must run
    with MOONCAKE_DISABLE_HIP_DMABUF=0, MC_HIP_KFD_PIN_FIX=0 and MC_DISABLE_HIP=1; it aborts otherwise.
    """
    global _amd_gem_allocator
    if _amd_gem_allocator is None:
        so = os.environ["SGLANG_AMD_GEM_ALLOCATOR"]
        _amd_gem_allocator = torch.cuda.memory.CUDAPluggableAllocator(
            so, "hip_gem_alloc", "hip_gem_free"
        )
    with torch.cuda.device(device):
        pool = torch.cuda.MemPool(_amd_gem_allocator.allocator())
    logger.info(f"KV memory pool on device {device}: amdgpu GEM BOs")
    return True, pool, "AMD_GEM"


def init_mooncake_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() == "AMD_GEM":
        return _init_amd_gem_mem_pool(device)

    enable_custom_mem_pool, custom_mem_pool_type = (
        check_mooncake_custom_mem_pool_enabled()
    )

    custom_mem_pool = None

    if enable_custom_mem_pool:
        try:
            # TODO(shangming): abstract custom allocator class for more backends
            if custom_mem_pool_type == "NVLINK":
                from mooncake.allocator import NVLinkAllocator

                allocator = NVLinkAllocator.get_allocator(device)
            elif custom_mem_pool_type == "BAREX":
                from mooncake.allocator import BarexAllocator

                allocator = BarexAllocator.get_allocator(device)
            elif custom_mem_pool_type == "INTRA_NODE_NVLINK":
                return False, None, None
            else:
                # This should not happen due to the enable_custom_mem_pool check above
                raise ValueError(
                    f"Unsupported custom mem pool type: {custom_mem_pool_type}"
                )

            # MemPool binds to the current device; on a non-main thread (e.g.
            # PD bootstrap) that is device 0, so ranks on other GPUs hit the
            # CUDACachingAllocator use_count assert in use_mem_pool().
            with torch.cuda.device(device):
                custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
            logger.debug(
                f"Initialized custom memory pool: {custom_mem_pool_type} on device {device}"
            )
        except ImportError as e:
            logger.warning(
                f"Failed to import mooncake allocator for {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
        except Exception as e:
            logger.error(
                f"Failed to initialize custom memory pool {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            enable_custom_mem_pool = False
            custom_mem_pool = None
            custom_mem_pool_type = None
    else:
        return False, None, None

    return enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type


def check_mooncake_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    """
    Check if custom memory pool is enabled without importing allocators.

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool_type)
    """
    custom_mem_pool_type = envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get()

    if custom_mem_pool_type is not None:
        # Handle boolean True as NVLINK
        if custom_mem_pool_type.lower() == "true":
            custom_mem_pool_type = "NVLINK"
        enable_custom_mem_pool = (
            custom_mem_pool_type in SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES
        )
    else:
        enable_custom_mem_pool = False
        custom_mem_pool_type = None

    return enable_custom_mem_pool, custom_mem_pool_type
