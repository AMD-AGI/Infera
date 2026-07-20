###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""infera-kvd — node-local KV cache daemon.

Phase 3.0 (this skeleton): host RAM only, priority-aware LRU, UDS IPC.
Phase 3.5+: SSD spillover + long regions, restart recovery, NIXL.

This package is engine-agnostic. Engine adapters (SGLang's
`HiCacheStorage` backend, vLLM's `KVConnectorBase_V1` impl) translate
between engine-internal block representations and kvd's wire format.
They live under `infera.engine.{sglang,vllm}.kvd_adapter`.
"""
