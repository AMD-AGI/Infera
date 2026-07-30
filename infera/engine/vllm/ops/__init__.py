###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Infera vLLM op-injection plugin (issue #40).

Injects Infera/HyperLoom-optimized **Attention** and **MoE** kernels into stock
vLLM via a single out-of-tree ``vllm.general_plugins`` hook — no vLLM fork. The
hook (:func:`infera.engine.vllm.ops.register.register_ops`) runs after
``vllm.platforms`` is initialized and patches the resolved platform / MoE layer,
so it is import-safe. No-op when ``INFERA_VLLM_OPS_DISABLE=1`` or off-ROCm.

Seams:
  * Attention → :func:`infera.engine.vllm.ops.attention.install_attention_ops`
    (patches ``get_attn_backend_cls``; ``INFERA_ATTN_BACKEND`` selects a backend).
  * MoE experts → :func:`infera.engine.vllm.ops.moe.install_moe_ops`.
"""
