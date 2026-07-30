###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Infera vLLM op-injection plugin (issue #40).

Injects Infera/HyperLoom-optimized **Attention** and **MoE** kernels into stock
vLLM via vLLM's out-of-tree plugin mechanism (``vllm.platform_plugins`` +
``vllm.general_plugins``) — no vLLM fork. Both hooks are no-ops when
``INFERA_VLLM_OPS_DISABLE=1``.

Seams:
  * Attention → :class:`infera.engine.vllm.ops.platform.InferaPlatform`
    (``get_attn_backend_cls``).
  * MoE experts → :func:`infera.engine.vllm.ops.moe.install_moe_ops`.
"""
