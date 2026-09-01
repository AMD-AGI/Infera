###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference (serving) projection.

This package models *autoregressive inference*:

  * **Forward-only** compute — no backward pass, optimizer or gradients.
  * **Two phases** — prefill (prompt → first token, drives TTFT) and decode
    (autoregressive generation, drives inter-token latency / throughput).
  * **KV cache** memory — the dominant inference-time memory term.
  * **Serving features** — chunked prefill, KV-cache quantization,
    batching / concurrency, and speculative decoding.

The CLI entry point is :func:`launch_projection_from_cli`, wired from
``inferasim projection inference``.
"""

from .launcher import launch_projection_from_cli

__all__ = ["launch_projection_from_cli"]
