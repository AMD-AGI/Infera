###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""dLLM-Sim: performance projection for non-autoregressive (diffusion) LMs.

A refinement step is a prefill-shaped, bidirectional forward over the canvas;
this package schedules those steps per cache mode, prices them on one replica
(compute, collectives at their wire precision, overlap contention, pipeline
micro-batches), checks they fit, and simulates a server running them. See
``manual/simulation/dllm.md``.
"""

from .comm import CommConfig, DLLMCommModel
from .contention import ContentionConfig
from .hardware import DLLMHardware
from .memory import max_batch, memory_report
from .model import DLLMModelSpec, Parallel, SeqShape, build_stage_graph, dllm_presets, load_model
from .simulator import simulate_serving, simulate_static
from .step_model import DLLMStepModel, InferaSimBackend, RooflineBackend
from .workload import CACHE_MODES, DLLMWorkload

__all__ = [
    "CACHE_MODES",
    "CommConfig",
    "ContentionConfig",
    "InferaSimBackend",
    "DLLMCommModel",
    "DLLMHardware",
    "DLLMModelSpec",
    "DLLMStepModel",
    "DLLMWorkload",
    "Parallel",
    "RooflineBackend",
    "SeqShape",
    "build_stage_graph",
    "load_model",
    "max_batch",
    "memory_report",
    "dllm_presets",
    "simulate_serving",
    "simulate_static",
]
