###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A dLLM-Sim run from one YAML file.

.. code-block:: yaml

    model: llada_8B              # a dLLM preset, a Megatron preset, or a dict
    model_overrides: {}
    dtypes: {weight: bf16, act: bf16, kv_cache: bf16}
    hardware: {arch: mi355x, gemm_backend: auto}   # plus any DLLMHardware field
    parallel: {tp: 8, pp: 1, ep: 1, replicas: 1, pp_microbatches: 0}
    workload: {prompt_len: 512, gen_len: 512, block_len: 32, steps: 128,
               cache: dual, logits: all, cfg: false}
    comm: {payload_dtype: fp8, quant_scope: inter_node, overlap: 0.0}
    contention: {enabled: true, comm_cu_fraction: 0.0625, hbm_traffic_factor: 3.0}
    run:
      static_batches: [1, 8, 32]
      serving: {rates: [1, 4, 16], num_requests: 200, batch_limit: 0, seed: 0}

Every section is optional; a missing one takes the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .comm import CommConfig
from .contention import ContentionConfig
from .hardware import DLLMHardware
from .model import DLLMModelSpec, Parallel, load_model, with_dtypes
from .workload import DLLMWorkload

_SECTIONS = {
    "model",
    "model_overrides",
    "dtypes",
    "hardware",
    "parallel",
    "workload",
    "comm",
    "contention",
    "run",
}


def _build(cls: type, d: dict[str, Any] | None, section: str) -> Any:
    d = dict(d or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"[dllm-sim] unknown key(s) in {section}: {', '.join(unknown)}")
    return cls(**d)


@dataclass
class DLLMRun:
    spec: DLLMModelSpec
    hw: DLLMHardware
    par: Parallel
    workload: DLLMWorkload
    comm: CommConfig
    contention: ContentionConfig
    static_batches: list[int] = field(default_factory=lambda: [1])
    serving: dict[str, Any] = field(default_factory=dict)


def from_dict(cfg: dict[str, Any]) -> DLLMRun:
    unknown = sorted(set(cfg) - _SECTIONS)
    if unknown:
        raise ValueError(f"[dllm-sim] unknown section(s): {', '.join(unknown)}")
    model = cfg.get("model", "llada_8B")
    if isinstance(model, dict):
        spec = DLLMModelSpec.from_dict({**model, **(cfg.get("model_overrides") or {})})
    else:
        spec = load_model(str(model), cfg.get("model_overrides"))
    dt = dict(cfg.get("dtypes") or {})
    bad = sorted(set(dt) - {"weight", "act", "kv_cache"})
    if bad:
        raise ValueError(f"[dllm-sim] unknown key(s) in dtypes: {', '.join(bad)}")
    spec = with_dtypes(
        spec,
        weight_dtype=dt.get("weight"),
        act_dtype=dt.get("act"),
        kv_cache_dtype=dt.get("kv_cache"),
    )
    hw_cfg = dict(cfg.get("hardware") or {})
    gemm_backend = hw_cfg.pop("gemm_backend", "auto")
    hw = DLLMHardware.from_arch(hw_cfg.pop("arch", "mi355x"), **hw_cfg)
    hw = hw.with_gemm_backend(gemm_backend)
    run = dict(cfg.get("run") or {})
    batches = run.get("static_batches", [1])
    if isinstance(batches, int):
        batches = [batches]
    return DLLMRun(
        spec=spec,
        hw=hw,
        par=_build(Parallel, cfg.get("parallel"), "parallel"),
        workload=_build(DLLMWorkload, cfg.get("workload"), "workload"),
        comm=_build(CommConfig, cfg.get("comm"), "comm"),
        contention=_build(ContentionConfig, cfg.get("contention"), "contention"),
        static_batches=[int(b) for b in batches],
        serving=dict(run.get("serving") or {}),
    )


def load_config(path: str) -> DLLMRun:
    from infera.projection.core.config.yaml_loader import parse_yaml

    return from_dict(parse_yaml(path) or {})
