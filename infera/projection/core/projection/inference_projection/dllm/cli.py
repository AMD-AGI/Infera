###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""``inferasim dllm``: project a non-autoregressive model's serving performance."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from typing import Any

from .config import DLLMRun, from_dict, load_config
from .hardware import dtype_bytes
from .memory import max_batch
from .simulator import simulate_serving, simulate_static
from .step_model import DLLMStepModel


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inferasim dllm",
        description=(
            "Project refinement-step latency, throughput and memory of a "
            "masked-diffusion (non-autoregressive) LM on AMD GPUs."
        ),
    )
    p.add_argument("--config", help="dLLM-Sim YAML (see manual/simulation/dllm.md)")
    p.add_argument("--model", help="dLLM or Megatron preset, overriding the config")
    p.add_argument("--tp", type=int)
    p.add_argument("--pp", type=int)
    p.add_argument("--ep", type=int)
    p.add_argument("--cache", choices=["none", "prefix", "dual", "block"])
    p.add_argument("--steps", type=int, help="refinement steps over the canvas")
    p.add_argument("--batch", type=int, action="append", help="static batch (repeatable)")
    p.add_argument("--rate", type=float, action="append", help="serving req/s (repeatable)")
    p.add_argument("--json", help="also write the results here")
    return p


def _apply_flags(run: DLLMRun, a: argparse.Namespace) -> DLLMRun:
    par = replace(
        run.par,
        **{k: getattr(a, k) for k in ("tp", "pp", "ep") if getattr(a, k) is not None},
    )
    wl = run.workload
    if a.cache or a.steps:
        wl = replace(wl, cache=a.cache or wl.cache, steps=a.steps or wl.steps)
    serving = dict(run.serving)
    if a.rate:
        serving["rates"] = a.rate
    return replace(
        run,
        par=par,
        workload=wl,
        static_batches=a.batch or run.static_batches,
        serving=serving,
    )


def run(job: DLLMRun) -> dict[str, Any]:
    model = DLLMStepModel(job.spec, job.par, job.hw, job.comm, job.contention)
    wl = job.workload
    out: dict[str, Any] = {
        "model": job.spec.name,
        "gemm_backend": "origami" if job.hw.gemm_pricer is not None else "roofline",
        "parallel": dict(job.par.__dict__),
        "workload": dict(wl.__dict__),
        "max_batch_per_replica": max_batch(job.spec, job.par, job.hw, wl),
        "static": [],
        "serving": [],
    }
    for b in job.static_batches:
        out["static"].append(simulate_static(model, wl, b).as_dict())
    sv = job.serving
    # A replica sees its share of the cluster's arrivals.
    for rate in sv.get("rates", []):
        res = simulate_serving(
            model,
            wl,
            rate_per_s=float(rate) / job.par.replicas,
            num_requests=int(sv.get("num_requests", 200)),
            batch_limit=int(sv.get("batch_limit", 0)),
            seed=int(sv.get("seed", 0)),
        ).as_dict()
        res["cluster_rate_per_s"] = float(rate)
        out["serving"].append(res)
    worst = max(wl.schedule(), key=lambda s: s.shape.q).shape
    tokens = worst.q * wl.seqs_per_request * max(job.static_batches)
    # Report against the configured payload, or FP8 when the run sends BF16.
    payload = job.comm.payload_dtype
    if dtype_bytes(payload) >= dtype_bytes(job.spec.act_dtype):
        payload = "fp8"
    out["quantization_payload"] = payload
    out["quantization_tradeoff"] = {
        kind: model.comm.tradeoff(kind, tokens, payload)
        for kind in ("tp_allreduce",)
        if job.par.tp > 1
    }
    return out


def _print(res: dict[str, Any]) -> None:
    par = res["parallel"]
    wl = res["workload"]
    print(
        f"dLLM-Sim  {res['model']}  [{res['gemm_backend']} GEMMs]  "
        f"TP{par['tp']} PP{par['pp']} EP{par['ep']} "
        f"x{par['replicas']}  cache={wl['cache']}  P={wl['prompt_len']} G={wl['gen_len']} "
        f"block={wl['block_len']} steps={wl['steps']}  "
        f"max batch/replica={res['max_batch_per_replica']}"
    )
    print(
        f"{'batch':>6} {'latency ms':>11} {'1st block':>10} {'step ms':>8} "
        f"{'tok/s/GPU':>10} {'comm':>7} {'quant':>6} {'mem GB':>7}"
    )
    for s in res["static"]:
        b = s["breakdown"]
        print(
            f"{s['batch']:>6} {s['latency_ms']:>11.1f} {s['first_block_ms']:>10.1f} "
            f"{s['mean_step_ms']:>8.2f} {s['tokens_per_s_per_gpu']:>10.1f} "
            f"{b.get('exposed_comm_ms', 0.0):>7.2f} {b.get('quant_ms', 0.0):>6.2f} "
            f"{s['memory']['total_gb']:>7.1f}" + ("" if s["memory"]["fits"] else "  OOM")
        )
    if res["serving"]:
        print(
            f"{'req/s':>6} {'tok/s/GPU':>10} {'p50 ms':>9} {'p99 ms':>9} "
            f"{'p50 1st blk':>11} {'batch':>6} saturated"
        )
        for s in res["serving"]:
            lat, fb = s["latency_ms"], s["first_block_ms"]
            print(
                f"{s['cluster_rate_per_s']:>6g} {s['tokens_per_s_per_gpu']:>10.1f} "
                f"{lat.get('p50', 0):>9.1f} {lat.get('p99', 0):>9.1f} "
                f"{fb.get('p50', 0):>11.1f} {s['mean_batch']:>6.1f} {s['saturated']}"
            )
    for kind, t in res["quantization_tradeoff"].items():
        print(
            f"{kind} per call: plain {t['plain_ms']:.3f} ms, {res['quantization_payload']} "
            f"{t['quantized_wire_ms']:.3f} + pack {t['quantized_pack_ms']:.3f} ms "
            f"({'saves' if t['saving_ms'] > 0 else 'costs'} {abs(t['saving_ms']):.3f} ms)"
        )


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    job = load_config(a.config) if a.config else from_dict({})
    if a.model:
        from .model import load_model

        s = job.spec
        spec = replace(
            load_model(a.model),
            weight_dtype=s.weight_dtype,
            act_dtype=s.act_dtype,
            kv_cache_dtype=s.kv_cache_dtype,
        )
        job = replace(job, spec=spec)
    job = _apply_flags(job, a)
    res = run(job)
    _print(res)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, default=float)
    return 0
