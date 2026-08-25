###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Measure a serving recipe on real GPUs, for ``--profiling-mode benchmark``.

The projector calibrates against measured step latencies in one artifact
schema, and the vLLM/SGLang harness behind ``inferasim anchor`` already emits
exactly that. So benchmark mode is that harness driven from the projection's
own config instead of a hand-written command line: measure this recipe now,
calibrate against it, and leave the artifact on disk so later runs can skip the
GPUs entirely via ``--load-benchmark``.

The one thing a structural config cannot supply is which weights to serve --
it describes an architecture, not a checkpoint -- so that comes from
``--bench-model``. Weights are served random by default, since the measurement
wants kernel timings rather than answers.
"""

import json
import os
import tempfile


def _resolve_bench_model(args) -> str:
    """Which checkpoint to serve while measuring."""
    model = getattr(args, "bench_model", None) or os.environ.get(
        "INFERASIM_BENCH_MODEL"
    )
    if model:
        return model
    # INFERASIM_MODEL carries a preset spelling ("gpt_oss_120B") for matching
    # anchors, which is a checkpoint id only when it happens to look like one.
    preset = os.environ.get("INFERASIM_MODEL", "")
    if "/" in preset or os.path.exists(preset):
        return preset
    raise ValueError(
        "[inferasim:Inference] --profiling-mode benchmark measures a real "
        "checkpoint, which a structural config does not name. Pass "
        "--bench-model <hf-id-or-path>, or set INFERASIM_BENCH_MODEL."
    )


def spawn_inference_benchmark(args, inference_config):
    """Measure ``inference_config`` on GPUs; return the anchor artifact dict."""
    from . import benchmark_vllm

    mp = inference_config.model_parallel_config
    req = inference_config.request_config
    save = getattr(args, "save_benchmark", None) or os.path.join(
        tempfile.gettempdir(), "inferasim_bench.json"
    )

    argv = [
        "--model", _resolve_bench_model(args),
        "--save", str(save),
        "--tp", str(int(mp.tensor_model_parallel_size)),
        "--pp", str(int(getattr(mp, "pipeline_model_parallel_size", 1) or 1)),
        "--input-len", str(int(req.input_seq_len)),
        "--output-len", str(int(req.output_seq_len)),
        # --concurrency, not --batch: it sweeps the engine's own CUDA-graph
        # capture ladder up to this batch and characterises decode against
        # context in the same engine build. A single batch point would leave the
        # projector holding one measurement flat across both axes, which is
        # wrong for anything asking off the measured point -- a load sweep, the
        # tuning agent, or a longer context.
        "--concurrency", str(int(req.max_concurrency or req.batch_size or 1)),
    ]
    # Measuring on fewer GPUs than the target is the expected case, not a
    # degraded one: the projector restores the step to the target parallelism.
    if getattr(args, "benchmark_gpus", None):
        argv += ["--benchmark-gpus", str(int(args.benchmark_gpus))]
    # Quantization and KV dtype are deliberately left to the harness, which
    # reads them from the checkpoint's own config -- a more reliable answer than
    # translating this package's dtype names into the engine's.

    print("[inferasim:Inference] measuring on GPUs: anchor " + " ".join(argv))
    benchmark_vllm.main(argv)
    with open(save) as fh:
        artifact = json.load(fh)
    print(f"[inferasim:Inference] measured anchor saved to {save}")
    return artifact
