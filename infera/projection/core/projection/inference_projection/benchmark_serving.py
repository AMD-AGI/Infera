###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Measure a calibration anchor against a real server, not the offline engine.

An anchor is only useful if it was measured on the machine the projection is
predicting, and offline vLLM is not that machine. Given identical flags,
``LLM()`` and ``vllm serve`` resolve different kernels for the two most
expensive operations -- on gpt-oss-120b/MI355X the offline engine picked
ROCM_AITER_FA + AITER_MXFP4_BF16 where the server picked
ROCM_AITER_UNIFIED_ATTN + TRITON. The served decode step was several times the
offline one, and the gap widened with concurrency because Triton MXFP4 scales
worse with batch. Anchoring on the offline number therefore predicted served
TPOT no better than not calibrating at all, while a served anchor transported
across parallelism tracked the served target closely.

The mapping is the one :func:`anchor_from_serving` documents: for a closed-loop
run at concurrency ``C``, mean TPOT *is* the steady-state decode step at ``C``
sequences, since every resident request advances one token per step. Only the
decode curve is anchored -- mean TTFT looks like the matching prefill
observable but is dominated by streaming and admission granularity, so
inverting it yields a "prefill step" one to two orders of magnitude too large.
Prefill stays simulated.

The artifact is recorded at the parallelism it actually ran at, with no
reduce/restore: the projector transports TP=4 to the target itself, which is
the path the 7.7% figure was measured on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time

# Kernel names vLLM prints once it has resolved them. Recorded on the anchor so
# an artifact can be told apart from one measured on a different stack.
_ATTENTION_RE = re.compile(r"Overriding with ([A-Z0-9_]+)|Using ([A-Z0-9_]+) backend")
_MOE_RE = re.compile(r"Using '([A-Za-z0-9_]+)' Mxfp4 MoE backend")

def resolved_kernels(log_text: str) -> dict:
    """The attention and MoE backends vLLM actually chose, from its own log.

    vLLM's phrasing only. Against another engine both come back ``None``, which
    reads as "not recorded" rather than as a false match.
    """
    attn = _ATTENTION_RE.search(log_text)
    moe = _MOE_RE.search(log_text)
    return {
        "resolved_attention_backend": (attn.group(1) or attn.group(2)) if attn else None,
        "resolved_moe_backend": moe.group(1) if moe else None,
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def _capture_engine_output(path: str):
    """Send the engine's log to ``path`` while it starts.

    The adapters hand the child our own stdout/stderr, which is what a worker
    wants in production. Pointing ours at a file for the duration of the spawn
    is therefore the way to keep the log without giving the platform's launch
    path a benchmark-only parameter. The child holds the redirected descriptors
    for its whole life, so it keeps logging here after ours are restored.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    with open(path, "w") as fh:
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(fh.fileno(), 1)
            os.dup2(fh.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _engine_argv(args, port: int, tp: int) -> list[str]:
    """The flags one engine wants for the intent every engine shares.

    Only the spelling differs; all three expose ``/health`` and an
    OpenAI-compatible ``/v1/completions``, so readiness and load generation are
    shared. Caller flags come last so they win over anything derived here.
    """
    if args.serving_backend == "sglang":
        argv = ["--model-path", args.model, "--host", "127.0.0.1",
                "--port", str(port), "--tp", str(tp)]
        if args.max_model_len:
            argv += ["--context-length", str(args.max_model_len)]
        if args.enable_expert_parallel:
            argv += ["--enable-ep-moe"]
        if args.enforce_eager:
            argv += ["--disable-cuda-graph"]
    elif args.serving_backend == "atom":
        # ATOM splits the two ports the other engines fold together: --port is
        # the torch-distributed MASTER_PORT, so the HTTP listener the client
        # and the health probe use is --server-port. Sharing one number here
        # would collide the rendezvous with the API.
        argv = ["--model", args.model, "--host", "127.0.0.1",
                "--server-port", str(port), "--port", str(_free_port()),
                "--tensor-parallel-size", str(tp)]
        if args.max_model_len:
            argv += ["--max-model-len", str(args.max_model_len)]
        if args.enable_expert_parallel:
            argv += ["--enable-expert-parallel"]
        if args.enforce_eager:
            argv += ["--enforce-eager"]
    else:
        argv = [args.model, "--host", "127.0.0.1", "--port", str(port),
                "--tensor-parallel-size", str(tp)]
        if args.max_model_len:
            argv += ["--max-model-len", str(args.max_model_len)]
        if args.enable_expert_parallel:
            argv += ["--enable-expert-parallel"]
        if args.enforce_eager:
            argv += ["--enforce-eager"]
    if args.quantization:
        argv += ["--quantization", args.quantization]
    if args.kv_cache_dtype:
        argv += ["--kv-cache-dtype", args.kv_cache_dtype]
    return argv + shlex.split(args.server_args or "")


def _build_engine(args, argv: list[str], port: int, tp: int):
    """The engine to measure, launched the way the platform launches it.

    Going through the engine adapters instead of a local command line is what
    lets an anchor cover ATOM at all, and it keeps the launch -- including
    adapter-side decisions such as SGLang's forced ``--enable-metrics`` -- the
    same one that serves production traffic. Each import is deferred to its own
    branch because an adapter may need its engine present just to import.
    """
    if args.serving_backend == "sglang":
        from sglang.srt.server_args import ServerArgs

        from infera.engine.sglang.worker import SglangEngine

        return SglangEngine(
            ServerArgs(model_path=args.model, host="127.0.0.1", port=port, tp_size=tp),
            sglang_argv=argv,
        )
    if args.serving_backend == "atom":
        from infera.engine.atom.worker import AtomEngine

        return AtomEngine(atom_argv=argv, model_name=args.model,
                          host="127.0.0.1", port=port)
    from infera.engine.vllm.worker import VllmEngine

    return VllmEngine(vllm_argv=argv, model_name=args.model,
                      host="127.0.0.1", port=port)


def _measure_concurrency(port: int, batch: int, args, out_dir: str) -> float:
    """Mean TPOT in ms at ``batch`` concurrent requests -- the decode step."""
    result = os.path.join(out_dir, f"bench_c{batch}.json")
    # Three waves is enough for the anchor concurrencies: re-running c128 with
    # ten waves and varied lengths moved TPOT by 5%, and c<=32 by less.
    num_prompts = max(24, batch * 3)
    # vLLM's client drives either server. Against SGLang it goes through the
    # plain OpenAI completions route rather than vLLM's own.
    client = "vllm" if args.serving_backend == "vllm" else "openai"
    cmd = [
        "vllm", "bench", "serve", "--backend", client, "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port), "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(args.input_len),
        "--random-output-len", str(args.output_len),
        "--num-prompts", str(num_prompts), "--max-concurrency", str(batch),
        "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result", "--result-filename", result,
    ]
    subprocess.run(cmd, check=True)
    with open(result) as fh:
        return float(json.load(fh)["mean_tpot_ms"])


def run_serving_benchmark(args) -> dict:
    """Launch a server, sweep the requested concurrencies, return an anchor."""
    # Package import; falls back to the flat form when this is run as a script,
    # which is how Hyperloom invokes it.
    try:
        from .benchmark_vllm import (_capture_batches_up_to,
                                     _default_capture_sizes, _regime_env,
                                     _resolved_weight_dtype, _server_arg_value,
                                     warmup_gpu_count)
    except ImportError:
        from benchmark_vllm import (_capture_batches_up_to,  # type: ignore
                                    _default_capture_sizes, _regime_env,
                                    _resolved_weight_dtype, _server_arg_value,
                                    warmup_gpu_count)

    target_tp = max(1, int(args.tp or 1))
    target_pp = max(1, int(args.pp or 1))
    target_ep = target_tp if args.enable_expert_parallel else 1
    bench_tp = int(args.benchmark_gpus or warmup_gpu_count(target_tp))

    # --concurrency sweeps the capture ladder rather than one batch, so the
    # projector can pad a batch UP to a measured point instead of holding a
    # single measurement flat. The engine runs in another process, so its real
    # capture list is unreadable from here; a default launch captures the
    # default ladder, which is what _default_capture_sizes mirrors.
    concurrency = int(getattr(args, "concurrency", None) or 0)
    capture_sizes = _default_capture_sizes(concurrency) if concurrency else None
    if concurrency:
        batches = _capture_batches_up_to(capture_sizes, concurrency)
        print(f"[inferasim:Inference:Serving] concurrency={concurrency} -> "
              f"capture-size batches {batches}")
    elif args.batches:
        batches = [int(b) for b in args.batches.split(",") if b]
    else:
        batches = [int(args.batch)]
    port = _free_port()
    argv = _engine_argv(args, port, bench_tp)
    engine = _build_engine(args, argv, port, bench_tp)

    out_dir = tempfile.mkdtemp(prefix="inferasim_serving_")
    log_path = os.path.join(out_dir, "server.log")
    print(f"[inferasim:Inference:Serving] {args.serving_backend} "
          f"{' '.join(shlex.quote(c) for c in argv)}")
    started = time.time()
    try:
        with _capture_engine_output(log_path):
            asyncio.run(engine.start())
    except Exception as exc:
        # The adapter tears down its own process group on a failed start; this
        # only makes sure a half-started engine cannot keep holding the GPUs.
        asyncio.run(engine.stop())
        raise RuntimeError(
            f"{args.serving_backend} did not come up; see {log_path}"
        ) from exc
    boot_s = time.time() - started
    print(f"[inferasim:Inference:Serving] ready in {boot_s:.0f}s")
    try:
        client_started = time.time()
        sweep = [{"batch": b,
                  "decode_ms": _measure_concurrency(port, b, args, out_dir)}
                 for b in batches]
        client_s = time.time() - client_started
    finally:
        asyncio.run(engine.stop())

    with open(log_path) as fh:
        kernels = resolved_kernels(fh.read())
    # In capture mode the anchor point is the bucket covering the concurrency,
    # which _capture_batches_up_to leaves last.
    if concurrency:
        ref = sweep[-1]
    else:
        ref = next((e for e in sweep if e["batch"] == args.batch), sweep[0])
    artifact = {
        "backend": args.serving_backend,
        # prefill_ms stays None: TTFT is not an invertible prefill observable.
        "measured": {"model": {"prefill_ms": None, "decode_ms": ref["decode_ms"]}},
        "sweep": sweep,
        "meta": {
            "batch": ref["batch"],
            "input_len": args.input_len,
            "output_len": args.output_len,
            # Recorded at the parallelism it ran at; the projector transports it.
            "tp": bench_tp,
            "ep": min(target_ep, bench_tp),
            "pp": 1,
            "target_tp": target_tp,
            "target_pp": target_pp,
            "benchmark_gpus": bench_tp,
            "quantization": args.quantization,
            "weight_dtype": _resolved_weight_dtype(args),
            "kv_cache_dtype": args.kv_cache_dtype,
            "enforce_eager": args.enforce_eager,
            "use_aiter": os.environ.get("VLLM_ROCM_USE_AITER", "0") == "1",
            "server_args": args.server_args or None,
            "env_overrides": dict(kv.split("=", 1) for kv in args.env or []) or None,
            "attention_backend": _server_arg_value(args.server_args or "",
                                                   "--attention-backend"),
            "load_format": "auto",
            "real_weights": True,
            "model": args.model,
            # What this anchor cost, so its own artifact carries the accounting.
            "boot_s": round(boot_s, 1),
            "anchor_client_s": round(client_s, 1),
            "derived_from": "serving benchmark (mean TPOT)",
            # Capture-size sweep mode: the projector pads decode UP to the
            # nearest measured size instead of interpolating.
            "concurrency": concurrency or None,
            "capture_sizes": capture_sizes,
            "decode_pad_to_capture": bool(concurrency),
            **kernels,
        },
    }
    # Index the anchor by regime so the store can find it without re-deriving.
    try:
        try:
            from .search.regime import recipe_from_bench_args, regime_signature
        except ImportError:
            from search.regime import (  # type: ignore
                recipe_from_bench_args, regime_signature,
            )
        artifact["meta"]["regime_signature"] = regime_signature(
            recipe_from_bench_args(args, _regime_env())
        )
    except Exception:  # noqa: BLE001 - signature is an index hint, not a result
        pass
    return artifact
