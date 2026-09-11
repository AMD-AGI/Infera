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
sequences, since every resident request advances one token per step. The
decode curve is anchored from that directly.

Prefill is anchored too, but never from an absolute TTFT: mean TTFT looks like
the matching prefill observable but is dominated by streaming and admission
granularity, so inverting it yields a "prefill step" one to two orders of
magnitude too large. Differencing escapes that. Those contaminating terms are
all constant in prompt length, so measuring TTFT at two prompt lengths and
subtracting cancels them and leaves the cost of the extra tokens -- see
:func:`prefill_rate_ms_per_token`. ``--no-prefill-anchor`` restores the
historical behaviour of leaving prefill simulated.

Leaving prefill simulated is not free, and the flag exists because of what it
costs. The analytical GEMM backend carries a large absolute bias (~5x; see the
origami-ratio note in ``performance.py``). Decode never pays it, because a
measured anchor means the simulator is only ever consulted as the ratio
sim(target)/sim(bench), in which the bias cancels. Unanchored prefill has no
such quotient and pays the bias in full, which is what makes projected TTFT run
several times faster than measured.

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
import shutil
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
        argv = [
            "--model-path",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp",
            str(tp),
        ]
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
        argv = [
            "--model",
            args.model,
            "--host",
            "127.0.0.1",
            "--server-port",
            str(port),
            "--port",
            str(_free_port()),
            "--tensor-parallel-size",
            str(tp),
        ]
        if args.max_model_len:
            argv += ["--max-model-len", str(args.max_model_len)]
        if args.enable_expert_parallel:
            argv += ["--enable-expert-parallel"]
        if args.enforce_eager:
            argv += ["--enforce-eager"]
    else:
        argv = [
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp),
        ]
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

        return AtomEngine(atom_argv=argv, model_name=args.model, host="127.0.0.1", port=port)
    from infera.engine.vllm.worker import VllmEngine

    return VllmEngine(vllm_argv=argv, model_name=args.model, host="127.0.0.1", port=port)


def _run_client(
    port: int,
    args,
    out_dir: str,
    tag: str,
    *,
    batch: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
) -> dict:
    """One closed-loop client run against the live server; its whole result."""
    result = os.path.join(out_dir, f"bench_{tag}.json")
    kind = client_kind(args)
    if kind == "vllm":
        # vLLM's client drives either server. Against SGLang it goes through the
        # plain OpenAI completions route rather than vLLM's own.
        client = "vllm" if args.serving_backend == "vllm" else "openai"
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            client,
            "--model",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--endpoint",
            "/v1/completions",
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(batch),
            "--ignore-eos",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--save-result",
            "--result-filename",
            result,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang-oai",
            "--model",
            args.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            # Exact lengths, not a sampled band. The anchor prices one prompt
            # length and one output length; left at its default this client
            # samples below both, and the TPOT it reported would belong to a
            # mixture of shapes rather than to the shape recorded beside it.
            "--random-range-ratio",
            "1.0",
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(batch),
            "--output-file",
            result,
        ]
        # This client appends, so a stale file from an earlier harvest at the
        # same tag would leave its last line -- another run's numbers -- as the
        # one read back.
        with contextlib.suppress(FileNotFoundError):
            os.remove(result)
    subprocess.run(cmd, check=True)
    return _client_result(result, kind)


def client_kind(args) -> str:
    """Which load generator drives the server.

    What a client owes this harness is two numbers -- mean TTFT and mean TPOT
    over a closed loop -- and both engines ship a script reporting them under
    those names. vLLM's drives either server over the plain OpenAI route, so it
    stays the default wherever it exists and every anchor already harvested
    keeps comparing against the ones harvested next.

    It does not always exist. An SGLang image need not contain vLLM at all, and
    the architectures worth measuring under SGLang are exactly the ones the
    local vLLM build cannot load -- so insisting on vLLM's client would make a
    model unmeasurable for want of a load generator rather than for want of an
    engine that can serve it. SGLang's own client stands in there, and the
    artifact records which one ran, because two clients pacing one server are
    two measurements.
    """
    if shutil.which("vllm"):
        return "vllm"
    if args.serving_backend == "sglang":
        return "sglang"
    raise RuntimeError(
        f"no load generator available: the vllm CLI is not on PATH and "
        f"serving backend {args.serving_backend!r} ships no client this "
        f"harness can drive"
    )


def _client_result(path: str, kind: str) -> dict:
    """One run's metrics, however its client chose to write them down."""
    with open(path) as fh:
        if kind == "vllm":
            return json.load(fh)
        # SGLang writes JSON Lines, one object per run appended to the file.
        lines = [line for line in fh.read().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"client wrote no result to {path}")
    return json.loads(lines[-1])


def _measure_concurrency(port: int, batch: int, args, out_dir: str) -> float:
    """Mean TPOT in ms at ``batch`` concurrent requests -- the decode step."""
    # Three waves is enough for the anchor concurrencies: re-running c128 with
    # ten waves and varied lengths moved TPOT by 5%, and c<=32 by less.
    doc = _run_client(
        port,
        args,
        out_dir,
        f"c{batch}",
        batch=batch,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=max(24, batch * 3),
    )
    return float(doc["mean_tpot_ms"])


# The prompt-length probes end at the first token, so the tail after it is pure
# wall time: a short output keeps a 128k probe from also decoding a full answer.
# Both probes share the value, which is what matters -- an identical tail is one
# more constant the difference cancels.
_PREFILL_PROBE_OUTPUT_LEN = 4
_PREFILL_PROBE_PROMPTS = 12
# Below this the difference is comparable to run-to-run TTFT noise and the slope
# is not resolvable.
_PREFILL_MIN_TOKEN_DELTA = 256


def prefill_probe_lengths(args) -> list[int]:
    """Prompt lengths to probe, shortest first, ending at the anchor's own.

    The long point is ``input_len`` itself so the rate is interpolated over the
    lengths the anchor is used at rather than extrapolated past them.
    """
    long_len = int(args.input_len)
    short = int(getattr(args, "prefill_anchor_short", 0) or 0) or long_len // 2
    short = max(1, min(short, long_len - _PREFILL_MIN_TOKEN_DELTA))
    lengths = [short, long_len]
    if getattr(args, "prefill_anchor_validate", False):
        # A third, interior point turns the assumption into something checkable:
        # with one pair the slope is whatever two numbers say, with two pairs
        # their disagreement measures how far from linear the prompt curve is.
        lengths.insert(1, (short + long_len) // 2)
    return lengths


def prefill_rate_ms_per_token(port: int, args, out_dir: str) -> tuple:
    """Per-token prefill cost, by differencing TTFT across prompt lengths.

    Absolute TTFT is not an invertible prefill observable -- it carries the
    scheduler's admission granularity, the streaming flush and the client's own
    overhead, and inverting it yields a prefill step one to two orders of
    magnitude too large. Every one of those terms is constant in prompt length,
    so they cancel in a difference: at a fixed concurrency and output length,
    ``TTFT(L2) - TTFT(L1)`` is the cost of the extra ``L2 - L1`` prompt tokens
    and nothing else.

    Probed at concurrency 1, where no request waits behind another's prefill and
    the difference is therefore compute rather than queueing. The rate is a
    cache-miss rate: the projector applies its own prefix-hit discount on top,
    so a probe that hit the prefix cache would be discounted twice.

    Returns ``(rate_ms_per_token, diagnostics)``, with a rate of 0.0 when the
    slope does not resolve -- the caller then leaves prefill simulated rather
    than anchoring on noise.
    """
    lengths = prefill_probe_lengths(args)
    pts = []
    for length in lengths:
        doc = _run_client(
            port,
            args,
            out_dir,
            f"prefill_L{length}",
            batch=1,
            input_len=length,
            output_len=_PREFILL_PROBE_OUTPUT_LEN,
            num_prompts=_PREFILL_PROBE_PROMPTS,
        )
        ttft = float(doc["mean_ttft_ms"])
        pts.append((length, ttft))
        print(f"[inferasim:Inference:Serving] prefill probe L={length} TTFT={ttft:.2f}ms")

    # Every adjacent pair is its own estimate of the slope; consistency between
    # them is the evidence that the cancelled terms really were constant.
    pairwise = [
        {
            "from": pts[i][0],
            "to": pts[i + 1][0],
            "ms_per_token": (pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]),
        }
        for i in range(len(pts) - 1)
    ]
    # Least squares over all points; identical to the lone difference when there
    # are only two, and a better estimate than any single pair when there are more.
    n = len(pts)
    mean_x = sum(x for x, _ in pts) / n
    mean_y = sum(y for _, y in pts) / n
    var = sum((x - mean_x) ** 2 for x, _ in pts)
    rate = (sum((x - mean_x) * (y - mean_y) for x, y in pts) / var) if var else 0.0

    diag = {
        "method": "ttft difference across prompt lengths",
        "concurrency": 1,
        "output_len": _PREFILL_PROBE_OUTPUT_LEN,
        "num_prompts": _PREFILL_PROBE_PROMPTS,
        "points": [{"input_len": x, "mean_ttft_ms": y} for x, y in pts],
        "pairwise_ms_per_token": pairwise,
        "ms_per_token": rate,
        # What the constant terms actually came to, as a sanity read: the
        # intercept is the admission/streaming/client floor the difference threw
        # away. A negative one means the probes were not on a straight line.
        "implied_fixed_ms": mean_y - rate * mean_x,
    }
    rates = [p["ms_per_token"] for p in pairwise]
    if len(rates) > 1 and min(rates) > 0:
        spread = max(rates) / min(rates)
        diag["pairwise_spread"] = spread
        if spread > 1.25:
            print(
                f"[inferasim:Inference:Serving] WARNING: prefill probes "
                f"disagree by {spread:.2f}x across length ({rates}); the "
                f"prompt curve is not linear over this range, so the anchor "
                f"is a chord through it rather than a rate."
            )
    if rate <= 0:
        print(
            "[inferasim:Inference:Serving] WARNING: prefill slope did not "
            "resolve (non-increasing TTFT across length); leaving prefill "
            "simulated."
        )
        return 0.0, diag
    return rate, diag


def run_serving_benchmark(args) -> dict:
    """Launch a server, sweep the requested concurrencies, return an anchor."""
    # Package import; falls back to the flat form when this is run as a script,
    # which is how Hyperloom invokes it.
    try:
        from .benchmark_vllm import (
            _capture_batches_up_to,
            _default_capture_sizes,
            _regime_env,
            _resolved_weight_dtype,
            _server_arg_value,
            warmup_gpu_count,
        )
    except ImportError:
        from benchmark_vllm import (
            _capture_batches_up_to,  # type: ignore
            _default_capture_sizes,
            _regime_env,
            _resolved_weight_dtype,
            _server_arg_value,
            warmup_gpu_count,
        )

    # Resolved before anything is launched. A missing load generator makes the
    # run pointless, and finding that out after the weights are resident costs
    # minutes to learn something knowable now.
    client = client_kind(args)

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
        print(
            f"[inferasim:Inference:Serving] concurrency={concurrency} -> "
            f"capture-size batches {batches}"
        )
    elif args.batches:
        batches = [int(b) for b in args.batches.split(",") if b]
    else:
        batches = [int(args.batch)]
    port = _free_port()
    argv = _engine_argv(args, port, bench_tp)
    engine = _build_engine(args, argv, port, bench_tp)

    out_dir = tempfile.mkdtemp(prefix="inferasim_serving_")
    log_path = os.path.join(out_dir, "server.log")
    print(
        f"[inferasim:Inference:Serving] {args.serving_backend} "
        f"{' '.join(shlex.quote(c) for c in argv)}"
    )
    started = time.time()
    try:
        with _capture_engine_output(log_path):
            asyncio.run(engine.start())
    except Exception as exc:
        # The adapter tears down its own process group on a failed start; this
        # only makes sure a half-started engine cannot keep holding the GPUs.
        asyncio.run(engine.stop())
        raise RuntimeError(f"{args.serving_backend} did not come up; see {log_path}") from exc
    boot_s = time.time() - started
    print(f"[inferasim:Inference:Serving] ready in {boot_s:.0f}s")
    try:
        client_started = time.time()
        sweep = [
            {"batch": b, "decode_ms": _measure_concurrency(port, b, args, out_dir)} for b in batches
        ]
        prefill_rate, prefill_diag = (
            prefill_rate_ms_per_token(port, args, out_dir)
            if getattr(args, "prefill_anchor", False)
            else (0.0, None)
        )
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
    # One measured rate fills the whole batch curve. Prefill is compute-bound and
    # linear in total prompt tokens -- the assumption the projector's own
    # per-token path already makes -- so stating it for every batch keeps the two
    # consumption paths agreeing. A lone point would instead be held flat across
    # batch by the measured batch transport, which for prefill is the one shape
    # it is certainly not.
    if prefill_rate > 0:
        for entry in sweep:
            entry["prefill_ms"] = prefill_rate * entry["batch"] * args.input_len
    artifact = {
        "backend": args.serving_backend,
        # Which client paced the server. Two clients driving one engine are two
        # measurements, so this is part of what the anchor describes.
        "client": client,
        # prefill_ms comes from differencing TTFT across prompt lengths, and is
        # None under --no-prefill-anchor or when the slope did not resolve. A
        # raw TTFT is not an invertible prefill observable, and is never
        # inverted here.
        "measured": {"model": {"prefill_ms": ref.get("prefill_ms"), "decode_ms": ref["decode_ms"]}},
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
            "attention_backend": _server_arg_value(args.server_args or "", "--attention-backend"),
            "load_format": "auto",
            "real_weights": True,
            "model": args.model,
            # What this anchor cost, so its own artifact carries the accounting.
            "boot_s": round(boot_s, 1),
            "anchor_client_s": round(client_s, 1),
            "derived_from": (
                "serving benchmark (mean TPOT; prefill by TTFT difference across prompt lengths)"
                if prefill_rate > 0
                else "serving benchmark (mean TPOT)"
            ),
            "prefill_anchor": prefill_diag,
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
                recipe_from_bench_args,
                regime_signature,
            )
        artifact["meta"]["regime_signature"] = regime_signature(
            recipe_from_bench_args(args, _regime_env())
        )
    except Exception:  # noqa: BLE001 - signature is an index hint, not a result
        pass
    return artifact
