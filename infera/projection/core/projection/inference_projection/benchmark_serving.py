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
import statistics
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
        # vLLM/ATOM take the width ("fp8") and pick a representation; SGLang
        # only takes the representation, so the shared spelling has to be
        # resolved to one before it reaches the engine.
        kv = args.kv_cache_dtype
        if args.serving_backend == "sglang" and kv == "fp8":
            kv = "fp8_e4m3"
        argv += ["--kv-cache-dtype", kv]
    # An anchor describes kernels, not weights, so the harness serves random
    # ones by default -- but only the offline path was told. Read defensively:
    # this function is called with hand-built arg objects as well as parsed ones.
    load_format = getattr(args, "load_format", None)
    if load_format:
        if args.serving_backend == "atom":
            # ATOM has no --load-format. Random weights are --load_dummy, which
            # takes the fill instead of the loader name, and passing the vLLM
            # spelling takes the server down on argv parsing before it ever
            # reports ready.
            #
            # "empty" rather than "zero": zero-filling calls torch fill_, which
            # is not implemented for the packed MXFP4 dtype
            # (Float4_e2m1fn_x2) that quantized checkpoints carry, and the
            # engine dies during init. Uninitialized memory needs no fill and
            # an anchor does not care what the weights contain.
            if load_format == "dummy":
                argv += ["--load_dummy", "empty"]
            else:
                print(f"[inferasim:Inference:Serving] WARNING: ATOM has no "
                      f"--load-format; ignoring --load-format {load_format} "
                      f"and letting the engine load the checkpoint.")
        else:
            argv += ["--load-format", load_format]
    if getattr(args, "trust_remote_code", False):
        argv += ["--trust-remote-code"]
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
            # Synthetic ids, not "random": this client's "random" samples its
            # prompts out of ShareGPT and downloads it, so an offline or
            # air-gapped harvest fails in the client after the server is
            # already up. The anchor wants exact lengths, not real text.
            "--dataset-name",
            "random-ids",
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
    """Unblocked decode step in ms at ``batch`` concurrent requests.

    Median inter-token latency, not mean TPOT, and the distinction is the whole
    point. This run is a closed loop, so while it decodes it is also prefilling
    the requests its finished ones were replaced by, and under exclusive prefill
    every such prefill stalls decode completely. Those stalls are rare and huge
    -- p99.9 ITL reaches 1392 ms against a 21.7 ms median at 64 concurrent --
    so they barely move the median and dominate the mean.

    The simulator schedules those stalls itself. Handing it a mean would make it
    pay for them twice, and the double-count is not small: measured against the
    DeepSeek-V4-Pro TP8 ladder, mean TPOT ran over the median ITL by 1.136x at
    16 concurrent, 1.227x at 32 and 1.405x at 64, which is the same shape and
    very nearly the same size as the TPOT over-prediction it produced (+13%,
    +23%, +31%). What the scheduler models, the anchor must not also contain.

    Measured as a single wave -- ``num_prompts == batch``, so the loop never
    refills -- and that detail is what makes the number mean the same thing on
    both kinds of engine.

    With refill there is always a prefill in flight. Under exclusive prefill
    that costs a few enormous stalls which the median steps over, so three
    waves read correctly; under vLLM's chunked co-scheduling the prefill is
    spread thinly across many steps instead, so the *typical* step carries
    prefill work and the median moves with it. The consequence was not subtle:
    a three-wave DeepSeek-V4-Flash-0731 harvest on vLLM recorded 19.71 ms at
    batch 4 where the engine's own median inter-token latency at concurrency 4
    is 13.44, and its sweep climbed 12.99 -> 25.96 ms from batch 1 to 64 while
    the measurement climbs 13.30 -> 17.85. Scored against the corpus that read
    TPOT 51% high. The same protocol on SGLang landed within 3%, which is why
    the fault looked like a model difference rather than a measurement one.

    One wave has no such phase: after the initial prompts are through, every
    remaining step is decode for both schedulers, and the median lands in that
    steady state. Sample count is not the constraint it looks like -- a wave of
    ``batch`` requests generating ``output_len`` tokens yields
    ``batch * (output_len - 1)`` inter-token gaps, 1023 of them even at batch 1.
    """
    doc = _run_client(
        port,
        args,
        out_dir,
        f"c{batch}",
        batch=batch,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=batch,
    )
    itl = doc.get("median_itl_ms")
    if itl is None:
        # Older client builds do not report the ITL percentiles. Falling back
        # keeps the harvest alive, but the number means something different, so
        # say so rather than letting it pass for a decode step.
        print(
            f"[inferasim:Inference:Serving] WARNING: client reported no "
            f"median_itl_ms at concurrency {batch}; falling back to mean "
            f"TPOT, which includes the prefill stalls the simulator also "
            f"models. Expect TPOT to be over-predicted at high concurrency."
        )
        return float(doc["mean_tpot_ms"])
    return float(itl)


# The prompt-length probes end at the first token, so the tail after it is pure
# wall time: a short output keeps a 128k probe from also decoding a full answer.
# Both probes share the value, which is what matters -- an identical tail is one
# more constant the difference cancels.
_PREFILL_PROBE_OUTPUT_LEN = 4
# Samples per length probe. Twelve is enough when the prompt-length signal is
# large against TTFT's own spread, and not enough when it is not: at ISL 1024
# the whole 128..1024 sweep moves TTFT by about the same few milliseconds that
# run-to-run noise does, so the fit sees noise and the slope check rejects it
# -- DeepSeek-R1-0528 probed 77.6, 67.5, 74.6, 66.9 ms across increasing
# lengths and left prefill simulated, which is the fallback the anchor exists
# to avoid. The averaging only improves as sqrt(n), so raising this is the
# blunt instrument; the sharp one is probing at a length where prefill is a
# larger share of TTFT. Env-settable so a short-prompt harvest can pay for the
# samples without making every harvest pay.
_PREFILL_PROBE_PROMPTS = max(
    2, int(os.environ.get("INFERASIM_PREFILL_PROBE_PROMPTS", "12") or 12))
# Repeats per packed-probe point. The wave must be exactly as wide as the point
# under test, so the sample count per run is fixed by the point itself and p99
# is the max of that many; repeating the wave and taking the median restores a
# stable estimate without changing what is being estimated. Three is enough to
# reject a single outlier, which is what went wrong without it.
#
# Three is not enough when the packed points sit within a few ms of each other,
# which is what a short prompt does to them: MiniMax-M2.7 at ISL 1024 probed
# 70.5, 75.1, 75.2, 71.3 ms across 1..8 sequences, one run in three came back
# at 120.6 ms, and the fitted slope went negative and was discarded -- leaving
# the packed regime, the one the scheduler actually runs at ISL 1024, with
# nothing measured behind it. Env-settable so a short-prompt harvest can buy
# the extra waves.
_PACKED_PROBE_REPEATS = max(
    1, int(os.environ.get("INFERASIM_PACKED_PROBE_REPEATS", "3") or 3))
# Below this the difference is comparable to run-to-run TTFT noise and the slope
# is not resolvable.
_PREFILL_MIN_TOKEN_DELTA = 256


def prefill_probe_lengths(args) -> list[int]:
    """Prompt lengths to probe, shortest first, ending at the anchor's own.

    The long point is ``input_len`` itself so the rate is interpolated over the
    lengths the anchor is used at rather than extrapolated past them.

    Two points buy a slope and nothing else, which is why they are not enough:
    prefill is not linear in prompt length (attention is quadratic in it), so a
    two-point chord hides all of that curvature in its intercept, and that
    intercept is then carried across parallelism as though it were fixed cost.
    ``--prefill-anchor-points`` asks for enough lengths to separate the two.
    """
    long_len = int(args.input_len)
    short = int(getattr(args, "prefill_anchor_short", 0) or 0) or long_len // 2
    short = max(1, min(short, long_len - _PREFILL_MIN_TOKEN_DELTA))
    want = int(getattr(args, "prefill_anchor_points", 0) or 0)
    if want >= 3:
        # Evenly spaced rather than geometric: the quadratic term needs leverage
        # at the long end, where a geometric ladder puts almost no points.
        step = (long_len - short) / (want - 1)
        lengths = sorted({int(round(short + i * step)) for i in range(want)})
        # A ladder so tight that adjacent differences are noise measures noise.
        if all(b - a >= _PREFILL_MIN_TOKEN_DELTA
               for a, b in zip(lengths, lengths[1:])):
            return lengths
        print(f"[inferasim:Inference:Serving] WARNING: {want} probe points over "
              f"{short}..{long_len} tokens would space them under "
              f"{_PREFILL_MIN_TOKEN_DELTA} apart, which is TTFT noise; falling "
              f"back to the two-point chord.")
    lengths = [short, long_len]
    if getattr(args, "prefill_anchor_validate", False):
        # A third, interior point turns the assumption into something checkable:
        # with one pair the slope is whatever two numbers say, with two pairs
        # their disagreement measures how far from linear the prompt curve is.
        lengths.insert(1, (short + long_len) // 2)
    return lengths


def _fit_prefill_curve(pts) -> dict | None:
    """Least-squares ``F + a*n + b*n^2`` over probe points, or None.

    Returned so the projector can shard ``a`` and ``b`` -- both per-token work,
    which splits across ranks -- while holding ``F``, which does not. That is
    the whole point of fitting a curve instead of a chord: a chord's intercept
    is part fixed cost and part swallowed curvature, and the two scale with
    parallelism completely differently.

    Needs four points, not three. Three determine a parabola exactly, leaving
    no residual to judge the fit by, and a parabola through three noisy points
    can put ``F`` anywhere.
    """
    if len(pts) < 4:
        return None
    n = float(len(pts))
    sx = sum(float(x) for x, _ in pts)
    sx2 = sum(float(x) ** 2 for x, _ in pts)
    sx3 = sum(float(x) ** 3 for x, _ in pts)
    sx4 = sum(float(x) ** 4 for x, _ in pts)
    sy = sum(float(y) for _, y in pts)
    sxy = sum(float(x) * float(y) for x, y in pts)
    sx2y = sum(float(x) ** 2 * float(y) for x, y in pts)
    # Normal equations for the design matrix [1, n, n^2], solved by elimination.
    # Scaled by the longest probe so the columns are comparable: raw n^4 at
    # 8192 tokens is ~4.5e15 and the pivot search is meaningless against it.
    s = max(float(x) for x, _ in pts)
    m = [[n, sx / s, sx2 / s ** 2, sy],
         [sx / s, sx2 / s ** 2, sx3 / s ** 3, sxy / s],
         [sx2 / s ** 2, sx3 / s ** 3, sx4 / s ** 4, sx2y / s ** 2]]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(3):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, 4):
                m[r][k] -= f * m[c][k]
    fixed, a, b = (m[i][3] / m[i][i] for i in range(3))
    a, b = a / s, b / s ** 2

    def model(x):
        return fixed + a * x + b * x * x

    resid = [abs(model(x) - y) for x, y in pts]
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for _, y in pts)
    ss_res = sum((model(x) - y) ** 2 for x, y in pts)
    return {
        "fixed_ms": fixed,
        "ms_per_token": a,
        "ms_per_token_sq": b,
        "max_resid_ms": max(resid),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def packed_prefill_probe(port: int, args, out_dir: str, length: int) -> dict | None:
    """Marginal prefill cost per token when a step packs many sequences.

    The length probe cannot supply this, and the reason is a confound rather
    than a lack of precision. It varies one prompt's length at concurrency 1,
    so the number of tokens in the step and the attention context of the
    sequence inside it are the same number; GEMM efficiency, which improves as
    the step gets wider, and attention, which grows with context, therefore
    move together and no fit over those points can tell them apart. The
    ``F + a*n + b*n^2`` fit resolves the pair it is given -- R2 0.999, and it
    reproduces single-prompt TTFT at 1024 tokens to 130.0 ms against 133.7
    measured -- but it attributes to ``b`` whatever small-step GEMM
    inefficiency is present, and leaves ``a`` at a rate typical of the narrow
    end of the sweep.

    That misattribution is invisible while a step holds one sequence and grows
    with how many it holds, which is exactly the error shape in the corpus:
    DeepSeek-V4-Flash-0731 at ISL 8192, where an 16384-token budget packs two
    sequences, lands within 5% on TTFT at every concurrency from 2 to 256,
    while the same anchor at ISL 1024, where the same budget packs sixteen,
    over-reads TTFT by 39% at 8 concurrent and 157% at 256.

    So this probe holds per-sequence length fixed and varies how many arrive at
    once. p99 TTFT over a single wave of ``S`` simultaneous requests is the
    time until the last one has its first token, which is the whole wave's
    prefill plus the same constant admission/streaming/client floor the length
    probe cancels -- so differencing across ``S`` cancels it here too, and the
    slope against ``S * length`` is the per-token cost in the packed regime.
    Returns None when the slope does not resolve, which leaves the projector on
    the single-sequence curve and warning about it rather than on a guess.
    """
    want = int(getattr(args, "prefill_packed_points", 0) or 0)
    if want < 2:
        return None
    # Bounded by the step the engine can actually build: past the token budget
    # the wave spills into a second step and the slope stops being one step's
    # marginal cost. Powers of two up to that bound, which is also how the
    # batch sweep is spaced.
    budget = int(getattr(args, "max_num_batched_tokens", 0) or 0) or 16384
    counts, s = [], 1
    while len(counts) < want and s * length <= budget:
        counts.append(s)
        s *= 2
    if len(counts) < 2:
        print(f"[inferasim:Inference:Serving] packed prefill probe skipped: a "
              f"{budget}-token budget holds under two {length}-token "
              f"sequences, so there is no packing to measure here.")
        return None

    _run_client(port, args, out_dir, "packed_warmup", batch=counts[-1],
                input_len=length, output_len=_PREFILL_PROBE_OUTPUT_LEN,
                num_prompts=counts[-1])
    pts = []
    for s in counts:
        # The wave has to be exactly ``s`` requests for its last first-token to
        # mean "this wave's prefill is done" -- refilling the loop would put
        # steady-state queueing into the same number. That caps the sample
        # count at ``s``, which for small ``s`` makes p99 the max of a handful
        # and lets one slow request set the point. So the wave is repeated and
        # the median taken across repeats: the quantity stays a wave
        # completion, and it stops being a single draw from the tail.
        seen = []
        for rep in range(_PACKED_PROBE_REPEATS):
            doc = _run_client(port, args, out_dir, f"packed_S{s}_r{rep}",
                              batch=s, input_len=length,
                              output_len=_PREFILL_PROBE_OUTPUT_LEN,
                              num_prompts=s)
            # p99 rather than mean: the wave's last first-token is what bounds
            # the whole packed prefill. At one request the two coincide.
            last = doc.get("p99_ttft_ms")
            if last is None:
                last = doc.get("mean_ttft_ms")
            seen.append(float(last))
        last = statistics.median(seen)
        pts.append((s * length, last))
        print(f"[inferasim:Inference:Serving] packed prefill probe S={s} "
              f"({s * length} tok) last-TTFT={last:.2f}ms "
              f"(median of {[round(v, 1) for v in seen]})")

    n = len(pts)
    mean_x = sum(x for x, _ in pts) / n
    mean_y = sum(y for _, y in pts) / n
    var = sum((x - mean_x) ** 2 for x, _ in pts)
    rate = (sum((x - mean_x) * (y - mean_y) for x, y in pts) / var) if var else 0.0
    if rate <= 0.0:
        print(f"[inferasim:Inference:Serving] WARNING: packed prefill slope is "
              f"{rate:.6f} ms/token -- the wave did not get slower as it got "
              f"wider, which is not a prefill curve. Leaving the packed term "
              f"unmeasured; check prefix caching and the warmup.")
        return None
    # A positive overall slope is not enough, because least squares will report
    # one through points that are not a curve at all. Three vLLM harvests came
    # back with the last point *below* the one before it -- 481.6, 532.9 then
    # 334.8 ms across 2, 4 and 8 sequences -- and still fitted a tidy positive
    # rate, which would have been used as a measurement. The observable is p99
    # over exactly as many requests as there are sequences, so at S=8 it is the
    # max of eight samples and carries the tail of a co-scheduled engine's
    # admission rather than one step's cost.
    #
    # Rejected rather than smoothed. The projector's fallback is the
    # single-sequence curve plus a warning naming this as the unmeasured term,
    # which is a worse prediction but an honest one; a rate fitted through
    # non-monotonic points is neither.
    falling = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
               if pts[i + 1][1] <= pts[i][1]]
    if falling:
        print(f"[inferasim:Inference:Serving] WARNING: packed prefill probe is "
              f"not monotonic -- "
              f"{', '.join(f'{a[0]}tok={a[1]:.1f}ms then {b[0]}tok={b[1]:.1f}ms' for a, b in falling)}"
              f". A wider step cannot be cheaper, so this is scheduler tail "
              f"rather than step cost, and the fitted "
              f"{rate * 1000:.1f} us/token is not a measurement of it. "
              f"Discarding. Raise the request count per point so p99 is not "
              f"the max of a handful of samples.")
        return None
    return {
        "method": "p99 TTFT difference across simultaneous sequence count",
        "seq_len": length,
        "budget_tokens": budget,
        # Recorded so a reader can tell a measurement from a single draw. Each
        # point's p99 is taken over exactly ``seqs`` requests, so without
        # repeats it is the max of a handful and one slow request sets it.
        "repeats": _PACKED_PROBE_REPEATS,
        "output_len": _PREFILL_PROBE_OUTPUT_LEN,
        "points": [{"step_tokens": x, "seqs": x // length, "last_ttft_ms": y}
                   for x, y in pts],
        "ms_per_token": rate,
        "implied_fixed_ms": mean_y - rate * mean_x,
    }


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
    # Discarded warmup at the longest length before anything is recorded.
    #
    # Without it the first measured point pays for one-time work the others do
    # not -- graph capture and JIT for the largest prompt shape it has seen --
    # and since the probe walks lengths shortest-first that cost lands on the
    # shortest prompt. The difference then reads as prefill getting *cheaper*
    # with more tokens: a GLM-5.2-MXFP4 harvest recorded 284.51 ms at 4096
    # tokens against 193.04 at 6144, giving a negative slope, and the rate was
    # correctly rejected -- leaving that model's TTFT on the analytical
    # roofline, which is where its error came from. Warming at the longest
    # length covers every shorter one, so a single extra client run fixes it.
    _run_client(port, args, out_dir, "prefill_warmup", batch=1,
                input_len=max(lengths),
                output_len=_PREFILL_PROBE_OUTPUT_LEN,
                num_prompts=_PREFILL_PROBE_PROMPTS)
    print(f"[inferasim:Inference:Serving] prefill warmup at L={max(lengths)} "
          f"complete (discarded)")
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
    curve = _fit_prefill_curve(pts)
    if curve:
        diag["curve_fit"] = curve
        # The reason the curve is worth the extra probes, stated in numbers the
        # reader can check: how much of the linear fit's "fixed" cost was really
        # curvature. Whatever the gap is, the old path was carrying it across
        # parallelism as though it did not shard.
        lin_fixed = diag["implied_fixed_ms"]
        print(f"[inferasim:Inference:Serving] prefill curve: "
              f"{curve['fixed_ms']:.1f} ms fixed + "
              f"{curve['ms_per_token'] * 1000:.2f} us/token + "
              f"{curve['ms_per_token_sq'] * 1e6:.4f} us/token^2 "
              f"(R2={curve['r2']:.4f}, max resid {curve['max_resid_ms']:.2f} ms); "
              f"the two-point chord called {lin_fixed:.1f} ms of this fixed.")
        # Only worth probing once the curve exists, since the packed rate is
        # read as a correction to its per-token terms rather than on its own.
        #
        # The shortest probed length is the default because it is what lets the
        # most sequences into one step, but short is also where the packing
        # signal is weakest: at 256 tokens the four points land within a few ms
        # of each other on top of a 60-80 ms floor, and on DeepSeek-R1-0528,
        # gpt-oss-120b and Qwen3-14B-FP8 the ladder came back non-monotonic and
        # was discarded, leaving the packed regime unmeasured. It is also not
        # the regime being billed: a run at ISL 1024 packs 1024-token
        # sequences, and probing at 256 makes the rate lean on the curve's
        # quadratic to be re-centred four times further than it need be.
        # Overridable so a harvest can probe at the length it will be used at,
        # as long as the budget still holds two of them.
        _pk_len = int(os.environ.get("INFERASIM_PACKED_PROBE_SEQ_LEN", "0") or 0)
        packed = packed_prefill_probe(
            port, args, out_dir, _pk_len if _pk_len > 0 else min(lengths))
        if packed:
            diag["packed"] = packed
            _l0 = int(packed.get("seq_len") or min(lengths))
            single = curve["ms_per_token"] + curve["ms_per_token_sq"] * _l0
            print(f"[inferasim:Inference:Serving] packed prefill: "
                  f"{packed['ms_per_token'] * 1000:.2f} us/token at "
                  f"{_l0}-token sequences, against "
                  f"{single * 1000:.2f} us/token read off the single-sequence "
                  f"curve at the same length "
                  f"({single / packed['ms_per_token']:.2f}x).")
    rates = [p["ms_per_token"] for p in pairwise]
    # A negative pairwise slope means TTFT fell as the prompt grew, which is
    # not a prefill curve at all. Reported before the spread check rather than
    # inside it: spread is a ratio, so it was only computed when every rate was
    # positive, which suppressed the diagnostic in exactly the case that needed
    # it and left the harvest looking merely unlucky.
    negative = [p for p in pairwise if p["ms_per_token"] <= 0]
    if negative:
        diag["negative_pairs"] = negative
        print(f"[inferasim:Inference:Serving] WARNING: TTFT fell as the prompt "
              f"grew over {len(negative)} of {len(pairwise)} length pairs "
              f"({[round(p['ms_per_token'], 6) for p in negative]} ms/token). "
              f"That is not a prefill curve. The usual causes are a cold first "
              f"probe (one-time graph capture or JIT billed to the shortest "
              f"prompt) or prefix caching serving a later probe from an earlier "
              f"one's tokens. Re-harvest with the warmup pass and "
              f"prefix caching off before trusting any prefill number here.")
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


from .performance import _prefix_caching_from_server_args  # noqa: E402


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
            # Recorded explicitly, not left to be re-derived from the flag
            # string by every reader. A prefill measured against a warm prefix
            # cache times a block lookup rather than prompt processing, so a
            # consumer that cannot tell which it holds has to refuse the curve
            # and simulate TTFT instead.
            "prefix_caching": _prefix_caching_from_server_args(
                args.server_args or ""),
            # Which decode observable the sweep holds. Artifacts harvested
            # before this key recorded mean TPOT, which carries the prefill
            # stalls the simulator also schedules, so a reader that cannot tell
            # the two apart double-counts them -- worth 31% of TPOT at 64
            # concurrent. Absence of this key means "mean_tpot", and the
            # projector says so out loud rather than assuming it was told.
            "decode_observable": "median_itl_single_wave",
            "env_overrides": dict(kv.split("=", 1) for kv in args.env or []) or None,
            "attention_backend": _server_arg_value(args.server_args or "", "--attention-backend"),
            # Reported, not asserted. These were hardcoded to "auto"/True
            # regardless of what the harvest actually did, so an anchor taken
            # with --load-format dummy claimed real weights -- and for an MoE
            # that is the difference between a router that spreads tokens
            # evenly over every expert and one that concentrates them on a few.
            # Measured on GLM-5.2-MXFP4, random weights read decode 1.337x too
            # fast at batch 64 and 1.045x at batch 4, the gap widening with
            # batch exactly as expert imbalance would.
            "load_format": getattr(args, "load_format", None) or "auto",
            "real_weights": (getattr(args, "load_format", None) or "auto")
            not in ("dummy", "random"),
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
