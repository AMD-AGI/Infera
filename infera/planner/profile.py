###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Offline profiling sweep: ``python -m infera.planner.profile``.

The planner cannot invent a performance model. Before the fleet takes traffic,
this drives **one prefill replica and one decode replica** through a sweep and
writes the ``profile.json`` that ``python -m infera.planner --profile-results``
reads.

Two sweeps, matching the two things the planner has to predict.

**Prefill** walks a list of prompt lengths, one request at a time so nothing
queues, and records the TTFT of an unqueued request at each length plus the
prefill throughput that implies (``isl / ttft``).

**Decode** walks a ``context_length x kv_usage`` grid. KV utilisation is not
something a client can ask for directly, so it is reached through concurrency:
holding ``c`` requests of ``L`` tokens each in flight occupies
``c * L / max_kv_tokens`` of the cache, so the concurrency that lands on a
target utilisation is ``u * max_kv_tokens / L``. Firing exactly that many
requests at once and measuring the resulting ITL fills one grid cell, which is
why the output is a full rectangle rather than scattered samples -- the planner
interpolates it with numpy alone.

Aggregate decode throughput comes from the steady state rather than from wall
time: ``c`` requests each emitting one token per ITL sustain ``c / itl``
tokens/s, which excludes the prefill phase that a wall-time measurement would
fold in.

Run it against a deployment of exactly one replica per role, and re-run it
after anything that moves the curves: a different engine version, quantisation,
tensor-parallel width, or GPU model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger("infera.planner.profile")

_DONE = "[DONE]"


@dataclass
class Reply:
    """One streamed completion, timed."""

    ttft: float
    total: float
    tokens: int
    prompt_tokens: int

    @property
    def itl(self) -> float:
        """Seconds per token after the first, or 0 for a one-token reply."""
        return (self.total - self.ttft) / (self.tokens - 1) if self.tokens > 1 else 0.0


class PromptFactory:
    """Builds prompts of an exact token length, unique on every call.

    Uniqueness matters more than it looks: a repeated prompt hits the prefix
    cache, and a cached prefill measures the cache rather than the engine.
    """

    def __init__(self, tokenizer_source: str | None) -> None:
        self._tok = self._load(tokenizer_source)
        self._rng = random.Random(0xC0FFEE)

    @staticmethod
    def _load(source: str | None):
        if not source:
            return None
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - any failure means approximate mode
            logger.warning(
                "could not load a tokenizer from %s (%s: %s); prompt lengths will be "
                "approximate, so the recorded ISL axis will not be exact",
                source,
                type(exc).__name__,
                exc,
            )
            return None

    def build(self, n_tokens: int) -> str:
        """A prompt of (as close as possible to) ``n_tokens`` tokens."""
        words = [f"{self._rng.randrange(1 << 30):x}" for _ in range(max(1, n_tokens))]
        text = " ".join(words)
        if self._tok is None:
            return text
        # Decoding random ids gives no control over how they re-tokenise, so
        # build long and truncate to the exact count instead.
        ids = self._tok.encode(text, add_special_tokens=False)
        while len(ids) < n_tokens:
            text += " " + " ".join(f"{self._rng.randrange(1 << 30):x}" for _ in range(n_tokens))
            ids = self._tok.encode(text, add_special_tokens=False)
        return self._tok.decode(ids[:n_tokens])


class Engine:
    """A streaming OpenAI-compatible completions client."""

    def __init__(self, base_url: str, model: str, *, timeout: float = 600.0) -> None:
        self._url = base_url.rstrip("/") + "/v1/completions"
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, prompt: str, max_tokens: int) -> Reply:
        """Send one streaming request and time its tokens.

        ``ignore_eos`` keeps the reply at exactly ``max_tokens``, which is what
        makes a decode grid cell hold its concurrency for the whole measurement
        instead of draining as short replies finish early.
        """
        body = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        }
        start = time.perf_counter()
        ttft = 0.0
        frames = 0
        reported_tokens = 0
        prompt_tokens = 0
        async with self._client.stream("POST", self._url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == _DONE:
                    break
                try:
                    frame = json.loads(payload)
                except ValueError:
                    continue
                usage = frame.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or 0) or prompt_tokens
                    reported_tokens = int(usage.get("completion_tokens") or 0) or reported_tokens
                choices = frame.get("choices") or []
                if choices and choices[0].get("text"):
                    frames += 1
                    if not ttft:
                        ttft = time.perf_counter() - start
        total = time.perf_counter() - start
        if not ttft:
            raise RuntimeError("the engine returned no tokens; is --model correct?")
        # Frame counting assumes one token per frame; usage supersedes it when
        # the engine reports it, which is what stream_options asks for.
        return Reply(
            ttft=ttft,
            total=total,
            tokens=reported_tokens or frames,
            prompt_tokens=prompt_tokens,
        )


async def sweep_prefill(
    engine: Engine, prompts: PromptFactory, isls: list[int], repeats: int, num_gpu: int
) -> dict:
    """One request at a time at each prompt length, so nothing queues."""
    ttft_ms: list[float] = []
    thpt: list[float] = []
    for isl in isls:
        samples: list[float] = []
        for i in range(repeats + 1):
            reply = await engine.complete(prompts.build(isl), max_tokens=1)
            if i:  # the first is warmup
                samples.append(reply.ttft)
            if reply.prompt_tokens and abs(reply.prompt_tokens - isl) > max(8, isl * 0.02):
                logger.warning(
                    "asked for a %d-token prompt but the engine counted %d; the recorded "
                    "ISL axis will be off unless --tokenizer matches the served model",
                    isl,
                    reply.prompt_tokens,
                )
        median = statistics.median(samples)
        ttft_ms.append(median * 1000.0)
        thpt.append(isl / median / num_gpu)
        logger.info("prefill isl=%-6d ttft=%7.1fms  %8.0f tok/s/gpu", isl, ttft_ms[-1], thpt[-1])
    return {"isl": isls, "ttft_ms": ttft_ms, "thpt_per_gpu": thpt}


async def sweep_decode(
    engine: Engine,
    prompts: PromptFactory,
    context_lengths: list[int],
    kv_usages: list[float],
    *,
    osl: int,
    max_kv_tokens: int,
    num_gpu: int,
) -> dict:
    """Fill the ``context_length x kv_usage`` grid by driving concurrency.

    A cell at utilisation ``u`` and context length ``L`` is reached by holding
    ``u * max_kv_tokens / L`` requests in flight. Each request is sized so its
    mean context length over the generation is ``L``: the model's context axis
    is ``isl + osl/2``, so the prompt is ``L - osl/2`` tokens long.
    """
    itl_grid: list[list[float]] = []
    thpt_grid: list[list[float]] = []
    for ctx in context_lengths:
        isl = max(16, ctx - osl // 2)
        itl_row: list[float] = []
        thpt_row: list[float] = []
        for usage in kv_usages:
            concurrency = max(1, round(usage * max_kv_tokens / ctx))
            replies = await asyncio.gather(
                *(engine.complete(prompts.build(isl), max_tokens=osl) for _ in range(concurrency))
            )
            itl_s = statistics.mean(r.itl for r in replies)
            if itl_s <= 0:
                raise RuntimeError(
                    f"decode at context_length={ctx}, kv_usage={usage} produced replies of "
                    f"one token; raise --decode-osl"
                )
            itl_row.append(itl_s * 1000.0)
            thpt_row.append(concurrency / itl_s / num_gpu)
            logger.info(
                "decode  ctx=%-6d kv=%.2f (c=%-4d) itl=%6.2fms  %8.0f tok/s/gpu",
                ctx,
                usage,
                concurrency,
                itl_row[-1],
                thpt_row[-1],
            )
        itl_grid.append(itl_row)
        thpt_grid.append(thpt_row)
    return {
        "kv_usage": kv_usages,
        "context_length": context_lengths,
        "itl_ms": itl_grid,
        "thpt_per_gpu": thpt_grid,
        "max_kv_tokens": max_kv_tokens,
    }


def _int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m infera.planner.profile",
        description="Sweep one prefill and one decode replica, and write the profile.json "
        "the SLA planner reads.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the Infera server in front of the replica (default: %(default)s).",
    )
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        metavar="PATH_OR_HF_ID",
        help="Tokenizer used to build prompts of an exact token length. Defaults to "
        "--model; without a loadable one, prompt lengths are approximate.",
    )
    parser.add_argument(
        "--output", default="profile.json", metavar="PATH", help="Where to write the results."
    )
    parser.add_argument(
        "--max-kv-tokens",
        type=int,
        required=True,
        metavar="N",
        help="The engine's total KV cache capacity in tokens (GPU blocks x block size; "
        "the engine prints it at startup). Concurrency is derived from this.",
    )
    parser.add_argument(
        "--isl",
        type=_int_list,
        default=[512, 1024, 2048, 4096],
        metavar="A,B,C",
        help="Prompt lengths to sweep for prefill, ascending (default: %(default)s).",
    )
    parser.add_argument(
        "--prefill-repeats",
        type=int,
        default=3,
        metavar="N",
        help="Timed requests per prefill point; the median is kept (default: %(default)s).",
    )
    parser.add_argument(
        "--context-length",
        type=_int_list,
        default=[1024, 4096, 16384],
        metavar="A,B,C",
        help="Context lengths to sweep for decode, ascending (default: %(default)s).",
    )
    parser.add_argument(
        "--kv-usage",
        type=_float_list,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        metavar="A,B,C",
        help="KV utilisation fractions to sweep, ascending (default: %(default)s).",
    )
    parser.add_argument(
        "--decode-osl",
        type=int,
        default=256,
        metavar="N",
        help="Tokens generated per decode request (default: %(default)s).",
    )
    parser.add_argument(
        "--prefill-engine-num-gpu",
        type=int,
        default=1,
        metavar="N",
        help="GPUs in the prefill replica being profiled (default: %(default)s).",
    )
    parser.add_argument(
        "--decode-engine-num-gpu",
        type=int,
        default=1,
        metavar="N",
        help="GPUs in the decode replica being profiled (default: %(default)s).",
    )
    parser.add_argument("--log-level", default="INFO")

    ns = parser.parse_args(argv)
    for name, values in (("--isl", ns.isl), ("--context-length", ns.context_length)):
        if not values or any(b <= a for a, b in zip(values, values[1:])):
            parser.error(f"{name} must be a non-empty, strictly ascending list")
        if any(value <= 0 for value in values):
            parser.error(f"{name} values must be positive")
    if not ns.kv_usage or any(b <= a for a, b in zip(ns.kv_usage, ns.kv_usage[1:])):
        parser.error("--kv-usage must be a non-empty, strictly ascending list")
    if any(not 0.0 < usage <= 1.0 for usage in ns.kv_usage):
        parser.error("--kv-usage values must be in (0, 1]")
    if ns.max_kv_tokens <= 0:
        parser.error("--max-kv-tokens must be positive")
    if ns.prefill_repeats <= 0:
        parser.error("--prefill-repeats must be positive")
    if ns.decode_osl < 2:
        parser.error("--decode-osl must be at least 2, or there is no ITL to measure")
    if ns.prefill_engine_num_gpu <= 0 or ns.decode_engine_num_gpu <= 0:
        parser.error("--prefill-engine-num-gpu and --decode-engine-num-gpu must be positive")
    return ns


async def _run(ns: argparse.Namespace) -> None:
    engine = Engine(ns.url, ns.model)
    prompts = PromptFactory(ns.tokenizer or ns.model)
    try:
        prefill = await sweep_prefill(
            engine, prompts, ns.isl, ns.prefill_repeats, ns.prefill_engine_num_gpu
        )
        decode = await sweep_decode(
            engine,
            prompts,
            ns.context_length,
            ns.kv_usage,
            osl=ns.decode_osl,
            max_kv_tokens=ns.max_kv_tokens,
            num_gpu=ns.decode_engine_num_gpu,
        )
    finally:
        await engine.aclose()

    document = {
        "prefill": prefill,
        "decode": decode,
        "prefill_engine_num_gpu": ns.prefill_engine_num_gpu,
        "decode_engine_num_gpu": ns.decode_engine_num_gpu,
    }
    # Validate before writing, so a bad sweep fails here rather than at planner
    # startup: the two go through the same loader.
    from infera.planner.profile_data import parse_profile_data

    parse_profile_data(document)
    Path(ns.output).write_text(json.dumps(document, indent=2), encoding="utf-8")
    logger.info("wrote %s", ns.output)


def main() -> None:
    ns = parse_args()
    logging.basicConfig(level=ns.log_level, format="%(asctime)s %(levelname)s: %(message)s")
    # One line per request would bury the sweep in its own transport chatter.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        asyncio.run(_run(ns))
    except KeyboardInterrupt:
        logger.info("profiling stopped")


if __name__ == "__main__":
    main()
