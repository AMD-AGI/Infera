#!/usr/bin/env python3
"""Drive one decode worker endpoint with exact-length synthetic token IDs."""

import argparse
import concurrent.futures
import http.client
import json
import math
import random
import statistics
import time
from pathlib import Path


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def request_payload(args, prompt, room):
    if args.engine == "sglang":
        return "/generate", {
            "input_ids": prompt,
            "sampling_params": {
                "max_new_tokens": args.osl,
                "temperature": 0,
                "ignore_eos": True,
            },
            "bootstrap_host": "2.2.2.2",
            "bootstrap_room": room,
            "stream": True,
        }
    return "/v1/completions", {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.osl,
        "temperature": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }


class Generation:
    """Accumulate engine token counters and sample client-visible arrival times."""

    def __init__(self):
        self.input_tokens = None
        self.output_tokens = 0
        self.cached_tokens = None
        self.retractions = None
        self.finish_reason = None
        self.first_token = None
        self.last_token = None

    def update(self, event, engine, received):
        if event.get("error"):
            raise ValueError(f"server reported an error: {str(event['error'])[:240]}")
        if engine == "sglang":
            meta = event["meta_info"]
            reason = meta.get("finish_reason")
            if reason is not None:
                self.finish_reason = reason["type"]
            self.input_tokens = meta["prompt_tokens"]
            output_tokens = meta["completion_tokens"]
            self.retractions = meta.get("num_retractions", self.retractions)
        else:
            for choice in event.get("choices", []):
                self.finish_reason = choice.get("finish_reason") or self.finish_reason
            usage = event.get("usage")
            if usage is None:
                return
            self.input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
            details = usage.get("prompt_tokens_details")
            if details is not None:
                self.cached_tokens = details.get("cached_tokens")

        if output_tokens < self.output_tokens:
            raise ValueError("server completion count decreased")
        if output_tokens > self.output_tokens:
            if self.first_token is None:
                self.first_token = received
            self.last_token = received
        self.output_tokens = output_tokens

    def result(self, args, started):
        if self.input_tokens != args.isl or self.output_tokens != args.osl:
            raise ValueError(
                f"wrong token count: got {self.input_tokens}/{self.output_tokens}, "
                f"expected {args.isl}/{args.osl}"
            )
        if self.finish_reason != "length":
            raise ValueError(f"unexpected finish reason: {self.finish_reason!r}")
        if args.engine == "vllm" and self.cached_tokens != args.isl - 1:
            raise ValueError(f"expected {args.isl - 1} cached tokens, got {self.cached_tokens}")
        if self.retractions:
            raise ValueError(f"decode request was retracted {self.retractions} times")
        assert self.first_token is not None and self.last_token is not None
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "retractions": self.retractions,
            "ttft_ms": 1000 * (self.first_token - started),
            "tpot_ms": 1000 * (self.last_token - self.first_token) / (args.osl - 1),
            "latency_ms": 1000 * (self.last_token - started),
        }


def read_stream(response, args, started):
    if response.status != 200:
        raise ValueError(f"HTTP {response.status}: {response.read(512).decode(errors='replace')}")
    generation = Generation()
    for line in response:
        received = time.perf_counter()
        if received - started > args.timeout:
            raise TimeoutError(f"stream exceeded {args.timeout} seconds")
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            return generation.result(args, started)
        generation.update(json.loads(payload), args.engine, received)
    raise ValueError("stream ended without [DONE]")


def run_request(args, request_id, prompt_index, tokens):
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=args.timeout)
    try:
        prompt = random.Random(args.seed + prompt_index).choices(tokens, k=args.isl)
        endpoint, payload = request_payload(args, prompt, request_id)
        body = json.dumps(payload, separators=(",", ":")).encode()
        started = time.perf_counter()
        connection.request("POST", endpoint, body=body, headers={"Content-Type": "application/json"})
        with connection.getresponse() as response:
            return read_stream(response, args, started)
    except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        connection.close()


def run_point(args, concurrency, count, phase, first_room, tokens):
    started = time.perf_counter()
    results = []
    # Warmup uses a different token stream, and the measured inputs repeat for
    # each concurrency point and for each engine with the same seed.
    seed_offset = args.prompts if phase == "warmup" else 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(run_request, args, first_room + index, seed_offset + index, tokens): index
            for index in range(count)
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            emit("request", phase=phase, concurrency=concurrency, index=futures[future], **result)
    return results, time.perf_counter() - started


def preemptions(args):
    """Sum the decoder's preemption or retraction counter when available."""
    metric = {
        "sglang": b"sglang:num_retracted_requests_total",
        "vllm": b"vllm:num_preemptions_total",
    }[args.engine]
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=5)
    try:
        connection.request("GET", "/metrics")
        with connection.getresponse() as response:
            if response.status != 200:
                return None
            total = None
            for line in response.read().splitlines():
                if line == b"# TYPE " + metric + b" counter" and total is None:
                    total = 0.0
                if line.startswith(metric + b"{"):
                    value = float(line.rsplit(b"}", 1)[1].split()[0])
                elif line.startswith(metric + b" "):
                    value = float(line.split()[1])
                else:
                    continue
                if not math.isfinite(value):
                    return None
                total = (total or 0.0) + value
            return total
    except (OSError, http.client.HTTPException, ValueError, IndexError):
        return None
    finally:
        connection.close()


def summarize(results, elapsed, gpu_count):
    successes = [result for result in results if "error" not in result]
    output_tokens = sum(result["output_tokens"] for result in successes)
    summary = {
        "elapsed_s": elapsed,
        "requests_succeeded": len(successes),
        "requests_failed": len(results) - len(successes),
        "generated_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed,
        "output_tokens_per_s_per_gpu": output_tokens / elapsed / gpu_count,
    }
    for key in ("ttft_ms", "tpot_ms", "latency_ms"):
        if successes:
            values = sorted(result[key] for result in successes)
            summary[key] = {"p50": statistics.median(values), "p95": values[math.ceil(0.95 * len(values)) - 1]}
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--concurrencies", type=lambda value: [int(part) for part in value.split(",")], required=True)
    for key in ("port", "gpu-count", "isl", "osl", "prompts", "warmup", "seed", "timeout"):
        parser.add_argument("--" + key, type=int, required=True)
    args = parser.parse_args()
    if not (args.isl >= 2 and args.osl >= 2 and args.gpu_count > 0 and args.warmup >= 0 and args.timeout > 0):
        parser.error("invalid lengths, GPU count, warmup count, or timeout")
    if not all(0 < point <= args.prompts for point in args.concurrencies):
        parser.error("concurrency must be between 1 and the prompt count")
    return args


def main():
    args = parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("the checkpoint must have model_type deepseek_v41")
    vocab_size = config.get("text_config", config)["vocab_size"]
    # The low, ordinary-token region avoids the model's special added tokens.
    tokens = range(1024, min(vocab_size, 30000))
    if not tokens:
        raise ValueError("checkpoint vocab_size does not cover benchmark token IDs")
    emit(
        "config", engine=args.engine, model=args.model, isl=args.isl, osl=args.osl,
        prompts=args.prompts, warmup=args.warmup, seed=args.seed,
        gpu_count=args.gpu_count, concurrencies=args.concurrencies,
    )
    # Room 0 is reserved for SGLang's server-side health request.
    room = args.gpu_count
    for concurrency in args.concurrencies:
        if args.warmup:
            results, _ = run_point(args, concurrency, args.warmup, "warmup", room, tokens)
            room += args.warmup
            if any("error" in result for result in results):
                emit("warmup_failed", concurrency=concurrency)
                return 1
        before = preemptions(args)
        results, elapsed = run_point(args, concurrency, args.prompts, "measure", room, tokens)
        room += args.prompts
        after = preemptions(args)
        delta = after - before if before is not None and after is not None else None
        summary = summarize(results, elapsed, args.gpu_count)
        valid = not summary["requests_failed"] and (delta is None or delta == 0)
        emit("summary", concurrency=concurrency, valid=valid, preemptions_delta=delta, **summary)
        if not valid:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
