#!/usr/bin/env python3
"""Single-stream decode ITL benchmark for the infera_decode MoE experiment (#40).

Batch-1, temperature 0, stream=True, measure inter-token latency (ITL). Reports
median/mean ITL (ms/tok) and decode throughput (tok/s) over the streamed tokens,
discarding the first token (TTFT / prefill) so we measure decode steps only.
"""

import argparse
import json
import statistics
import time
import urllib.request


def run(port, prompt, max_tokens, warmup):
    url = f"http://127.0.0.1:{port}/v1/completions"
    body = {
        "model": "qwen35",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t_prev = None
    itls = []
    n = 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            txt = obj["choices"][0].get("text", "")
            if txt == "":
                continue
            now = time.perf_counter()
            if t_prev is not None:
                itls.append((now - t_prev) * 1000.0)
            t_prev = now
            n += 1
    wall = time.perf_counter() - t0
    # discard warmup decode steps
    body_itls = itls[warmup:] if len(itls) > warmup else itls
    return {
        "tokens": n,
        "wall_s": wall,
        "median_itl_ms": statistics.median(body_itls) if body_itls else float("nan"),
        "mean_itl_ms": statistics.mean(body_itls) if body_itls else float("nan"),
        "p10_itl_ms": statistics.quantiles(body_itls, n=10)[0]
        if len(body_itls) > 10
        else float("nan"),
        "decode_toks": len(body_itls),
        "decode_tok_s": (len(body_itls) / (sum(body_itls) / 1000.0)) if body_itls else float("nan"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8, help="decode steps to discard")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    prompt = (
        "You are a helpful assistant. Write a detailed, step-by-step explanation "
        "of how a modern GPU executes a matrix multiplication, covering memory "
        "hierarchy, tiling, and warp scheduling. Be thorough and precise.\n\nAnswer:"
    )
    results = []
    for r in range(args.reps):
        res = run(args.port, prompt, args.max_tokens, args.warmup)
        results.append(res)
        print(
            f"[{args.label}] rep{r}: tokens={res['tokens']} "
            f"median_itl={res['median_itl_ms']:.3f}ms mean_itl={res['mean_itl_ms']:.3f}ms "
            f"decode_tok/s={res['decode_tok_s']:.2f} (n_decode={res['decode_toks']})"
        )
    # aggregate across reps on the per-rep medians
    med = statistics.median([r["median_itl_ms"] for r in results])
    tps = statistics.median([r["decode_tok_s"] for r in results])
    print(f"[{args.label}] AGG median_itl={med:.3f}ms  decode_tok/s={tps:.2f}")
