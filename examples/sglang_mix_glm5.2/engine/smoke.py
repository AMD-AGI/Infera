#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Correctness checks through the router.

Checks the worker registration, a completion, a chat answer, a streamed chat
answer, and a long needle-in-a-haystack prompt sent twice (cold, then prefix
reuse). Every response is saved to OUT_DIR; exits non-zero if any check fails.

Usage: smoke.py ROUTER_URL MODEL OUT_DIR
Env:   LONG_PREFIX_REPEATS  filler lines of the long prompt (default 6000, ~66k tokens)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def request(
    url: str, payload: dict | None = None, headers: dict | None = None
) -> tuple[str, float]:
    """GET (no payload) or POST JSON; return the body and the elapsed seconds."""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json", **(headers or {})})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        body = response.read().decode()
    return body, time.monotonic() - start


def stream_text(body: str) -> str:
    """Join the content deltas of an SSE chat stream; require its [DONE] marker."""
    lines = [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]
    if lines[-1:] != ["[DONE]"]:
        raise AssertionError("SSE stream did not end with [DONE]")
    chunks = [json.loads(line) for line in lines[:-1]]
    return "".join(
        choice.get("delta", {}).get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )


def main() -> int:
    router, model, out_dir = sys.argv[1:]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    base = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def record(
        name: str, body: str, elapsed: float, text: str, expected: str, exact: bool = False
    ) -> None:
        (out / f"{name}.txt").write_text(body)
        result = {"name": name, "elapsed_s": round(elapsed, 3), "text": text, "expected": expected}
        result["passed"] = text == expected if exact else expected in text
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    def chat(name: str, content: str, expected: str, headers: dict | None = None) -> None:
        body, elapsed = request(
            f"{router}/v1/chat/completions",
            {**base, "messages": [{"role": "user", "content": content}]},
            headers,
        )
        text = json.loads(body)["choices"][0]["message"]["content"] or ""
        record(name, body, elapsed, text, expected)

    try:
        # One mixed worker with its four DP-attention ranks.
        body, elapsed = request(f"{router}/v1/workers")
        workers = json.loads(body)["workers"]
        found = " ".join(f"{w.get('disagg_mode')}/dp{w.get('dp_size')}" for w in workers)
        record("workers", body, elapsed, found, "mixed/dp4", exact=True)

        body, elapsed = request(
            f"{router}/v1/completions",
            {**base, "prompt": "The capital of France is", "max_tokens": 16},
        )
        record("completion", body, elapsed, json.loads(body)["choices"][0]["text"], "Paris")

        chat("chat", "What is 17 * 23? Reply with the number only.", "391")

        body, elapsed = request(
            f"{router}/v1/chat/completions",
            {
                **base,
                "messages": [
                    {"role": "user", "content": "What is 29 * 31? Reply with the number only."}
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        record("stream", body, elapsed, stream_text(body), "899")

        # The second send must hit the prefix cache; compare the two elapsed_s.
        # One session id keeps both on the same DP rank under session affinity.
        repeats = int(os.getenv("LONG_PREFIX_REPEATS", "6000"))
        filler = "This is a background record with no secret value.\n" * repeats
        half = len(filler) // 2
        prompt = (
            f"{filler[:half]}\nThe secret code is CEDAR-4729.\n{filler[half:]}"
            "\nWhat is the secret code? Return only the code."
        )
        for name in ("long_prefix_cold", "long_prefix_repeat"):
            chat(name, prompt, "CEDAR-4729", {"x-session-id": "smoke-long-prefix"})
    finally:
        (out / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    failed = [r["name"] for r in results if not r["passed"]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed", flush=True)
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
