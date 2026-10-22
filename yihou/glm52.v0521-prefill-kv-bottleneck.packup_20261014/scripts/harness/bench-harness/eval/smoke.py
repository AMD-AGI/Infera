#!/usr/bin/env python3
"""Purpose: Run direct HTTP health, discovery, chat, tool, and burst checks.
Usage:
    python3 eval/smoke.py --url URL --model MODEL --prefill-count N
    --decode-count N --output-dir DIR [--timeout SECONDS]
Artifacts:
    One JSON file per smoke check.
Artifact paths:
    OUTPUT_DIR/{health,workers,factual,tool_call,concurrent_burst}.json.

Run direct HTTP smoke checks against a multi-P/D router.

Usage: smoke.py --url URL --model MODEL --prefill-count N --decode-count N --output-dir DIR
"""

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, timeout: float, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, headers=headers), timeout=timeout
        ) as response:
            return json.loads(response.read())
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc


def message(response):
    try:
        value = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"invalid chat response: {response!r}") from exc
    return value


def worker_role(worker) -> str:
    return str(
        worker.get("disagg_mode") or worker.get("role") or worker.get("mode") or ""
    ).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-count", required=True, type=int)
    parser.add_argument("--decode-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if min(args.prefill_count, args.decode_count, args.timeout) <= 0:
        raise SystemExit("counts and timeout must be positive")
    base = args.url.rstrip("/")
    artifacts = {}
    try:
        artifacts["health"] = request(base + "/health", args.timeout)
        payload = request(base + "/v1/workers", args.timeout)
        workers = payload
        if isinstance(payload, dict):
            workers = payload.get("workers") or payload.get("data") or payload.get("instances") or []
        if not isinstance(workers, list):
            raise RuntimeError("/v1/workers did not return a list")
        actual = {
            "prefill": sum(worker_role(item) == "prefill" for item in workers),
            "decode": sum(worker_role(item) == "decode" for item in workers),
        }
        expected = {"prefill": args.prefill_count, "decode": args.decode_count}
        if actual != expected:
            raise RuntimeError(f"worker counts differ: expected={expected}, actual={actual}")
        artifacts["workers"] = payload

        factual = request(base + "/v1/chat/completions", args.timeout, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with only the largest planet in the Solar System."}],
            "max_tokens": 512, "temperature": 0,
        })
        factual_message = message(factual)
        text = str(factual_message.get("content") or "") + str(factual_message.get("reasoning_content") or "")
        if "jupiter" not in text.lower() or "\ufffd" in text:
            raise RuntimeError(f"factual response failed: {factual_message!r}")
        artifacts["factual"] = factual

        tool = request(base + "/v1/chat/completions", args.timeout, {
            "model": args.model,
            "messages": [{"role": "user", "content": "Use get_weather to check Paris."}],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }}],
            "tool_choice": "required", "max_tokens": 512, "temperature": 0,
        })
        calls = message(tool).get("tool_calls") or []
        if not calls or calls[0].get("function", {}).get("name") != "get_weather":
            raise RuntimeError(f"tool-call response failed: {calls!r}")
        arguments = json.loads(calls[0]["function"]["arguments"])
        if str(arguments.get("city", "")).lower() != "paris":
            raise RuntimeError(f"tool-call city mismatch: {arguments!r}")
        artifacts["tool_call"] = tool

        burst = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Name one prime number larger than 100."}],
            "max_tokens": 96,
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: message(request(base + "/v1/chat/completions", args.timeout, burst)), range(24)))
        artifacts["concurrent_burst"] = {"requests": 24, "concurrency": 8, "status": "pass"}
    except (RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"smoke: FAIL: {exc}", file=__import__("sys").stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"smoke: PASS ({args.output_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
