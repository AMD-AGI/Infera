#!/usr/bin/env python3
"""Run direct HTTP smoke checks against the multi-P/D router."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class SmokeError(RuntimeError):
    pass


def request(
    url: str, *, timeout: float, payload: dict[str, Any] | None = None
) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data, headers=headers), timeout=timeout
        ) as response:
            body = response.read()
    except (OSError, urllib.error.HTTPError) as exc:
        raise SmokeError(f"request failed for {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")


def message(response: Any) -> dict[str, Any]:
    try:
        value = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SmokeError(f"invalid chat completion response: {response!r}") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"chat message is not an object: {value!r}")
    return value


def workers(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = (
            payload.get("workers")
            or payload.get("data")
            or payload.get("instances")
            or []
        )
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise SmokeError("/v1/workers did not return a worker list")
    return payload


def role(worker: dict[str, Any]) -> str:
    return str(
        worker.get("disagg_mode") or worker.get("role") or worker.get("mode") or ""
    ).lower()


def run(
    base_url: str,
    model: str,
    *,
    prefill_count: int,
    decode_count: int,
    timeout: float,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    artifacts: dict[str, Any] = {}
    artifacts["health"] = request(base_url + "/health", timeout=timeout)

    worker_payload = request(base_url + "/v1/workers", timeout=timeout)
    discovered = workers(worker_payload)
    actual_counts = {
        "prefill": sum(role(item) == "prefill" for item in discovered),
        "decode": sum(role(item) == "decode" for item in discovered),
    }
    expected_counts = {"prefill": prefill_count, "decode": decode_count}
    if actual_counts != expected_counts:
        raise SmokeError(
            f"worker counts differ: expected={expected_counts}, actual={actual_counts}"
        )
    artifacts["workers"] = worker_payload

    factual = request(
        base_url + "/v1/chat/completions",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Complete this fact with only the planet name: "
                        "The largest planet in the Solar System is"
                    ),
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
        },
    )
    factual_message = message(factual)
    text = str(factual_message.get("content") or "") + str(
        factual_message.get("reasoning_content") or ""
    )
    if "jupiter" not in text.lower() or "\ufffd" in text:
        raise SmokeError(f"factual response failed: {factual_message!r}")
    artifacts["factual"] = factual

    tool_result = request(
        base_url + "/v1/chat/completions",
        timeout=timeout,
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use get_weather to check the weather in Paris.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "max_tokens": 512,
            "temperature": 0,
        },
    )
    calls = message(tool_result).get("tool_calls") or []
    if not calls or calls[0].get("function", {}).get("name") != "get_weather":
        raise SmokeError(f"tool-call response failed: {calls!r}")
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SmokeError(f"tool-call arguments are invalid: {calls[0]!r}") from exc
    if str(arguments.get("city", "")).lower() != "paris":
        raise SmokeError(f"tool-call city mismatch: {arguments!r}")
    artifacts["tool_call"] = tool_result

    burst_payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Name one prime number larger than 100."}
        ],
        "max_tokens": 96,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda _: message(
                    request(
                        base_url + "/v1/chat/completions",
                        timeout=timeout,
                        payload=burst_payload,
                    )
                ),
                range(24),
            )
        )
    artifacts["concurrent_burst"] = {
        "requests": 24,
        "concurrency": 8,
        "status": "pass",
    }
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-count", required=True, type=int)
    parser.add_argument("--decode-count", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.prefill_count, args.decode_count, args.timeout) <= 0:
        raise SystemExit("counts and timeout must be positive")
    try:
        artifacts = run(
            args.url,
            args.model,
            prefill_count=args.prefill_count,
            decode_count=args.decode_count,
            timeout=args.timeout,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, value in artifacts.items():
            suffix = "json" if not isinstance(value, str) else "txt"
            text = (
                json.dumps(value, indent=2, sort_keys=True) + "\n"
                if suffix == "json"
                else value + ("" if value.endswith("\n") else "\n")
            )
            (args.output_dir / f"{name}.{suffix}").write_text(text, encoding="utf-8")
    except SmokeError as exc:
        print(f"smoke: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"smoke: PASS ({args.output_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
