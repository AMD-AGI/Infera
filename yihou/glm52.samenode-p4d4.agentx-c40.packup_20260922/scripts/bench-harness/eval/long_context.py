#!/usr/bin/env python3
"""Purpose: Send one long-context request and record its response and timing.
Usage:
    python3 eval/long_context.py --url URL --model MODEL --output-dir DIR
    [--tokens N] [--timeout SECONDS]
Artifacts:
    Full response JSON and a timing/token summary.
Artifact paths:
    OUTPUT_DIR/response.json and OUTPUT_DIR/summary.json.

Send one long-context request directly to the router.

Usage: long_context.py --url URL --model MODEL --output-dir DIR [--tokens N]
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=250_000)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--tokens and --timeout must be positive")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": (" token" * args.tokens) + "\nReply OK."}],
        "max_tokens": 4, "temperature": 0,
    }
    try:
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        elapsed = time.monotonic() - started
        if not result.get("choices"):
            raise ValueError("response has no choices")
        choice = result["choices"][0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("response choice has no message")
        content = message.get("content")
        if not isinstance(content, str) or not re.fullmatch(
            r"\s*OK[.!]?\s*", content, re.IGNORECASE
        ):
            raise ValueError(f"unexpected response content: {content!r}")
        prompt_tokens = int((result.get("usage") or {}).get("prompt_tokens", 0))
        if prompt_tokens < args.tokens:
            raise ValueError(
                f"server counted {prompt_tokens} prompt tokens, expected at least {args.tokens}"
            )
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        print(f"long_context: FAIL: {exc}", file=__import__("sys").stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "response.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "elapsed_seconds": round(elapsed, 3),
        "requested_tokens": args.tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int((result.get("usage") or {}).get("completion_tokens", 0)),
        "finish_reason": choice.get("finish_reason"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"long_context: PASS ({args.output_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
