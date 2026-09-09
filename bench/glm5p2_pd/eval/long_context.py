#!/usr/bin/env python3
"""Send one long-context request directly to the router."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class LongContextError(RuntimeError):
    pass


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=250_000)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.tokens <= 0 or args.timeout <= 0:
        parser.error("--tokens and --timeout must be positive")

    try:
        payload = json.dumps(
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (" token" * args.tokens) + "\nReply OK.",
                    }
                ],
                "max_tokens": 4,
                "temperature": 0,
            }
        ).encode()
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        elapsed = time.monotonic() - started
        if not result.get("choices"):
            raise LongContextError(f"response has no choices: {result!r}")
        prompt_tokens = int((result.get("usage") or {}).get("prompt_tokens", 0))
        if prompt_tokens < args.tokens:
            raise LongContextError(
                f"server counted {prompt_tokens} prompt tokens, expected at least {args.tokens}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "response.json", result)
        write_json(
            args.output_dir / "summary.json",
            {
                "elapsed_seconds": round(elapsed, 3),
                "requested_tokens": args.tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(
                    (result.get("usage") or {}).get("completion_tokens", 0)
                ),
                "finish_reason": result["choices"][0].get("finish_reason"),
            },
        )
    except (LongContextError, OSError, ValueError, urllib.error.HTTPError) as exc:
        print(f"long_context: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"long_context: PASS ({args.output_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
