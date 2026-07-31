#!/usr/bin/env python3
"""Load probe for the VLM multimodal-affinity routing example.

Sends N OpenAI chat/completions requests that all carry the SAME image (a valid
solid-colour PNG built with the stdlib, inlined as a data: URI so the worker
needs no network). Under the kv-aware policy these should co-locate on one
prefill worker (image affinity); under round-robin they split evenly. The
routing verdict is read from the router log by run.sh — this script only drives
the load and reports HTTP success.

Usage: mm_probe.py --url http://HOST:8000/v1/chat/completions --n 20
"""

import argparse
import base64
import json
import struct
import sys
import urllib.request
import zlib


def solid_png(r: int, g: int, b: int, side: int = 64) -> bytes:
    """Minimal solid-colour RGB PNG, stdlib only (no Pillow)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = (b"\x00" + bytes([r, g, b]) * side) * side  # per-row filter byte + pixels
    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def request_body(model: str, image: str, max_tokens: int) -> dict:
    return {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Name the dominant colour in one word."},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ],
    }


def post(url: str, body: dict, timeout: float = 120.0) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
        return resp.status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="qwen-vl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    image = data_uri(solid_png(220, 20, 60))  # one hot image, reused every request
    ok = 0
    for i in range(args.n):
        try:
            ok += post(args.url, request_body(args.model, image, args.max_tokens)) == 200
        except Exception as exc:  # noqa: BLE001
            print(f"  request {i}: ERROR {exc}", file=sys.stderr)
    print(f"{ok}/{args.n} requests ok")
    return 0 if ok == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
