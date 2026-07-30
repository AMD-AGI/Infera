#!/usr/bin/env python3
"""Multimodal smoke test for Kimi-K3 (VL) on vLLM.

Sends an image + question via the OpenAI chat/completions API and prints the
answer — proves the vision encode → prefill → decode path works end to end.
Default image is a stdlib-generated solid colour (no network); pass --url for a
real image.

  python mm_test.py --port 8000
  python mm_test.py --port 8000 --url https://.../cat.jpg
"""

import argparse
import base64
import json
import struct
import sys
import urllib.request
import zlib


def solid_png(r, g, b, side=64) -> bytes:
    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = (b"\x00" + bytes([r, g, b]) * side) * side
    ihdr = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--url", help="image URL; default = a generated crimson square")
    ap.add_argument("--prompt", default="What is the dominant colour of this image? Answer in one word.")
    args = ap.parse_args()

    image = args.url or ("data:image/png;base64," + base64.b64encode(solid_png(220, 20, 60)).decode())
    body = {
        "model": args.model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    print("answer:", repr(msg.get("content")))
    if msg.get("reasoning_content"):
        print("reasoning:", repr(msg["reasoning_content"])[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
