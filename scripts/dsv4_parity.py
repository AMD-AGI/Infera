#!/usr/bin/env python3
"""Assert the Rust DeepSeek-V4 encoder still renders what sglang's does.

`rust/router/src/encoding_dsv4.rs` is a hand port of
`sglang.srt.entrypoints.openai.encoding_dsv4`. A port that drifts from its
original does not fail loudly: the router simply hashes a prefix the engine
never produced, and kv-aware routing settles at a permanent 0% hit rate with
nothing in the logs. Freezing a snapshot of the expected output would only
prove the port matched on the day it was written, so compare against whatever
sglang is installed instead, and fail the build when the two disagree.

sglang's module is pure stdlib, so it loads straight off disk without importing
the sglang package (no torch, no GPU). `pip install --no-deps sglang` is enough.

Usage: scripts/dsv4_parity.py [--sglang-encoder PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Cases are written here rather than vendored from the model repo: they only
# need to reach every branch that could plausibly drift, and both sides read the
# same list, so a case can never go stale against one implementation only.
CASES: list[dict] = [
    {
        "name": "plain chat",
        "thinking_mode": "chat",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ],
    },
    {
        "name": "no system turn",
        "thinking_mode": "chat",
        "messages": [{"role": "user", "content": "hi"}],
    },
    {
        "name": "multi turn",
        "thinking_mode": "chat",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ],
    },
    {
        "name": "thinking keeps only the latest reasoning",
        "thinking_mode": "thinking",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "reasoning_content": "stale"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "yes", "reasoning_content": "fresh"},
        ],
    },
    {
        "name": "tool call",
        "thinking_mode": "chat",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "SF"}',
                        },
                    }
                ],
            },
        ],
    },
    {
        "name": "tool result",
        "thinking_mode": "chat",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
    },
    {
        "name": "system tools",
        "thinking_mode": "chat",
        "messages": [
            {
                "role": "system",
                "content": "S",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "description": "d",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            {"role": "user", "content": "hi"},
        ],
    },
    {
        "name": "cjk and escapes",
        "thinking_mode": "chat",
        "messages": [
            {"role": "system", "content": "中文"},
            {"role": "user", "content": 'quote " backslash \\ newline \n'},
        ],
    },
]


def find_sglang_encoder(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit(
            "sglang not importable and --sglang-encoder not given.\n"
            "Install just the sources with: pip install --no-deps sglang"
        )
    root = Path(next(iter(spec.submodule_search_locations)))
    path = root / "srt" / "entrypoints" / "openai" / "encoding_dsv4.py"
    if not path.is_file():
        sys.exit(f"sglang is installed but has no dsv4 encoder at {path}")
    return path


def load_reference(path: Path):
    """Load sglang's encoder off disk, bypassing the sglang package import."""
    spec = importlib.util.spec_from_file_location("sglang_encoding_dsv4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def rust_renderings(cases: list[dict]) -> list[str | None]:
    stdin = "\n".join(
        json.dumps({"messages": c["messages"], "thinking_mode": c["thinking_mode"]})
        for c in cases
    )
    proc = subprocess.run(
        ["cargo", "run", "--quiet", "--example", "dsv4_render"],
        cwd=REPO_ROOT / "rust",
        input=stdin,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"rust renderer failed:\n{proc.stderr}")
    return [json.loads(line) for line in proc.stdout.splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-encoder", default=None, help="path to sglang's encoding_dsv4.py")
    args = ap.parse_args()

    encode_messages = load_reference(find_sglang_encoder(args.sglang_encoder))
    rust = rust_renderings(CASES)
    if len(rust) != len(CASES):
        sys.exit(f"rust returned {len(rust)} renderings for {len(CASES)} cases")

    failures = 0
    for case, got in zip(CASES, rust):
        try:
            want = encode_messages(case["messages"], thinking_mode=case["thinking_mode"])
        except Exception as exc:  # noqa: BLE001 - a raising reference is a real mismatch
            want = f"<raised {type(exc).__name__}: {exc}>"
        if got == want:
            print(f"ok   {case['name']}")
            continue
        failures += 1
        print(f"FAIL {case['name']}")
        print(f"  sglang: {want!r}")
        print(f"  rust  : {got!r}")

    if failures:
        print(
            f"\n{failures}/{len(CASES)} cases diverge. The Rust port in "
            "rust/router/src/encoding_dsv4.rs no longer matches the installed sglang "
            "encoder; port the upstream change across before merging."
        )
        return 1
    print(f"\nall {len(CASES)} cases match the installed sglang encoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
